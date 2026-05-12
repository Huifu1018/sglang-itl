"""SGLang spec-v1 worker for Token-ITL heterogeneous draft proposals."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.observability.trace import get_global_tracing_enabled
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import generate_token_bitmask

from .config import TokenITLSGLangConfig
from .proposer import HFDraftProposer

logger = logging.getLogger(__name__)


class TokenITLWorker:
    """TokenTiming-style speculative worker for heterogeneous tokenizers.

    Candidate generation is handled by a normal HF draft model. Target
    verification, KV cache allocation/freeing, request mutation, and batching
    are delegated to SGLang's existing NGRAM verify machinery.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num = int(server_args.speculative_num_draft_tokens)
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.pad_token_id = self._resolve_pad_token_id(target_worker.tokenizer)

        config = TokenITLSGLangConfig.from_env(default_draft_device=self.device)
        if config.disable_cuda_graph and hasattr(server_args, "disable_cuda_graph"):
            server_args.disable_cuda_graph = True

        self.proposer = HFDraftProposer(
            draft_model_path=server_args.speculative_draft_model_path,
            target_tokenizer=target_worker.tokenizer,
            config=config,
            trust_remote_code=bool(server_args.trust_remote_code),
        )
        logger.info(
            "Initialized TOKEN_ITL worker: draft=%s, draft_tokens=%s, device=%s",
            server_args.speculative_draft_model_path,
            self.draft_token_num,
            self.device,
        )

    def clear_cache_pool(self):
        pass

    def update_weights_from_tensor(self, recv_req):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def add_external_corpus(self, corpus_id: str, token_chunks: list[list[int]]) -> int:
        logger.warning("TOKEN_ITL ignores NGRAM external corpus load: %s", corpus_id)
        return 0

    def commit_corpus_load(self, corpus_id: str, loaded_token_count: int) -> None:
        return None

    def remove_external_corpus(self, corpus_id: str) -> None:
        return None

    def list_external_corpora(self) -> dict[str, int]:
        return {}

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
            return GenerationBatchResult(
                logits_output=batch_result.logits_output,
                next_token_ids=batch_result.next_token_ids,
                num_correct_drafts=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        if not batch.sampling_info.is_all_greedy:
            raise ValueError(
                "TOKEN_ITL SGLang integration currently supports greedy decoding only. "
                "Use temperature=0 for OpenAI-compatible requests."
            )

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        model_worker_batch = batch.get_model_worker_batch()
        spec_info = model_worker_batch.spec_info
        num_correct_drafts = 0
        accept_lens = None
        num_correct_drafts_per_req_cpu = None

        if model_worker_batch.forward_mode.is_target_verify():
            if batch.has_grammar:
                retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
                draft_tokens_cpu = spec_info.draft_token.view(
                    spec_info.retrieve_next_token.shape
                ).cpu()

            set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch,
                is_verify=True,
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_input: NgramVerifyInput = model_worker_batch.spec_info
            vocab_mask = None
            if batch.has_grammar:
                vocab_mask = generate_token_bitmask(
                    batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    batch.sampling_info.vocab_size,
                )
                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                    batch.sampling_info.vocab_mask = None

            logits_output, next_token_ids, num_correct_drafts = verify_input.verify(
                batch,
                logits_output,
                self.page_size,
                vocab_mask,
            )
            num_correct_drafts_per_req_cpu = (
                verify_input.num_correct_drafts.cpu().tolist()
            )

            if get_global_tracing_enabled():
                for idx, req in enumerate(batch.reqs):
                    correct = (
                        verify_input.num_correct_drafts[idx].item()
                        if verify_input.num_correct_drafts is not None
                        else 0
                    )
                    req.time_stats.set_spec_verify_end_time(num_correct_drafts=correct)

            accept_lens = verify_input.num_accept_tokens
            if batch.return_logprob:
                add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)
            batch.forward_mode = ForwardMode.DECODE
        else:
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
        )

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch) -> None:
        bs = batch.batch_size()
        rows = self._build_candidate_rows(batch)
        draft_token = torch.tensor(
            [token for row in rows for token in row],
            dtype=torch.int64,
            device=self.device,
        )

        retrieve_index = torch.arange(
            bs * self.draft_token_num,
            dtype=torch.int64,
            device=self.device,
        ).reshape(bs, self.draft_token_num)
        next_row = torch.arange(
            1,
            self.draft_token_num + 1,
            dtype=torch.int64,
            device=self.device,
        )
        next_row[-1] = -1
        retrieve_next_token = next_row.unsqueeze(0).repeat(bs, 1)
        retrieve_next_sibling = torch.full(
            (bs, self.draft_token_num),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        offsets = torch.arange(
            self.draft_token_num,
            dtype=torch.int64,
            device=self.device,
        )
        positions = (batch.seq_lens.to(torch.int64).unsqueeze(1) + offsets).reshape(-1)

        linear_mask = torch.tril(
            torch.ones(
                (self.draft_token_num, self.draft_token_num),
                dtype=torch.bool,
                device=self.device,
            )
        )
        custom_mask_parts = []
        for i in range(bs):
            prefix_len = int(batch.seq_lens_cpu[i].item())
            prefix_mask = torch.ones(
                (self.draft_token_num, prefix_len),
                dtype=torch.bool,
                device=self.device,
            )
            custom_mask_parts.append(torch.cat((prefix_mask, linear_mask), dim=1).flatten())
        custom_mask = torch.cat(custom_mask_parts) if custom_mask_parts else torch.empty(0, dtype=torch.bool, device=self.device)

        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            draft_token,
            custom_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            self.draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _build_candidate_rows(self, batch: ScheduleBatch) -> list[list[int]]:
        rows: list[list[int]] = []
        max_proxy_tokens = self.draft_token_num - 1
        for req in batch.reqs:
            root = self._root_token(req)
            proxies: tuple[int, ...] = ()
            if getattr(req, "multimodal_inputs", None) is None:
                try:
                    current_text = self._current_text(req)
                    proposal = self.proposer.propose(
                        current_text,
                        max_proxy_tokens=max_proxy_tokens,
                    )
                    proxies = proposal.proxy_target_token_ids
                except Exception:
                    logger.exception("TOKEN_ITL proposal failed for request %s", req.rid)

            row = [root] + [int(token_id) for token_id in proxies[:max_proxy_tokens]]
            while len(row) < self.draft_token_num:
                row.append(self.pad_token_id)
            rows.append(row[: self.draft_token_num])
        return rows

    def _current_text(self, req: object) -> str:
        input_ids = list(getattr(req, "origin_input_ids_unpadded", None) or req.origin_input_ids)
        token_ids = input_ids + list(req.output_ids)
        tokenizer = getattr(req, "tokenizer", None) or self.target_worker.tokenizer
        try:
            return tokenizer.decode(
                [int(token_id) for token_id in token_ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode([int(token_id) for token_id in token_ids])

    @staticmethod
    def _root_token(req: object) -> int:
        if getattr(req, "output_ids", None):
            return int(req.output_ids[-1])
        return int(req.origin_input_ids[-1])

    @staticmethod
    def _resolve_pad_token_id(tokenizer: object) -> int:
        for attr in ("eos_token_id", "pad_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                return int(value)
        return 0
