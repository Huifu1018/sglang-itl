"""SGLang spec-v1 worker for TOKEN_ITL heterogeneous draft proposals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import monotonic
from typing import Optional

import torch

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import generate_token_bitmask

try:
    from sglang.srt.observability.req_time_stats import set_time_batch
    from sglang.srt.observability.trace import get_global_tracing_enabled
except ModuleNotFoundError:

    def set_time_batch(*args, **kwargs) -> None:
        return None

    def get_global_tracing_enabled() -> bool:
        return False

from .candidates import build_linear_candidate_rows
from .config import TokenITLSGLangConfig
from .proposer import DraftProposal, HFDraftProposer

logger = logging.getLogger(__name__)


@dataclass
class TokenITLWorkerStats:
    batches: int = 0
    verify_batches: int = 0
    target_only_batches: int = 0
    requests: int = 0
    proposed_proxy_tokens: int = 0
    accepted_draft_tokens: int = 0
    evicted_requests: int = 0


def _result_field_names() -> set[str]:
    return set(getattr(GenerationBatchResult, "__dataclass_fields__", {}))


def _make_generation_result(
    *,
    logits_output,
    next_token_ids,
    accepted_tokens: int = 0,
    accepted_per_req_cpu: list[int] | None = None,
    can_run_cuda_graph: bool = False,
    accept_lens=None,
) -> GenerationBatchResult:
    fields = _result_field_names()
    kwargs = {
        "logits_output": logits_output,
        "next_token_ids": next_token_ids,
        "can_run_cuda_graph": can_run_cuda_graph,
        "accept_lens": accept_lens,
    }
    if "num_correct_drafts" in fields:
        kwargs["num_correct_drafts"] = accepted_tokens
        kwargs["num_correct_drafts_per_req_cpu"] = accepted_per_req_cpu
    else:
        kwargs["num_accepted_tokens"] = accepted_tokens
        kwargs["accept_length_per_req_cpu"] = accepted_per_req_cpu
    return GenerationBatchResult(
        **{key: value for key, value in kwargs.items() if key in fields}
    )


def _spec_tensor(spec_info: object, modern_name: str, legacy_name: str):
    if hasattr(spec_info, modern_name):
        return getattr(spec_info, modern_name)
    return getattr(spec_info, legacy_name)


def _accept_lengths(verify_input: object):
    if hasattr(verify_input, "num_correct_drafts"):
        return getattr(verify_input, "num_correct_drafts")
    return getattr(verify_input, "accept_length", None)


def _accept_lens_for_result(verify_input: object):
    if hasattr(verify_input, "num_accept_tokens"):
        return getattr(verify_input, "num_accept_tokens")
    return getattr(verify_input, "accept_length", None)


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
        self.max_draft_token_num = int(server_args.speculative_num_draft_tokens)
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.pad_token_id = self._resolve_pad_token_id(target_worker.tokenizer)

        self.config = TokenITLSGLangConfig.from_env(default_draft_device=self.device)
        if self.config.disable_cuda_graph and hasattr(server_args, "disable_cuda_graph"):
            server_args.disable_cuda_graph = True

        self.proposer = HFDraftProposer(
            draft_model_path=server_args.speculative_draft_model_path,
            target_tokenizer=target_worker.tokenizer,
            config=self.config,
            trust_remote_code=bool(server_args.trust_remote_code),
        )
        self.stats = TokenITLWorkerStats()
        self._last_metrics_log_time = monotonic()
        logger.info(
            "Initialized TOKEN_ITL worker: draft=%s, max_draft_tokens=%s, device=%s",
            server_args.speculative_draft_model_path,
            self.max_draft_token_num,
            self.device,
        )

    def clear_cache_pool(self):
        self.proposer.clear()

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
        self.stats.batches += 1
        self.stats.requests += batch.batch_size()
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
            return _make_generation_result(
                logits_output=batch_result.logits_output,
                next_token_ids=batch_result.next_token_ids,
                accepted_tokens=0,
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
        accepted_tokens = 0
        accept_lens = None
        accepted_per_req_cpu = None

        if model_worker_batch.forward_mode.is_target_verify():
            self.stats.verify_batches += 1
            if batch.has_grammar:
                retrieve_next_token = _spec_tensor(
                    spec_info, "retrieve_next_token", "retrive_next_token"
                )
                retrieve_next_sibling = _spec_tensor(
                    spec_info, "retrieve_next_sibling", "retrive_next_sibling"
                )
                retrieve_next_token_cpu = retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = retrieve_next_sibling.cpu()
                draft_tokens_cpu = spec_info.draft_token.view(
                    retrieve_next_token.shape
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
                    retrieve_next_token = _spec_tensor(
                        verify_input, "retrieve_next_token", "retrive_next_token"
                    )
                    vocab_mask = vocab_mask.to(retrieve_next_token.device)
                    batch.sampling_info.vocab_mask = None

            logits_output, next_token_ids, accepted_tokens = verify_input.verify(
                batch,
                logits_output,
                self.page_size,
                vocab_mask,
            )
            accept_lengths = _accept_lengths(verify_input)
            accepted_per_req_cpu = (
                accept_lengths.cpu().tolist() if accept_lengths is not None else None
            )
            if accepted_per_req_cpu is not None:
                self.stats.accepted_draft_tokens += sum(accepted_per_req_cpu)

            if get_global_tracing_enabled():
                for idx, req in enumerate(batch.reqs):
                    correct = (
                        accept_lengths[idx].item()
                        if accept_lengths is not None
                        else 0
                    )
                    if hasattr(req.time_stats, "set_spec_verify_end_time"):
                        req.time_stats.set_spec_verify_end_time(
                            num_correct_drafts=correct
                        )

            accept_lens = _accept_lens_for_result(verify_input)
            if batch.return_logprob:
                add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)
            self._evict_finished_requests(batch)
            self._maybe_log_metrics()
            batch.forward_mode = ForwardMode.DECODE
        else:
            self.stats.target_only_batches += 1
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return _make_generation_result(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            accepted_tokens=accepted_tokens,
            accepted_per_req_cpu=accepted_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
        )

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch) -> None:
        bs = batch.batch_size()
        candidate_rows = self._build_candidate_rows(batch)
        rows = candidate_rows.rows
        draft_token_num = candidate_rows.draft_token_num
        draft_token = torch.tensor(
            [token for row in rows for token in row],
            dtype=torch.int64,
            device=self.device,
        )

        retrieve_index = torch.arange(
            bs * draft_token_num,
            dtype=torch.int64,
            device=self.device,
        ).reshape(bs, draft_token_num)
        next_row = torch.arange(
            1,
            draft_token_num + 1,
            dtype=torch.int64,
            device=self.device,
        )
        next_row[-1] = -1
        retrieve_next_token = next_row.unsqueeze(0).repeat(bs, 1)
        retrieve_next_sibling = torch.full(
            (bs, draft_token_num),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        offsets = torch.arange(
            draft_token_num,
            dtype=torch.int64,
            device=self.device,
        )
        positions = (batch.seq_lens.to(torch.int64).unsqueeze(1) + offsets).reshape(-1)

        linear_mask = torch.tril(
            torch.ones(
                (draft_token_num, draft_token_num),
                dtype=torch.bool,
                device=self.device,
            )
        )
        custom_mask_parts = []
        for i in range(bs):
            prefix_len = int(batch.seq_lens_cpu[i].item())
            prefix_mask = torch.ones(
                (draft_token_num, prefix_len),
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
            draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _build_candidate_rows(self, batch: ScheduleBatch):
        roots: list[int] = []
        proxy_rows: list[tuple[int, ...]] = []
        max_proxy_tokens = self.max_draft_token_num - 1
        for req in batch.reqs:
            root = self._root_token(req)
            roots.append(root)
            proposal = DraftProposal((), (), None, "skipped", 0)
            if getattr(req, "multimodal_inputs", None) is None:
                try:
                    current_text = self._current_text(req)
                    proposal = self.proposer.propose(
                        str(req.rid),
                        current_text,
                        max_proxy_tokens=max_proxy_tokens,
                    )
                except Exception:
                    logger.exception("TOKEN_ITL proposal failed for request %s", req.rid)

            proxy_rows.append(proposal.proxy_target_token_ids[:max_proxy_tokens])
        candidate_rows = build_linear_candidate_rows(
            roots,
            proxy_rows,
            max_draft_token_num=self.max_draft_token_num,
        )
        self.stats.proposed_proxy_tokens += candidate_rows.proposed_proxy_tokens
        return candidate_rows

    def _evict_finished_requests(self, batch: ScheduleBatch) -> None:
        finished_req_ids = [
            str(req.rid)
            for req in batch.reqs
            if req.finished() or getattr(req, "is_retracted", False)
        ]
        if finished_req_ids:
            self.proposer.evict(finished_req_ids)
            self.stats.evicted_requests += len(finished_req_ids)

    def _maybe_log_metrics(self) -> None:
        interval = self.config.metrics_log_interval
        if interval is None:
            return
        now = monotonic()
        if now - self._last_metrics_log_time < interval:
            return
        self._last_metrics_log_time = now
        proposer_stats = self.proposer.stats.snapshot()
        logger.info(
            "TOKEN_ITL metrics: worker=%s proposer=%s cache_size=%s",
            self.stats,
            proposer_stats,
            self.proposer.cache_size(),
        )

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
