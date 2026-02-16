from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelLoadConfig:
    model_name_or_path: str
    device: str = "cuda:0"
    device_map: Optional[str] = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = None


class HFModelAdapter:
    """
    Thin adapter around a HuggingFace causal LM providing helpers used by MIA attacks.
    Focuses on clarity and correctness over micro-optimizations.
    """

    def __init__(self, config: ModelLoadConfig):
        self.config = config
        self.device = config.device
        self.device_map = config.device_map
        self.name = config.model_name_or_path

        model_kwargs = {}
        if config.revision:
            model_kwargs["revision"] = config.revision
        if config.cache_dir:
            model_kwargs["cache_dir"] = config.cache_dir

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map=self.device_map,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=True, **({"cache_dir": config.cache_dir} if config.cache_dir else {})
        )
        # Ensure pad token exists; many causal LMs don't define it
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            try:
                self.model.resize_token_embeddings(len(self.tokenizer))
            except Exception:
                pass

        # Resolve device for single-device placement
        if self.device_map is None:
            self.model.to(self.device, non_blocking=True)

        # Basic model properties
        self.max_length = self._infer_max_length()
        self.stride = max(1, self.max_length // 2)

    def _infer_max_length(self) -> int:
        # Try common config fields; fall back to 1024
        if hasattr(self.model.config, "max_position_embeddings") and self.model.config.max_position_embeddings:
            return int(self.model.config.max_position_embeddings)
        if hasattr(self.model.config, "n_positions") and self.model.config.n_positions:
            return int(self.model.config.n_positions)
        return 1024

    def get_probabilities(
        self,
        text: str,
        tokens: Optional[np.ndarray] = None,
        no_grads: bool = True,
        return_all_probs: bool = False,
    ) -> List[float] | Tuple[List[float], torch.Tensor]:
        """
        Return token-level log probabilities for observed tokens in `text`.
        When `no_grads=False`, returns a torch tensor suitable for backprop.
        If `return_all_probs=True`, also returns the full log-softmax tensor per position.
        """
        grad_context = torch.no_grad if no_grads else torch.enable_grad
        with grad_context():
            if tokens is not None:
                labels = torch.from_numpy(tokens.astype(np.int64))
                if labels.ndim == 1:
                    labels = labels.unsqueeze(0)
            else:
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                labels = enc.input_ids

            target_token_log_prob: List[float] | List[torch.Tensor] = []
            all_token_log_prob: List[torch.Tensor] = []

            # Sliding window over long sequences
            for i in range(0, labels.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, labels.size(1))
                trg_len = end_loc - i

                input_ids = labels[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                outputs = self.model(input_ids, labels=target_ids)
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                shift_labels = target_ids[..., 1:].contiguous()
                labels_processed = shift_labels[0]

                for j, token_id in enumerate(labels_processed):
                    if token_id != -100:
                        lp = log_probs[0, j, token_id]
                        target_token_log_prob.append(lp if not no_grads else lp.item())
                        all_token_log_prob.append(log_probs[0, j])

                del input_ids
                del target_ids

            assert len(target_token_log_prob) == labels.size(1) - 1
            all_token_log_prob_tensor = torch.stack(all_token_log_prob, dim=0)

        if return_all_probs:
            if no_grads:
                return target_token_log_prob, all_token_log_prob_tensor
            else:
                # Ensure the first return is a tensor when grads are enabled
                target_tensor = torch.stack(target_token_log_prob)  # type: ignore[arg-type]
                return target_tensor, all_token_log_prob_tensor
        else:
            if no_grads:
                return target_token_log_prob  # type: ignore[return-value]
            else:
                # Return a tensor when gradients are enabled
                return torch.stack(target_token_log_prob)

    @torch.no_grad()
    def get_ll(self, text: str, tokens: Optional[np.ndarray] = None, probs: Optional[List[float]] = None) -> float:
        all_prob = probs if probs is not None else self.get_probabilities(text, tokens=tokens)  # type: ignore[arg-type]
        return -float(np.mean(all_prob))

    @torch.no_grad()
    def get_lls(self, texts: List[str], batch_size: int = 8) -> List[float]:
        losses: List[float] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            tokenized = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask

            needs_sliding = input_ids.size(1) > self.max_length // 2

            if not needs_sliding:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

            per_sample_logprobs: List[List[float]] = [[] for _ in range(input_ids.size(0))]
            for i in range(0, input_ids.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, input_ids.size(1))
                trg_len = end_loc - i

                ids = input_ids[:, begin_loc:end_loc]
                mask = attention_mask[:, begin_loc:end_loc]
                if needs_sliding:
                    ids = ids.to(self.device)
                    mask = mask.to(self.device)

                target_ids = ids.clone()
                target_ids[:, :-trg_len] = -100

                logits = self.model(ids, labels=target_ids, attention_mask=mask).logits
                shift_logits = logits[..., :-1, :].contiguous()
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                shift_labels = target_ids[..., 1:].contiguous()

                for bi in range(shift_labels.size(0)):
                    for bj in range(shift_labels.size(1)):
                        token_id = shift_labels[bi, bj]
                        if token_id != -100 and token_id != self.tokenizer.pad_token_id:
                            per_sample_logprobs[bi].append(log_probs[bi, bj, token_id].item())

                del ids
                del mask

            losses.extend([-float(np.mean(x)) if len(x) > 0 else 0.0 for x in per_sample_logprobs])

        return losses


class HFReferenceModelAdapter(HFModelAdapter):
    """
    Identical to HFModelAdapter but intended for clarity where a separate
    reference model is used in an attack.
    """

    pass


