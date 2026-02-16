from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model_adapter import HFModelAdapter, HFReferenceModelAdapter


@dataclass
class AttackConfig:
    # Common controls
    pretokenized: bool = False
    max_tokens: int = 4096

    # Min-K params
    min_k_k: float = 0.2
    min_k_window: int = 1
    min_k_stride: int = 1

    # Min-K++ params
    minkpp_k: float = 0.2

    # Grad-norm
    gradnorm_p: float = float("inf")


class BaseAttack:
    def __init__(self, target: HFModelAdapter, cfg: AttackConfig, reference: Optional[HFReferenceModelAdapter] = None):
        self.target = target
        self.reference = reference
        self.cfg = cfg

    def score(self, document: str, probs: Optional[List[float]] = None, **kwargs) -> float:
        raise NotImplementedError


class LossAttack(BaseAttack):
    @torch.no_grad()
    def score(self, document: str, probs: Optional[List[float]] = None, **kwargs) -> float:
        return self.target.get_ll(document, probs=probs)


class ReferenceAttack(BaseAttack):
    @torch.no_grad()
    def score(self, document: str, probs: Optional[List[float]] = None, **kwargs) -> float:
        if self.reference is None:
            raise ValueError("ReferenceAttack requires a reference model")
        loss = self.target.get_ll(document, probs=probs)
        ref_loss = self.reference.get_ll(document, probs=probs)
        return loss - ref_loss


class ZlibAttack(BaseAttack):
    @torch.no_grad()
    def score(self, document: str, probs: Optional[List[float]] = None, **kwargs) -> float:
        import zlib

        loss = self.target.get_ll(document, probs=probs)
        z_entropy = len(zlib.compress(document.encode("utf-8")))
        return loss / float(z_entropy)


class MinKProbAttack(BaseAttack):
    @torch.no_grad()
    def score(self, document: str, probs: Optional[List[float]] = None, **kwargs) -> float:
        k = float(kwargs.get("k", self.cfg.min_k_k))
        window = int(kwargs.get("window", self.cfg.min_k_window))
        stride = int(kwargs.get("stride", self.cfg.min_k_stride))

        all_prob = probs if probs is not None else self.target.get_probabilities(document)
        ngram_probs: List[float] = []
        for i in range(0, len(all_prob) - window + 1, stride):
            ngram = all_prob[i : i + window]
            ngram_probs.append(float(np.mean(ngram)))
        min_k = sorted(ngram_probs)[: max(1, int(len(ngram_probs) * k))]
        return -float(np.mean(min_k))


class MinKPlusPlusAttack(BaseAttack):
    @torch.no_grad()
    def score(self, document: str, probs: Optional[List[float]] = None, **kwargs) -> float:
        k = float(kwargs.get("k", self.cfg.minkpp_k))

        target_prob, all_probs = self.target.get_probabilities(document, return_all_probs=True)  # type: ignore[assignment]
        # all_probs: (T, V) log-probs, target_prob: List[float]
        ap = all_probs
        mu = (torch.exp(ap) * ap).sum(-1)
        sigma = (torch.exp(ap) * ap.pow(2)).sum(-1) - mu.pow(2)
        target = torch.tensor(target_prob, device=ap.device)
        scores = (target - mu).cpu().numpy() / (sigma.sqrt().cpu().numpy() + 1e-12)
        scores = scores.tolist()
        return -float(np.mean(sorted(scores)[: max(1, int(len(scores) * k))]))


class GradNormAttack(BaseAttack):
    def score(self, document: str, probs: Optional[List[float]] = None, **kwargs) -> float:
        p = kwargs.get("p", self.cfg.gradnorm_p)
        if p not in [1, 2, float("inf")]:
            raise ValueError("p must be one of {1, 2, inf}")

        self.target.model.zero_grad(set_to_none=True)
        target_probs = self.target.get_probabilities(document, tokens=None, no_grads=False)  # type: ignore[assignment]
        if isinstance(target_probs, list):
            raise RuntimeError("Internal: expected tensor probabilities when no_grads=False")
        loss = -target_probs.mean()
        loss.backward()
        norms = []
        for param in self.target.model.parameters():
            if param.grad is not None:
                norms.append(param.grad.detach().norm(p=p))
        self.target.model.zero_grad(set_to_none=True)
        if not norms:
            return 0.0
        return -float(torch.stack(norms).mean().cpu().numpy())


def build_attack(name: str, target: HFModelAdapter, cfg: AttackConfig, reference: Optional[HFReferenceModelAdapter] = None) -> BaseAttack:
    name = name.lower()
    if name in {"loss"}:
        return LossAttack(target, cfg, reference)
    if name in {"ref", "reference", "reference_based"}:
        if reference is None:
            raise ValueError("Reference attack requires a reference model name")
        return ReferenceAttack(target, cfg, reference)
    if name == "zlib":
        return ZlibAttack(target, cfg)
    if name in {"min_k", "min-k", "min-k%", "mink"}:
        return MinKProbAttack(target, cfg)
    if name in {"min_k++", "min-k++", "mink++"}:
        return MinKPlusPlusAttack(target, cfg)
    if name in {"gradnorm", "grad-norm"}:
        return GradNormAttack(target, cfg)
    raise ValueError(f"Unknown attack: {name}")


