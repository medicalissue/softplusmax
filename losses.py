"""Softmax vs Softplusmax losses for classification, distillation, and InfoNCE.

All "softplusmax" variants replace exp with softplus in the
normalization step:

    softmax(z)_i      = exp(z_i)      / Σ_j exp(z_j)
    softplusmax(z)_i  = softplus(z_i) / Σ_j softplus(z_j)

Softplus is asymmetric — exp-like for z<<0, linear for z>>0 — which
gives a per-class consideration gate σ(z) = ζ'(z) on top of L1
normalization. See proposal for full derivation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Stage A: Classification ────────────────────────────────────────
class SoftmaxCE(nn.Module):
    """Standard cross-entropy: -log softmax(logits)[label]."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target)


class SoftplusmaxCE(nn.Module):
    """Cross-entropy on softplusmax (drop-in for SoftmaxCE).

    Both losses minimize -log p[y]; the only difference is the
    normalization that produces p:
        SoftmaxCE     :  p = softmax(logits)     = exp(z) / Σ exp(z)
        SoftplusmaxCE :  p = softplusmax(logits) = ζ(z)   / Σ ζ(z)
                         where ζ(z) = log(1 + e^z) = softplus(z)

    F.cross_entropy is hard-coded to softmax, so we compute the
    softplusmax variant explicitly.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sp = F.softplus(logits)                                    # [B, C]
        denom = sp.sum(-1, keepdim=True).clamp_min(self.eps)        # [B, 1]
        log_p = sp.gather(-1, target.unsqueeze(-1)).log() - denom.log()
        return -log_p.mean()


# ─── Stage B: Knowledge Distillation ───────────────────────────────
class SoftmaxKD(nn.Module):
    """Hinton-style KD with softmax(z/T) on both teacher and student.

    L = α · CE(student, label)
      + (1-α) · KL(softmax(teacher/T) || softmax(student/T)) · T²
    """

    def __init__(self, alpha: float = 0.1, T: float = 4.0):
        super().__init__()
        self.alpha = float(alpha)
        self.T = float(T)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        T = self.T
        ce = F.cross_entropy(student_logits, target)
        # log_softmax for student, softmax for teacher (KL standard form).
        log_s = F.log_softmax(student_logits / T, dim=-1)
        soft_t = F.softmax(teacher_logits / T, dim=-1)
        # KL(P||Q) = Σ P · (log P − log Q); here P=teacher (no grad), Q=student.
        kl = F.kl_div(log_s, soft_t, reduction="batchmean") * (T * T)
        return self.alpha * ce + (1.0 - self.alpha) * kl


class SoftplusmaxKD(nn.Module):
    """KD with softplusmax(z/T) — direct mirror of softmax KD.

    For a fair comparison with softmax KD T=4, we apply the same z/T
    scaling and × T² gradient correction. While ReLU's positive
    homogeneity makes c·z a no-op in the strict ReLU limit, exact
    softplus does respond to multiplicative scaling for c < 1
    (transition zone widens, negatives lifted into the active region),
    so z/T with T>1 is a meaningful softening dial.

        L = α · NLL_softplusmax(student, label)
          + (1-α) · KL(softplusmax(teacher/T) || softplusmax(student/T)) · T²
    """

    def __init__(self, alpha: float = 0.1, T: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.T = float(T)
        self.eps = eps

    @staticmethod
    def softplusmax(z: torch.Tensor, eps: float) -> torch.Tensor:
        sp = F.softplus(z)
        return sp / sp.sum(-1, keepdim=True).clamp_min(eps)

    def _nll(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sp = F.softplus(logits)
        denom = sp.sum(-1, keepdim=True).clamp_min(self.eps)
        log_p = sp.gather(-1, target.unsqueeze(-1)).log() - denom.log()
        return -log_p.mean()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        T = self.T
        # CE on raw student logits — student must learn label.
        ce = self._nll(student_logits, target)
        # KL on temperature-scaled distributions, with T² correction
        # (mirrors Hinton softmax KD).
        p_t = self.softplusmax(teacher_logits.detach() / T, self.eps)
        p_s = self.softplusmax(student_logits / T, self.eps)
        kl = (p_t * (p_t.clamp_min(self.eps).log()
                     - p_s.clamp_min(self.eps).log())).sum(-1).mean()
        return self.alpha * ce + (1.0 - self.alpha) * kl * (T * T)


# ─── Stage C: InfoNCE / Contrastive ────────────────────────────────
class SoftmaxInfoNCE(nn.Module):
    """Standard SimCLR-style InfoNCE.

    For each anchor in the batch, the positive is its augmented partner;
    negatives are the other 2(B-1) views in the batch. We compute
    similarities z_i · z_j / τ and use softmax cross-entropy.
    """

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = float(tau)

    def forward(
        self,
        z_a: torch.Tensor,            # [B, D]  anchor embeddings (l2-normalized)
        z_b: torch.Tensor,            # [B, D]  paired view embeddings
    ) -> torch.Tensor:
        B = z_a.size(0)
        # Concatenate both views, treat as 2B samples; positives are
        # off-diagonal pairs (i, i+B) and (i+B, i).
        z = torch.cat([z_a, z_b], dim=0)                # [2B, D]
        sim = z @ z.t() / self.tau                       # [2B, 2B]
        # Mask self-similarity.
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, float("-inf"))
        # Positive index: for row i (0..B-1), positive is i+B; for row
        # i (B..2B-1), positive is i-B.
        pos_idx = torch.arange(2 * B, device=z.device)
        pos_idx = (pos_idx + B) % (2 * B)
        return F.cross_entropy(sim, pos_idx)


class SoftplusmaxInfoNCE(nn.Module):
    """Softplusmax InfoNCE — direct mirror of softmax InfoNCE with τ.

    For a fair comparison with softmax InfoNCE, we apply the same s/τ
    scaling so the only difference is the function (exp vs softplus):

        sm:  p_i = exp(s_ij/τ)      / Σ exp(s_ik/τ)
        sp:  p_i = softplus(s_ij/τ) / Σ softplus(s_ik/τ)

    With L2-normalized embeddings, s ∈ [-1, 1]; dividing by τ stretches
    similarities into the active region of the chosen function.
    """

    def __init__(self, tau: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.tau = float(tau)
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        B = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)                # [2B, D]
        sim = z @ z.t() / self.tau                       # [2B, 2B]
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sp = F.softplus(sim)                             # [2B, 2B]
        sp = sp.masked_fill(mask_self, 0.0)
        denom = sp.sum(-1, keepdim=True).clamp_min(self.eps)
        pos_idx = torch.arange(2 * B, device=z.device)
        pos_idx = (pos_idx + B) % (2 * B)
        sp_pos = sp.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)
        log_p = sp_pos.clamp_min(self.eps).log() - denom.squeeze(-1).log()
        return -log_p.mean()


class SqJumpReLUInfoNCE(nn.Module):
    """Squared Jump Softplus InfoNCE: smooth sparse contrastive loss.

        f(x)  = softplus(x − θ)^2 = log²(1 + exp(x − θ))
        p_i   = f(s_ij) / Σ_k f(s_ik)

    Same family as SqJumpReLU but with softplus instead of ReLU. Hard
    ReLU cutoff has a fatal failure mode: a sample whose positive
    similarity falls at or below θ has f(s_pos) = 0, log(0) = −∞, and
    both embedding and θ gradients vanish — the sample becomes dead and
    cannot recover. Replacing ReLU with softplus gives quadratic
    amplification for x ≫ θ and exponential decay for x ≪ θ while
    keeping f strictly positive everywhere; gradients always flow.

    Because softplus removes the dead-sample failure mode, θ does not
    need to be bounded — there is no scenario where an unbounded θ
    causes catastrophic failure (only slow learning at extremes, which
    the loss self-corrects). θ is a plain unconstrained nn.Parameter,
    learned end-to-end with no STE and no reparameterization.

    No τ: the loss is scale-invariant under positive multiplicative
    rescaling of f, so a temperature would be a no-op. θ is the only
    knob.
    """

    def __init__(self, theta_init: float = 0.0, learnable: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        if learnable:
            self.theta = nn.Parameter(torch.tensor(float(theta_init)))
        else:
            self.register_buffer("theta", torch.tensor(float(theta_init)))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        B = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)                # [2B, D]
        sim = z @ z.t()                                  # [2B, 2B]  ∈ [-1, 1]
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        f = F.softplus(sim - self.theta).pow(2)
        f = f.masked_fill(mask_self, 0.0)
        denom = f.sum(-1, keepdim=True).clamp_min(self.eps)
        pos_idx = torch.arange(2 * B, device=z.device)
        pos_idx = (pos_idx + B) % (2 * B)
        f_pos = f.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)
        log_p = f_pos.clamp_min(self.eps).log() - denom.squeeze(-1).log()
        return -log_p.mean()


# ─── Sanity / smoke test ───────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, D = 8, 100, 128

    # classification
    logits = torch.randn(B, C, requires_grad=True)
    target = torch.randint(0, C, (B,))
    print("SoftmaxCE     :", SoftmaxCE()(logits, target).item())
    print("SoftplusmaxCE:", SoftplusmaxCE()(logits, target).item())

    # distillation
    s_logits = torch.randn(B, C, requires_grad=True)
    t_logits = torch.randn(B, C)
    print("SoftmaxKD     :", SoftmaxKD()(s_logits, t_logits, target).item())
    print("SoftplusmaxKD :", SoftplusmaxKD()(s_logits, t_logits, target).item())

    # InfoNCE
    z_a = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
    z_b = F.normalize(torch.randn(B, D), dim=-1)
    print("SoftmaxNCE    :", SoftmaxInfoNCE(tau=0.5)(z_a, z_b).item())
    print("SoftplusmaxNCE:", SoftplusmaxInfoNCE()(z_a, z_b).item())

    # backward sanity
    SoftplusmaxCE()(logits, target).backward()
    SoftplusmaxKD()(s_logits, t_logits, target).backward()
    SoftplusmaxInfoNCE()(z_a, z_b).backward()
    print("backward OK")
