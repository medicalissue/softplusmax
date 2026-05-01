"""Train script for the softmax vs softplusmax comparison.

Three tasks dispatch to one common training loop:

  cls     ResNet-56 / CIFAR-100 supervised classification
  kd      ResNet-20 student distilled from a pre-trained ResNet-56 teacher
  infonce SimCLR-style contrastive on CIFAR-100 (unlabeled)

Loss head: --loss {sm, sp}
  sm  softmax-based standard
  sp  softplusmax variant

Single seed by default; pass --seed to override.

Usage examples:
  python train.py --task cls   --loss sm
  python train.py --task cls   --loss sp
  python train.py --task kd    --loss sm --T 4 --teacher_ckpt runs/cls_sm/best.pt
  python train.py --task kd    --loss sp --teacher_ckpt runs/cls_sm/best.pt
  python train.py --task infonce --loss sm --tau 0.1
  python train.py --task infonce --loss sp --tau 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

from models import resnet20, resnet56
from models.proj_head import ProjectionHead, BackboneWithHead
from losses import (
    SoftmaxCE, SoftplusmaxCE,
    SoftmaxKD, SoftplusmaxKD,
    SoftmaxInfoNCE, SoftplusmaxInfoNCE, SqJumpReLUInfoNCE,
)

# wandb is optional; only import if --wandb is set.
try:
    import wandb
except ImportError:
    wandb = None


# ─── Diagnostics ────────────────────────────────────────────────────
@torch.no_grad()
def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


@torch.no_grad()
def logit_diagnostics(logits: torch.Tensor) -> dict:
    """For one batch of [B, C] logits, return summary stats."""
    return {
        "logit_mean": logits.mean().item(),
        "logit_std":  logits.std().item(),
        "logit_max":  logits.abs().max().item(),
        "sigmoid_mean": torch.sigmoid(logits).mean().item(),
    }


def softplusmax_prob(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Same shape, returns the softplusmax distribution."""
    sp = F.softplus(logits)
    return sp / sp.sum(-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def calibration_metrics(probs: torch.Tensor, targets: torch.Tensor,
                         n_bins: int = 15) -> dict:
    """Expected Calibration Error and overall confidence/accuracy.

    Standard binned ECE (Naeini et al. 2015 / Guo et al. 2017):
      bin samples by max prob; per bin compute |acc(bin) - conf(bin)|;
      weight by bin frequency.
    """
    confs, preds = probs.max(dim=-1)
    correct = (preds == targets).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    n = float(probs.size(0))
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (confs > lo) & (confs <= hi) if b > 0 else (confs >= lo) & (confs <= hi)
        if mask.any():
            bin_n = mask.float().sum()
            bin_acc  = correct[mask].mean()
            bin_conf = confs[mask].mean()
            ece = ece + (bin_n / n) * (bin_acc - bin_conf).abs()
    return {
        "ece":         ece.item(),
        "mean_conf":   confs.mean().item(),
        "mean_correct": correct.mean().item(),
    }


@torch.no_grad()
def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> int:
    _, idx = logits.topk(k, dim=-1)
    return (idx == target.unsqueeze(-1)).any(dim=-1).sum().item()


@torch.no_grad()
def infonce_diagnostics(z_a: torch.Tensor, z_b: torch.Tensor,
                         loss_kind: str, tau: float | None = None,
                         theta: float | None = None) -> dict:
    """Snapshot the gate distribution for one batch.

    Supported loss_kind:
      "sm"   — softmax(sim/τ)
      "sp"   — softplus(sim/τ) / Σ
      "sqjr" — [sim - θ]_+^2 / Σ  (Squared JumpReLU; θ may be learnable)

    Returns:
      pos_gate_share, gate_entropy, pos_sim, neg_sim_top10  (existing)
      n_active        — mean #classes with nonzero gate (per row)
      n_cutoff        — mean #negatives that are exactly cut off
      hard_easy_ratio — mean(gate on hard negs) / mean(gate on easy negs)
      effective_k     — exp(entropy) over the negative-only distribution
      pos_neg_gap     — pos_sim − mean(top-10% neg sim)
      theta_value     — current theta (if applicable, else 0)
    """
    B = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)            # [2B, D]
    sim = z @ z.t()                              # [2B, 2B]
    mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)

    if loss_kind == "sm":
        s = sim / (tau or 1.0)
        s = s.masked_fill(mask_self, float("-inf"))
        gate = F.softmax(s, dim=-1)
    elif loss_kind == "sqjr":
        # SqJumpReLU: cutoff on raw sim, τ amplifies magnitude only.
        th = float(theta) if theta is not None else 0.0
        f = F.relu(sim - th).pow(2).masked_fill(mask_self, 0.0)
        gate = f / f.sum(-1, keepdim=True).clamp_min(1e-30)
    else:  # sp
        s = sim / (tau or 1.0)
        sp = F.softplus(s).masked_fill(mask_self, 0.0)
        gate = sp / sp.sum(-1, keepdim=True).clamp_min(1e-8)

    pos_gate = gate.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)  # [2B]
    # Entropy normalized by log(N-1).
    g_eps = gate.clamp_min(1e-12)
    entropy = -(gate * g_eps.log()).sum(-1)
    norm_entropy = entropy / math.log(2 * B - 1)

    # Negative-only mask.
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    pos_mask.scatter_(-1, pos_idx.unsqueeze(-1), True)
    neg_mask = ~(mask_self | pos_mask)
    neg_sims = sim[neg_mask].view(2 * B, -1)
    k_top = max(1, neg_sims.size(-1) // 10)
    top_neg_sim = neg_sims.topk(k_top, dim=-1).values.mean()
    pos_sim_mean = sim.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1).mean()

    # Sparsity / cutoff stats (meaningful for sp/sqjr).
    n_active = (gate > 0).sum(-1).float().mean()
    # n_cutoff: negatives with exactly zero gate.
    neg_gate = gate.masked_fill(~neg_mask, 1.0)  # nonzero filler so we don't count pos/self
    n_cutoff = (neg_gate == 0).sum(-1).float().mean()

    # Hard vs easy negatives by similarity quartile.
    sim_d = sim.detach()
    # Per-row top-25% as hard, bottom-50% as easy (excluding self/pos).
    neg_sims_only = sim_d.masked_fill(~neg_mask, float("-inf"))
    n_neg = neg_mask.sum(-1).clamp_min(1)  # 2B-2
    k_hard = (n_neg.float() * 0.25).long().clamp_min(1)
    k_easy = (n_neg.float() * 0.50).long().clamp_min(1)
    # Hard threshold per row.
    hard_thresh = neg_sims_only.topk(k_hard.max().item(), dim=-1).values[:, -1:]
    easy_thresh = (-neg_sims_only).topk(k_easy.max().item(), dim=-1).values[:, -1:]
    easy_thresh = -easy_thresh  # values below this = easy
    hard_mask = neg_mask & (sim_d >= hard_thresh)
    easy_mask = neg_mask & (sim_d <= easy_thresh)
    n_hard = hard_mask.sum(-1).float().clamp_min(1)
    n_easy = easy_mask.sum(-1).float().clamp_min(1)
    mass_hard = (gate * hard_mask).sum(-1) / n_hard
    mass_easy = (gate * easy_mask).sum(-1) / n_easy
    hard_easy_ratio = (mass_hard.mean() / mass_easy.mean().clamp_min(1e-12))

    # Effective k over the *negative-only* distribution (renormalized).
    neg_gate_mass = (gate * neg_mask).sum(-1, keepdim=True).clamp_min(1e-30)
    p_neg = (gate * neg_mask) / neg_gate_mass
    p_neg_eps = p_neg.clamp_min(1e-12)
    H_neg = -(p_neg * p_neg_eps.log()).sum(-1)
    eff_k = H_neg.exp().mean()

    return {
        "pos_gate_share":    pos_gate.mean().item(),
        "gate_entropy":      norm_entropy.mean().item(),
        "pos_sim":           pos_sim_mean.item(),
        "neg_sim_top10":     top_neg_sim.item(),
        "pos_neg_gap":       (pos_sim_mean - top_neg_sim).item(),
        "n_active":          n_active.item(),
        "n_cutoff":          n_cutoff.item(),
        "hard_easy_ratio":   hard_easy_ratio.item(),
        "effective_k":       eff_k.item(),
        "theta_value":       float(theta) if theta is not None else 0.0,
    }


# ─── Repro ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Data ───────────────────────────────────────────────────────────
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD  = (0.2673, 0.2564, 0.2762)


def cls_transforms():
    train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return train, test


class TwoCropTransform:
    """SimCLR augmentation: returns two views per image."""

    def __init__(self):
        self.t = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    def __call__(self, x):
        return self.t(x), self.t(x)


def build_loaders(args):
    if args.task in ("cls", "kd"):
        tr_tf, te_tf = cls_transforms()
        train = torchvision.datasets.CIFAR100(args.data, train=True,
                                              download=True, transform=tr_tf)
        test  = torchvision.datasets.CIFAR100(args.data, train=False,
                                              download=True, transform=te_tf)
        tl = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        vl = DataLoader(test, batch_size=512, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        return tl, vl
    if args.task == "infonce":
        tf = TwoCropTransform()
        train = torchvision.datasets.CIFAR100(args.data, train=True,
                                              download=True, transform=tf)
        # For linear-eval pre-cache, we'll use the cls test transform when
        # evaluating; eval is run by eval.py separately.
        tl = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        return tl, None
    raise ValueError(f"unknown task {args.task}")


# ─── Model + loss factories ────────────────────────────────────────
def build_model(args, num_classes: int = 100) -> nn.Module:
    if args.task in ("cls", "kd"):
        net = (resnet20 if args.task == "kd" else resnet56)(num_classes=num_classes)
        return net
    if args.task == "infonce":
        backbone = resnet56(num_classes=num_classes, return_features=True)
        head = ProjectionHead(in_dim=64, hidden_dim=256, out_dim=128)
        return BackboneWithHead(backbone, head)
    raise ValueError(args.task)


def build_loss(args) -> nn.Module:
    if args.task == "cls":
        return SoftmaxCE() if args.loss == "sm" else SoftplusmaxCE()
    if args.task == "kd":
        if args.loss == "sm":
            return SoftmaxKD(alpha=args.kd_alpha, T=args.T)
        return SoftplusmaxKD(alpha=args.kd_alpha, T=args.T)
    if args.task == "infonce":
        if args.loss == "sm":
            return SoftmaxInfoNCE(tau=args.tau)
        if args.loss == "sqjr":
            return SqJumpReLUInfoNCE(
                theta_init=args.sqjr_theta_init,
                learnable=args.sqjr_theta_learnable,
            )
        return SoftplusmaxInfoNCE(tau=args.tau)
    raise ValueError(args.task)


def build_teacher(args, num_classes: int = 100) -> nn.Module:
    """Load a frozen ResNet-56 teacher for distillation."""
    teacher = resnet56(num_classes=num_classes)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# ─── Eval ──────────────────────────────────────────────────────────
@torch.no_grad()
def eval_full(model, loader, device, loss_kind: str, autocast_dtype=None):
    """Returns top-1, top-5, ECE for the full validation set.

    loss_kind: "sm" → ECE under softmax probs
               "sp" → ECE under softplusmax probs
    """
    model.eval()
    n_correct1 = n_correct5 = n_total = 0
    all_probs, all_targets = [], []
    use_amp = autocast_dtype is not None and device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                logits = model(x)
        else:
            logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        # Cast back to fp32 for stable softmax/softplusmax + calibration.
        logits = logits.float()
        n_correct1 += (logits.argmax(-1) == y).sum().item()
        n_correct5 += topk_correct(logits, y, k=5)
        n_total += y.numel()
        if loss_kind == "sp":
            probs = softplusmax_prob(logits)
        else:
            probs = F.softmax(logits, dim=-1)
        all_probs.append(probs)
        all_targets.append(y)
    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    cal = calibration_metrics(probs, targets)
    return {
        "top1":  100.0 * n_correct1 / max(n_total, 1),
        "top5":  100.0 * n_correct5 / max(n_total, 1),
        **cal,
    }


# ─── Train loops ───────────────────────────────────────────────────
def train_cls_or_kd(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    tl, vl = build_loaders(args)
    model = build_model(args).to(device)
    loss_fn = build_loss(args).to(device)
    teacher = None
    if args.task == "kd":
        teacher = build_teacher(args).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay,
                          nesterov=True)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)

    # bf16 + torch.compile (CUDA only). bf16 has full fp32 dynamic range, so
    # we don't need a GradScaler. Compile is "default" mode — short warm-up
    # and ~1.5-2× steady-state speedup on L4 / H100 for ResNet-CIFAR.
    use_amp = args.amp and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else None
    raw_model = model  # keep handle to the un-compiled module for state_dict
    if args.compile and device.type == "cuda":
        print(f"[compile] torch.compile(mode='{args.compile_mode}')")
        model = torch.compile(model, mode=args.compile_mode)
        if teacher is not None:
            teacher = torch.compile(teacher, mode=args.compile_mode)

    best_acc = 0.0
    last_logits = None  # for diagnostics

    for epoch in range(args.epochs):
        model.train()
        if epoch < args.warmup_epochs:
            warm_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in opt.param_groups:
                pg["lr"] = warm_lr

        running = 0.0
        n = 0
        gn_running = 0.0
        n_batches = 0
        train_correct = 0
        t0 = time.time()
        for x, y in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(x)
                        loss = loss_fn(logits, t_logits, y)
                    else:
                        loss = loss_fn(logits, y)
            else:
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = loss_fn(logits, t_logits, y)
                else:
                    loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn_running += grad_norm(model)
            n_batches += 1
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
            train_correct += (logits.detach().argmax(-1) == y).sum().item()
            last_logits = logits.detach().float()
        if epoch >= args.warmup_epochs:
            sched_cos.step()
        train_loss = running / max(n, 1)
        train_acc = 100.0 * train_correct / max(n, 1)
        avg_gn = gn_running / max(n_batches, 1)
        elapsed = time.time() - t0

        # Validation: top-1, top-5, ECE (under the model's own probability head).
        ev = eval_full(model, vl, device, loss_kind=args.loss,
                       autocast_dtype=autocast_dtype)
        is_best = ev["top1"] > best_acc
        if is_best:
            best_acc = ev["top1"]
            torch.save({"model": raw_model.state_dict(), "epoch": epoch,
                        "acc": ev["top1"]}, args.out_dir / "best.pt")
        cur_lr = opt.param_groups[0]["lr"]

        diag = logit_diagnostics(last_logits) if last_logits is not None else {}
        log_dict = dict(
            epoch=epoch, lr=cur_lr, time=elapsed, grad_norm=avg_gn,
            train_loss=train_loss, train_acc=train_acc,
            val_top1=ev["top1"], val_top5=ev["top5"],
            ece=ev["ece"], mean_conf=ev["mean_conf"],
            best_top1=best_acc,
            **diag,
        )
        print(f"ep {epoch:3d}  lr={cur_lr:.4f}  loss={train_loss:.4f}  "
              f"trn={train_acc:.2f}  top1={ev['top1']:.2f}  "
              f"top5={ev['top5']:.2f}  ECE={ev['ece']:.4f}  "
              f"best={best_acc:.2f}  gn={avg_gn:.2f}  time={elapsed:.1f}s")
        with open(args.out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(log_dict) + "\n")
        if args.wandb:
            wandb.log(log_dict, step=epoch)

    torch.save({"model": raw_model.state_dict(), "epoch": args.epochs - 1,
                "acc": ev["top1"]}, args.out_dir / "last.pt")
    if args.wandb:
        wandb.summary["best_top1"] = best_acc
        wandb.summary["final_top1"] = ev["top1"]
        wandb.summary["final_top5"] = ev["top5"]
        wandb.summary["final_ece"]  = ev["ece"]
    print(f"\nDone. best_top1 = {best_acc:.2f}  final_ece = {ev['ece']:.4f}")
    return best_acc


def train_infonce(args):
    """Pre-train ResNet-56 backbone with InfoNCE; eval is separate."""
    device = torch.device(args.device)
    set_seed(args.seed)

    tl, _ = build_loaders(args)
    model = build_model(args).to(device)
    loss_fn = build_loss(args).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay,
                          nesterov=True)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)

    use_amp = args.amp and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else None
    raw_model = model
    if args.compile and device.type == "cuda":
        print(f"[compile] torch.compile(mode='{args.compile_mode}')")
        model = torch.compile(model, mode=args.compile_mode)

    for epoch in range(args.epochs):
        model.train()
        if epoch < args.warmup_epochs:
            warm_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in opt.param_groups:
                pg["lr"] = warm_lr

        running = 0.0
        n = 0
        gn_running = 0.0
        n_batches = 0
        last_za = last_zb = None
        t0 = time.time()
        for (xa, xb), _ in tl:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    za = model(xa)
                    zb = model(xb)
                    loss = loss_fn(za, zb)
            else:
                za = model(xa)
                zb = model(xb)
                loss = loss_fn(za, zb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn_running += grad_norm(model)
            n_batches += 1
            opt.step()
            running += loss.item() * xa.size(0)
            n += xa.size(0)
            last_za, last_zb = za.detach().float(), zb.detach().float()
        if epoch >= args.warmup_epochs:
            sched_cos.step()
        train_loss = running / max(n, 1)
        avg_gn = gn_running / max(n_batches, 1)
        elapsed = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]

        # Pull current theta from the loss module if it has one (SqJumpReLU).
        cur_theta = None
        if hasattr(loss_fn, "theta"):
            t = loss_fn.theta
            cur_theta = t.item() if hasattr(t, "item") else float(t)

        diag = infonce_diagnostics(
            last_za, last_zb,
            loss_kind=args.loss,
            tau=args.tau,
            theta=cur_theta,
        ) if last_za is not None else {}

        log_dict = dict(epoch=epoch, lr=cur_lr, train_loss=train_loss,
                        time=elapsed, grad_norm=avg_gn,
                        theta=cur_theta if cur_theta is not None else 0.0,
                        **diag)
        theta_str = f"  θ={cur_theta:.3f}" if cur_theta is not None else ""
        print(f"ep {epoch:3d}  lr={cur_lr:.4f}  loss={train_loss:.4f}  "
              f"pos_g={diag.get('pos_gate_share',0):.4f}  "
              f"H={diag.get('gate_entropy',0):.3f}  "
              f"hard/easy={diag.get('hard_easy_ratio',0):.2f}  "
              f"n_cut={diag.get('n_cutoff',0):.1f}{theta_str}  "
              f"gn={avg_gn:.2f}  time={elapsed:.1f}s")
        with open(args.out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(log_dict) + "\n")
        if args.wandb:
            wandb.log(log_dict, step=epoch)

        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            torch.save({"model": raw_model.state_dict(), "epoch": epoch},
                       args.out_dir / f"ckpt_{epoch+1:04d}.pt")
    torch.save({"model": raw_model.state_dict(), "epoch": args.epochs - 1},
               args.out_dir / "last.pt")
    print("\nDone. Run eval.py with --backbone for linear eval.")
    return None


# ─── Args ──────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["cls", "kd", "infonce"], required=True)
    p.add_argument("--loss", choices=["sm", "sp", "sqjr"], required=True)
    p.add_argument("--data", default="./data")
    p.add_argument("--out_dir", default=None,
                   help="Default: runs/<task>_<loss>[_T#|_tau#]/")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # cls / kd common
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # kd
    p.add_argument("--T", type=float, default=4.0,
                   help="KD temperature for both softmax KD (z/T) and "
                        "softplusmax KD (also z/T, with × T² correction).")
    p.add_argument("--kd_alpha", type=float, default=0.1)
    p.add_argument("--teacher_ckpt", default=None)

    # infonce: SimCLR CIFAR-100 standard τ=0.5 (Chen et al. 2020 Appendix B,
    # vs τ=0.1 used for ImageNet).
    p.add_argument("--tau", type=float, default=0.5)
    # SqJumpReLU specific.
    p.add_argument("--sqjr_theta_init", type=float, default=0.0,
                   help="Initial θ for Squared JumpReLU normalization.")
    p.add_argument("--sqjr_theta_learnable",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="If true, θ is learned via SGD (no STE needed).")

    # wandb
    p.add_argument("--wandb", action="store_true",
                   help="Log to W&B (requires WANDB_API_KEY env var)")
    p.add_argument("--wandb_project", default="softplusmax")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None,
                   help="Default: derived from --task / --loss / --T / --tau / --seed")

    # AMP + compile (default ON; pass --no-amp / --no-compile to disable)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Use bf16 autocast on CUDA (full fp32 dynamic range, "
                        "no GradScaler needed). Default: on.")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="Wrap model with torch.compile after model build. "
                        "Default: on.")
    p.add_argument("--compile_mode", choices=["default", "reduce-overhead", "max-autotune"],
                   default="default")
    return p.parse_args()


def main():
    args = get_args()
    if args.out_dir is None:
        suffix = ""
        if args.task == "kd":
            suffix = f"_T{args.T:g}"
        if args.task == "infonce":
            if args.loss == "sqjr":
                lr = "L" if args.sqjr_theta_learnable else "F"
                suffix = f"_th{args.sqjr_theta_init:g}{lr}"
            else:
                suffix = f"_tau{args.tau:g}"
        args.out_dir = f"runs/{args.task}_{args.loss}{suffix}"
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(args.out_dir / "args.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()}, f, indent=2)

    print(f"task={args.task}  loss={args.loss}  out={args.out_dir}")

    # wandb init (optional)
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb requested but not installed (pip install wandb)")
        run_name = args.wandb_run_name or args.out_dir.name + f"_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()},
        )

    if args.task == "infonce":
        train_infonce(args)
    else:
        train_cls_or_kd(args)
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
