"""Linear evaluation for InfoNCE-pre-trained backbones.

Freeze the backbone, train a linear classifier on top of penultimate
features for 100 epochs on CIFAR-100, then report test top-1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models import resnet56

try:
    import wandb
except ImportError:
    wandb = None


CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD  = (0.2673, 0.2564, 0.2762)


def get_loaders(data_root, batch_size, num_workers):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    tr = torchvision.datasets.CIFAR100(data_root, train=True, download=True,
                                       transform=train_tf)
    te = torchvision.datasets.CIFAR100(data_root, train=False, download=True,
                                       transform=test_tf)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(te, batch_size=512, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tl, vl


def load_backbone(ckpt_path: str) -> nn.Module:
    """Strip projection head; keep ResNet-56 + features."""
    backbone = resnet56(num_classes=100, return_features=True)
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("model", raw)
    # If model was BackboneWithHead, keys are prefixed "backbone.".
    bb_state = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            bb_state[k[len("backbone."):]] = v
    if not bb_state:
        # fall back: assume keys already match.
        bb_state = state
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    if missing or unexpected:
        print(f"[load_backbone] missing={len(missing)} unexpected={len(unexpected)}")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to a .pt with backbone weights")
    p.add_argument("--data", default="./data")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true",
                   help="log to W&B (requires WANDB_API_KEY)")
    p.add_argument("--wandb_project", default="softplusmax")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile_mode", default="default")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.out_dir is None:
        args.out_dir = str(Path(args.ckpt).parent / "linear_eval")
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed (pip install wandb)")
        # Run name reflects the parent infonce run.
        parent_name = Path(args.ckpt).parent.name
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=f"{parent_name}_lineval",
                   config={"ckpt": str(args.ckpt), "epochs": args.epochs,
                           "lr": args.lr, "batch_size": args.batch_size})

    tl, vl = get_loaders(args.data, args.batch_size, args.num_workers)
    backbone = load_backbone(args.ckpt).to(device)

    head = nn.Linear(64, 100).to(device)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    opt = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    use_amp = args.amp and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else None
    if args.compile and device.type == "cuda":
        backbone = torch.compile(backbone, mode=args.compile_mode)

    best = 0.0
    for ep in range(args.epochs):
        backbone.eval()
        head.train()
        t0 = time.time()
        running = 0.0; n = 0
        for x, y in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        _, feats = backbone(x)
                    feats = feats.float()
                else:
                    _, feats = backbone(x)
            logits = head(feats)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()

        head.eval()
        n_c = n_t = 0
        with torch.no_grad():
            for x, y in vl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        _, feats = backbone(x)
                    feats = feats.float()
                else:
                    _, feats = backbone(x)
                pred = head(feats).argmax(-1)
                n_c += (pred == y).sum().item()
                n_t += y.numel()
        acc = 100.0 * n_c / max(n_t, 1)
        best = max(best, acc)
        print(f"ep {ep:3d}  lr={opt.param_groups[0]['lr']:.4f}  "
              f"loss={running/max(n,1):.4f}  acc={acc:.2f}  best={best:.2f}  "
              f"time={time.time()-t0:.1f}s")
        log_dict = dict(epoch=ep, loss=running/max(n, 1),
                        acc=acc, best=best,
                        lr=opt.param_groups[0]["lr"])
        with open(args.out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(log_dict) + "\n")
        if args.wandb:
            wandb.log(log_dict, step=ep)

    print(f"\nlinear eval best = {best:.2f}")
    with open(args.out_dir / "final.json", "w") as f:
        json.dump(dict(best=best, epochs=args.epochs), f, indent=2)
    if args.wandb:
        wandb.summary["lineval_best"] = best
        wandb.finish()


if __name__ == "__main__":
    main()
