"""V4 command-line interface (python -m v4 <cmd>)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from v4.core.config import config_hash, load_config, merge_overrides
from v4.core.logging import configure, get_logger
from v4.core.paths import OUTPUTS_ROOT
from v4.core.registry import build_fusion, import_all
from v4.core.seed import seed_all
from v4.data.datasets.cached import CachedFeatureDataset, collate_cached
from v4.data.manifest import manifest_sha256

log = get_logger(__name__)


def _pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _build_full_model(cfg) -> torch.nn.Module:
    import_all()
    return build_fusion(dict(cfg.fusion))


def train_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="v4 train")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    configure("INFO")
    overrides: dict = {}
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.out_dir is not None:
        overrides["train.out_dir"] = args.out_dir
    if args.epochs is not None:
        overrides["train.epochs"] = int(args.epochs)

    cfg = load_config(args.config)
    if overrides:
        cfg = merge_overrides(cfg, overrides)
    seed_all(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = _pick_device()
    log.info("device=%s seed=%d stage=%s", device, cfg.seed, cfg.stage)

    data_cfg = cfg.data
    cache_root = data_cfg.get("cache_root", "cache/v4")
    feature_keys = data_cfg.get("feature_keys", [])
    csv_path = data_cfg["csv_path"]

    train_ds = CachedFeatureDataset(csv_path=csv_path, cache_root=cache_root,
                                    feature_keys=feature_keys, split="train", limit=args.limit)
    val_ds = CachedFeatureDataset(csv_path=csv_path, cache_root=cache_root,
                                  feature_keys=feature_keys, split="val", limit=args.limit)

    bs = int(cfg.train.get("batch_size", 64))
    nw = int(cfg.train.get("num_workers", 2))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              collate_fn=collate_cached, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            collate_fn=collate_cached, persistent_workers=nw > 0)

    model = _build_full_model(cfg)
    log.info("trainable params: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    cfg_hash = config_hash(cfg)
    try:
        data_hash = manifest_sha256(csv_path)
    except FileNotFoundError:
        log.warning("manifest %s not found", csv_path)
        data_hash = "no-manifest"

    from v4.training.trainer import BaseTrainer
    trainer = BaseTrainer(model=model, train_loader=train_loader, val_loader=val_loader,
                          cfg=cfg, device=device, config_hash=cfg_hash, data_hash=data_hash)
    summary = trainer.fit()
    log.info("done. best F1=%.4f at ep %d", summary["best_f1_macro"], summary["best_epoch"])
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="v4 eval")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test",
                   choices=["test", "val", "train", "mmfakebench-transfer"])
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    configure("INFO")
    cfg = load_config(args.config)
    device = _pick_device()

    csv_path = cfg.data["csv_path"]
    cache_root = cfg.data.get("cache_root", "cache/v4")
    feature_keys = cfg.data.get("feature_keys", [])
    split = "test" if args.split == "mmfakebench-transfer" else args.split

    ds = CachedFeatureDataset(csv_path=csv_path, cache_root=cache_root,
                              feature_keys=feature_keys, split=split)
    if args.split == "mmfakebench-transfer":
        ds.df = ds.df[ds.df["source"].str.startswith("MMFakeBench_")].reset_index(drop=True)
        log.info("MMFakeBench transfer: %d rows", len(ds.df))

    loader = DataLoader(ds, batch_size=int(cfg.train.get("batch_size", 64)),
                        num_workers=int(cfg.train.get("num_workers", 2)),
                        collate_fn=collate_cached)

    model = _build_full_model(cfg)
    ckpt_path = args.checkpoint or str(
        Path(cfg.train.get("out_dir", "outputs/v4/run")) / "best.pt"
    )
    from v4.core.checkpoints import load_checkpoint
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info("loaded ckpt epoch=%d", ckpt.get("epoch", -1))

    out_dir = args.out_dir or str(
        OUTPUTS_ROOT / "eval" / (
            "mmfakebench_transfer" if args.split == "mmfakebench-transfer"
            else f"in_distribution_{split}"
        )
    )

    from v4.evaluation.evaluator import BaseEvaluator
    evaluator = BaseEvaluator(model=model, device=device,
                              num_classes=int(cfg.fusion.get("num_classes", 5)))
    metrics = evaluator.evaluate(loader, out_dir=out_dir)
    log.info("eval done -> %s (F1-macro=%.4f)", out_dir, metrics["f1_macro"])
    return 0


def build_manifest_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="v4 build-manifest")
    p.add_argument("--source", required=True,
                   choices=["newsclippings", "dgm4", "mmfakebench", "genimage", "stage1"])
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    configure("INFO")
    from v4.data.builders import (
        build_dgm4, build_genimage_stage1, build_mmfakebench,
        build_newsclippings, build_stage1_binary,
    )
    fn = {
        "newsclippings": build_newsclippings.build,
        "dgm4": build_dgm4.build,
        "mmfakebench": build_mmfakebench.build,
        "genimage": build_genimage_stage1.build,
        "stage1": build_stage1_binary.build,
    }[args.source]
    out = fn(args.out)
    log.info("manifest -> %s", out)
    return 0


def precompute_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="v4 precompute")
    p.add_argument("--modality", required=True,
                   choices=["v_imgfor_dct", "v_semantic_fnd", "v_textfor_qwen"])
    p.add_argument("--config", default=None)
    p.add_argument("--csv", default="data/processed/forensic_5class_v4.csv")
    p.add_argument("--cache-root", default="cache/v4")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    configure("INFO")
    from scripts.precompute import run_precompute
    out_dir = run_precompute(modality=args.modality, csv_path=args.csv,
                             cache_root=args.cache_root, config_path=args.config,
                             limit=args.limit)
    log.info("precompute -> %s", out_dir)
    return 0


def leakage_audit_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="v4 leakage-audit")
    p.add_argument("--csv", default="data/processed/forensic_5class_v4.csv")
    args = p.parse_args(argv)
    configure("INFO")
    from v4.data.builders.leakage_audit import audit
    return audit(args.csv)


def verify_paths_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="v4 verify-image-paths")
    p.add_argument("--csv", default="data/processed/forensic_5class_v4.csv")
    args = p.parse_args(argv)
    configure("INFO")
    from v4.data.builders.leakage_audit import verify_image_paths
    return verify_image_paths(args.csv)


def merge_manifest_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="v4 merge-manifest")
    p.add_argument("--out", default="data/processed/forensic_5class_v4.csv")
    args = p.parse_args(argv)
    configure("INFO")
    from v4.data.builders.merge import merge_all
    out = merge_all(args.out)
    log.info("merged manifest -> %s", out)
    return 0


def server_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="v4 server")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--config", default="app/server_config.yaml")
    args = p.parse_args(argv)
    import os
    os.environ["V4_SERVER_CONFIG"] = args.config
    import uvicorn
    uvicorn.run("app.server:app", host=args.host, port=args.port, log_level="info")
    return 0


SUBCOMMANDS = {
    "train": train_main,
    "eval": eval_main,
    "build-manifest": build_manifest_main,
    "merge-manifest": merge_manifest_main,
    "precompute": precompute_main,
    "leakage-audit": leakage_audit_main,
    "verify-image-paths": verify_paths_main,
    "server": server_main,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python -m v4 <subcommand> [args]")
        print("Subcommands: " + ", ".join(SUBCOMMANDS))
        return 0
    sub = sys.argv[1]
    if sub not in SUBCOMMANDS:
        print(f"unknown subcommand: {sub}")
        return 2
    return SUBCOMMANDS[sub](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
