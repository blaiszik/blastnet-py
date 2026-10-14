"""`blastnet` command line: convert | train | eval | eval-shipped | leaderboard.

    blastnet convert ROOT OUT [--splits ...] [--ratios 8 16 32]
    blastnet train --root DATA --name edsr --size 0.8M [--loss mse] [--aug aug|noaug] [--seed 0]
                   --out RUNS [--devices 4 --strategy ddp --accelerator gpu] [--max-time 01:20:00]
    blastnet eval WEIGHTS.pt [...] --root DATA --out rows.jsonl [--splits test forcedhit paramvar]
    blastnet eval-shipped --root DATA --out rows.jsonl [--shard i --num-shards 4] [--families ...]
    blastnet leaderboard rows.jsonl [...] [--out table.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["build_parser", "main"]


def _device(name: str | None) -> str:
    if name:
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _devices(v: str):
    return v if v == "auto" else int(v)


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for every subcommand (defaults = upstream train.py defaults)."""
    from blastnet.data import RATIOS, SPLITS
    from blastnet.eval import EVAL_SPLITS
    from blastnet.train import LOSSES, UPSTREAM

    p = argparse.ArgumentParser(prog="blastnet", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="download a public Kaggle dataset")
    from blastnet.data.kaggle import MOMENTUM128

    dl.add_argument("slug", nargs="?", default=MOMENTUM128)
    dl.add_argument("--out", required=True, help="download directory")
    dl.add_argument("--file", default=None, help="download one file from the dataset")

    cat = sub.add_parser("catalog", help="search public Kaggle datasets")
    cat.add_argument("--search", default="blastnet")
    cat.add_argument("--csv", default=None, help="write results as CSV")

    c = sub.add_parser("convert", help="convert a .dat tree to Zarr")
    c.add_argument("root")
    c.add_argument("out")
    c.add_argument("--splits", nargs="+", default=list(SPLITS))
    c.add_argument("--ratios", nargs="+", type=int, default=[8], choices=RATIOS)
    c.add_argument("--hr-size", type=int, default=128)

    t = sub.add_parser("train", help="train one SR net (upstream recipe)")
    t.add_argument("--root", required=True, help="Momentum128 .dat tree")
    t.add_argument("--name", required=True, help="model family (edsr, rcan, rrdb, gconv, ...)")
    t.add_argument("--size", required=True, help="upstream size label, e.g. 0.8M")
    t.add_argument("--ratio", type=int, default=8)
    t.add_argument("--loss", choices=LOSSES, default="mse")
    t.add_argument("--aug", choices=("aug", "noaug"), default="aug")
    t.add_argument("--seed", type=int, default=UPSTREAM["seed"])
    t.add_argument("--out", default="runs", help="runs directory (run dir = OUT/<stem>)")
    t.add_argument("--epochs", type=int, default=UPSTREAM["epochs"])
    t.add_argument("--batch-size", type=int, default=UPSTREAM["batch_size"], help="per device")
    t.add_argument("--lr", type=float, default=UPSTREAM["lr"])
    t.add_argument("--lr-sched-factor", type=float, default=UPSTREAM["lr_sched_factor"])
    t.add_argument("--grad-loss-weight", type=float, default=UPSTREAM["grad_loss_weight"])
    t.add_argument("--precision", default=UPSTREAM["precision"])
    t.add_argument("--accelerator", default="auto")
    t.add_argument("--devices", type=_devices, default="auto")
    t.add_argument("--strategy", default="auto")
    t.add_argument("--num-nodes", type=int, default=1)
    t.add_argument("--num-workers", type=int, default=UPSTREAM["num_workers"])
    t.add_argument("--crop", type=int, default=None, help="HR crop edge (default: full cube)")
    t.add_argument("--preload", action="store_true", help="load train/val into RAM per rank")
    t.add_argument("--max-time", default=None, help="seconds or [DD:]HH:MM:SS for this process")
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--save-period", type=int, default=UPSTREAM["save_period"])
    t.add_argument("--accumulate-grad-batches", type=int, default=1)
    t.add_argument("--hr-size", type=int, default=None, help="default: inferred from files")
    t.add_argument("--no-resume", action="store_true")
    t.add_argument("--gradient-clip-val", type=float, default=None,
                   help="clip the global gradient norm (default: off, as upstream)")

    def eval_common(q):
        q.add_argument("--root", required=True)
        q.add_argument("--out", required=True, help="JSON Lines file (appended)")
        q.add_argument("--splits", nargs="+", default=list(EVAL_SPLITS))
        q.add_argument("--batch-size", type=int, default=1)
        q.add_argument("--device", default=None, help="default: cuda if available")
        q.add_argument("--num-workers", type=int, default=4)
        q.add_argument("--hr-size", type=int, default=None)
        q.add_argument("--no-extras", action="store_true", help="upstream keys only")

    e = sub.add_parser("eval", help="evaluate .pt state_dicts in shipped naming")
    e.add_argument("weights", nargs="+")
    e.add_argument("--source", default=None, help="row source label (default: parent dir name)")
    eval_common(e)

    s = sub.add_parser("eval-shipped", help="evaluate root/best_model_weights/seed*/*.pt")
    eval_common(s)
    s.add_argument("--families", nargs="+", default=None)
    s.add_argument("--seeds", nargs="+", type=int, default=None)
    s.add_argument("--weights-dir", default=None)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)

    lb = sub.add_parser("leaderboard", help="markdown table from JSON Lines rows")
    lb.add_argument("rows", nargs="+")
    lb.add_argument("--out", default=None)
    lb.add_argument("--keys", nargs="+", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `blastnet` console script."""
    args = build_parser().parse_args(argv)
    if args.cmd == "download":
        from blastnet.data.kaggle import download

        print(download(args.slug, args.out, file=args.file))
    elif args.cmd == "catalog":
        from blastnet.data.kaggle import catalog

        results = catalog(args.search)
        if args.csv:
            results.to_csv(args.csv, index=False)
        else:
            sys.stdout.write(results.to_string(index=False) + "\n")
    elif args.cmd == "convert":
        from blastnet.data import zarr_convert

        print(zarr_convert(args.root, args.out, tuple(args.splits), tuple(args.ratios),
                           chunks=(1, 4, args.hr_size, args.hr_size, args.hr_size),
                           hr_size=args.hr_size))
    elif args.cmd == "train":
        from blastnet.train import train

        run = train(args.root, args.name, args.size, args.ratio, args.loss, args.aug == "aug",
                    args.seed, args.out, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, lr_sched_factor=args.lr_sched_factor,
                    grad_loss_weight=args.grad_loss_weight, precision=args.precision,
                    accelerator=args.accelerator, devices=args.devices, strategy=args.strategy,
                    num_nodes=args.num_nodes, num_workers=args.num_workers, crop=args.crop,
                    preload=args.preload, max_time=args.max_time, max_steps=args.max_steps,
                    save_period=args.save_period,
                    accumulate_grad_batches=args.accumulate_grad_batches,
                    hr_size=args.hr_size, resume=not args.no_resume,
                    gradient_clip_val=args.gradient_clip_val)
        print(run)
    elif args.cmd in ("eval", "eval-shipped"):
        from blastnet.eval import evaluate_shipped, evaluate_weights, read_rows

        kw = dict(batch_size=args.batch_size, device=_device(args.device),
                  num_workers=args.num_workers, hr_size=args.hr_size, extras=not args.no_extras)
        if args.cmd == "eval":
            done = {(r["file"], r["split"]) for r in read_rows(args.out)}
            for w in args.weights:
                w = Path(w)
                source = args.source or w.resolve().parent.name
                file = f"{source}/{w.name}"
                for row in evaluate_weights(w, args.root, tuple(args.splits), args.out,
                                            source=source, file=file, skip=done, **kw):
                    print(row["file"], row["split"], flush=True)
        else:
            for row in evaluate_shipped(args.root, args.out, args.families, args.seeds,
                                        tuple(args.splits), weights_dir=args.weights_dir,
                                        shard=args.shard, num_shards=args.num_shards, **kw):
                print(row["file"], row["split"], row["seconds"], flush=True)
    elif args.cmd == "leaderboard":
        from blastnet.eval import leaderboard, read_rows

        table = leaderboard(read_rows(args.rows), args.keys)
        if args.out:
            Path(args.out).write_text(table)
        sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
