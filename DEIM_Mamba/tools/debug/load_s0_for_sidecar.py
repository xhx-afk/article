"""Audit S0 local-anchor weights before an S1/S2 sidecar warm-start."""

from __future__ import annotations

import argparse

from labs_mamba_debug_utils import (
    SIDECAR_PREFIX,
    audit_checkpoint_load,
    build_config,
    load_checkpoint,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="S1 or S2 YAML config")
    parser.add_argument("--checkpoint", required=True, help="converged S0 checkpoint")
    parser.add_argument(
        "--output",
        default="artifacts/labs_mamba_v3/s0_to_sidecar_load_report.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args.config)
    model = cfg.model
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "mamba_sidecar_p4"):
        raise RuntimeError("target config must create encoder.mamba_sidecar_p4")
    report = audit_checkpoint_load(
        model,
        load_checkpoint(args.checkpoint),
        require_all_sidecar_missing=True,
    )
    report.update(
        {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "allowed_missing_prefix": SIDECAR_PREFIX,
        }
    )
    write_json(args.output, report)
    if not report["compatible"]:
        raise RuntimeError(
            "S0 -> sidecar warm-start audit failed; inspect {}".format(args.output)
        )
    print("S0 -> sidecar warm-start audit passed: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
