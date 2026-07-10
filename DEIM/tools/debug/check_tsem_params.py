"""
Check whether all trainable TSEM parameters are covered by optimizer groups.

Run:
    python tools/debug/check_tsem_params.py -c configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from engine.core import YAMLConfig  # noqa: E402


def _param_id_map(model) -> Dict[int, str]:
    return {id(param): name for name, param in model.named_parameters()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TSEM optimizer parameter coverage")
    parser.add_argument("-c", "--config", default="configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml")
    args = parser.parse_args()

    cfg = YAMLConfig(args.config, HGNetv2={"pretrained": False})
    model = cfg.model
    optimizer = cfg.optimizer

    id_to_name = _param_id_map(model)
    tsem_params = {
        name: param
        for name, param in model.named_parameters()
        if ".tsem." in name or name.endswith(".tsem.gamma")
    }
    trainable_tsem = {name: param for name, param in tsem_params.items() if param.requires_grad}

    optimizer_ids = set()
    group_tsem_counts = []
    for group in optimizer.param_groups:
        count = 0
        for param in group["params"]:
            optimizer_ids.add(id(param))
            if id_to_name.get(id(param), "").find(".tsem.") >= 0:
                count += param.numel()
        group_tsem_counts.append(count)

    missing = [name for name, param in trainable_tsem.items() if id(param) not in optimizer_ids]
    total_params = sum(param.numel() for param in model.parameters())
    tsem_total = sum(param.numel() for param in tsem_params.values())
    tsem_trainable = sum(param.numel() for param in trainable_tsem.values())

    print(f"total_params: {total_params}")
    print(f"tsem_params: {tsem_total}")
    print(f"trainable_tsem_params: {tsem_trainable}")
    for idx, count in enumerate(group_tsem_counts):
        print(f"optimizer_group_{idx}_tsem_params: {count}")
    print(f"missing_trainable_tsem_params: {len(missing)}")
    if missing:
        for name in missing:
            print(f"  MISSING {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
