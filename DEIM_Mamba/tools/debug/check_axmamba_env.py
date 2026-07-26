"""Report AxMamba environment compatibility and run a Mamba gradient check."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "axmamba_v1" / "environment_check.json"


def main() -> int:
    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }

    try:
        import mamba_ssm
        from mamba_ssm import Mamba

        report["mamba_ssm"] = getattr(mamba_ssm, "__version__", "unknown")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Mamba(d_model=256, d_state=16, d_conv=4, expand=2).to(device)
        sequence = torch.randn(2, 64, 256, device=device, requires_grad=True)
        output = model(sequence)
        loss = output.float().square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        report.update(
            {
                "device": str(device),
                "forward_shape": list(output.shape),
                "forward_finite": bool(torch.isfinite(output).all().item()),
                "backward_finite": bool(
                    gradients
                    and all(
                        gradient is not None
                        and torch.isfinite(gradient).all().item()
                        for gradient in gradients
                    )
                ),
                "status": "passed",
            }
        )
        exit_code = 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        exit_code = 1

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("report:", ARTIFACT_PATH)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
