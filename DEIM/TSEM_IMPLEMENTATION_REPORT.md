# TSEM Implementation Report

## 1. 环境与基线

- Git base commit: `d2669d2171a36636b9996661b1eac5016fbee7f0`
- Git branch: `feature/tsem-v1`
- Baseline config: `configs/deim_dfine/deim_hgnetv2_l_wood.yml`
- Local Python/PyTorch observed by lightweight TSEM check: `torch 1.9.0+cu111`
- 修改前工作区状态：`E:\article` 仓库中除已提交的 `DEIM/README.md` 外，已有大量文件处于 staged `A` 状态；本次实现仅修改/新增 TSEM 相关路径，未覆盖 baseline 配置。

## 2. 修改摘要

| 文件 | 新建/修改 | 主要内容 |
|---|---|---|
| `engine/deim/tsem.py` | 新建 | TSEM 主模块：自适应平滑、高频分支、多感受野上下文、空间门控、跨尺度门控、零初始化残差 |
| `engine/deim/hybrid_encoder.py` | 修改 | 在 `input_proj` 后、Transformer/FPN/PAN 前可选插入 TSEM；默认关闭 |
| `configs/deim_dfine/deim_hgnetv2_l_wood_tsem_high.yml` | 新建 | `high_only` 消融配置 |
| `configs/deim_dfine/deim_hgnetv2_l_wood_tsem_context.yml` | 新建 | `context_only` 消融配置 |
| `configs/deim_dfine/deim_hgnetv2_l_wood_tsem_dual_sum.yml` | 新建 | `dual_sum` 消融配置 |
| `configs/deim_dfine/deim_hgnetv2_l_wood_tsem_gate.yml` | 新建 | `dual_gate` 消融配置 |
| `configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml` | 新建 | `full` 配置，含空间门控和跨尺度门控 |
| `tools/debug/test_tsem_shapes.py` | 新建 | TSEM 本体 shape、identity、backward、AMP 测试 |
| `tools/debug/check_tsem_params.py` | 新建 | 检查 TSEM 参数是否被 optimizer 覆盖 |
| `tools/debug/test_tsem_model_forward.py` | 新建 | 完整模型随机前向和 state_dict 保存/加载测试 |
| `tools/visualization/tsem_feature_vis.py` | 新建 | 单图 high/context/gate/cross-scale debug 可视化工具 |
| `TSEM_IMPLEMENTATION_REPORT.md` | 新建 | 实现报告 |

## 3. 模块结构

- 输入：`list[Tensor]`，长度为 3，每层形状为 `[B, C, H, W]`。
- 输出：新建的 `list[Tensor]`，长度、shape、dtype、device 与输入一致。
- 插入点：`HybridEncoder.input_proj` 之后，原 Transformer encoder / FPN / PAN 之前。
- 残差形式：`out_i = feat_i + gamma_i * scale_i * residual_i`。
- `gamma` 初始化为 `0.0`，因此首次加载时为恒等映射。
- `full` 模式参数量：`920479`，低于 1.5M 目标。

模式实例化：

| mode | 组件 |
|---|---|
| `off` | 不实例化 TSEM，baseline 路径不变 |
| `high_only` | smooth + high branch |
| `context_only` | smooth + dilated context branches + context gate |
| `dual_sum` | high + context |
| `dual_gate` | high + context + spatial gate |
| `full` | dual_gate + cross-scale gate |

## 4. 测试结果

| 测试 | 结果 | 说明 |
|---|---|---|
| Python syntax compile | PASS | `py_compile` 通过 |
| `git diff --check` | PASS | tracked 修改无 whitespace error |
| TSEM shape | PASS | CPU float32，5 种模式 × 4 组 level |
| gamma=0 identity | PASS | `max_abs_error < 1e-6` |
| backward | PASS | trainable TSEM 参数 grad 非 None 且有限 |
| CUDA/AMP | 未执行 | 本机未跑 CUDA，训练服务器可用 `--skip-cuda` 以外模式复测 |
| optimizer coverage | 未执行 | 本机完整 DEIM 导入受 PyTorch/CUDA 环境限制失败 |
| model forward | 未执行 | 本机 `torch 1.9.0+cu111` 与仓库代码不匹配，且页面文件不足导致 CUDA 库加载失败 |
| DDP smoke | 未执行 | 需在训练服务器用 2 GPU 跑 2 epoch |
| test-only | 未执行 | 需使用 smoke checkpoint 在训练服务器验证 |
| ONNX export | 未执行 | 需在训练服务器使用训练后权重验证 |

本地完整模型相关失败信息：

- `ImportError: cannot import name 'LRScheduler' from torch.optim.lr_scheduler`
- `OSError: [WinError 1455] 页面文件太小，无法完成操作。Error loading ... cusolverMg64_11.dll`

## 5. 预训练加载摘要

本地未完成完整 checkpoint 加载测试。预期从 baseline COCO 权重 tuning 时：

- TSEM 新增参数应作为 missing keys 出现，例如 `encoder.tsem.*`。
- 原 backbone / encoder 非 TSEM 部分 / decoder 不应出现大量 unexpected keys。
- 9 类分类头处理应与当前 baseline tuning 行为一致。

## 6. 正式训练命令

high_only:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --master_port=7791 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/deim_hgnetv2_l_wood_tsem_high.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output/deim_hgnetv2_l_wood_tsem_high_960
```

dual_gate:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --master_port=7792 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/deim_hgnetv2_l_wood_tsem_gate.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output/deim_hgnetv2_l_wood_tsem_gate_960
```

full:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --master_port=7793 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output/deim_hgnetv2_l_wood_tsem_full_960
```

建议 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --master_port=7794 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/deim_hgnetv2_l_wood_tsem_high.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output/deim_hgnetv2_l_wood_tsem_high_smoke \
-u epoches=2 train_dataloader.dataset.transforms.policy.epoch=[1,1,2] train_dataloader.collate_fn.mixup_epochs=[1,1] train_dataloader.collate_fn.stop_epoch=2
```

## 7. 可视化命令

```bash
python tools/visualization/tsem_feature_vis.py \
-c configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml \
-r output/deim_hgnetv2_l_wood_tsem_full_960/best_stg2.pth \
--input /path/to/image.jpg \
--output-dir output/tsem_vis \
--level 0
```

## 8. 已知风险与未完成项

- C3 级别特征分辨率高，`full` 模式会增加显存与计算开销。
- `context_only` 和 `full` 包含多空洞率分支，训练速度会低于 baseline。
- 本地 Windows 环境无法完成完整模型前向、optimizer coverage、DDP smoke 和 ONNX 验证；需在实际训练服务器复测。
- `tsem_feature_vis.py` 依赖 `tsem_debug=True` 的 detach 后统计，只适合单图分析，不应在正式训练中开启。
