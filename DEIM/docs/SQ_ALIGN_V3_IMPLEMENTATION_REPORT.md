# SQ-Align V3 / CMSQ-DEIM 实现报告

## 1. 范围与结论

本次按仓库根目录 `SQ_ALIGN_V3_Codex_Destructive_Refactor_Spec.md` 对当前本地代码进行了破坏性重构。

已完成：

- 删除失效的 Localization Quality 分支及其输出、loss、融合配置；
- 删除旧 far-background ranking；
- 保持原 DEIM MAL 主分类目标不变；
- 保留并接入 Binary Defect Auxiliary（BDA）；
- 实现 Query-Conditioned Semantic Evidence（QCSE）；
- 实现 Continuous Mask-Supported Quality（CMSQ）；
- 实现只抑制、不放大的 Semantic Suppression Gate；
- 实现 per-GT Near-GT Candidate Ranking（NGCR）；
- 实现 V0～V4 干净消融配置；
- 实现单一 JSON 统一评估器与验证集 gate sweep；
- 完成单元测试、batch=2 forward/backward、10 iteration 和 100 iteration CUDA AMP smoke；
- 验证官方 COCO 预训练权重仍可兼容加载。

未启动完整训练，不声明 AP、AP75 或规范中的最终精度目标已经达到。

## 2. Git/版本状态

`E:\article\DEIM_sqmal_v1` 本身没有 `.git`，父目录 `E:\article` 将整个目录显示为 untracked，因此无法从该目录可靠给出“重构前项目 commit”。

- 父工作树分支（仅审计）：`feature/sq-mal-v1`
- 父工作树当前 commit（仅审计）：`2830352a9d72bd05e0cd202401b76e67a162c95a`
- 重构后 commit：无；本次未执行 commit 或 push
- 用户现有父工作树状态未被清理或重置

## 3. 删除内容

### 3.1 删除文件/目录

- `engine/deim/semantic_query_alignment.py`
- `configs/deim_dfine/ablation_sqalign/`（旧 V2 R0～R4 配置集合；最终空目录也已移除）

### 3.2 从现有代码彻底删除的接口和字段

- `QueryLocalizationQualityHead`
- `pred_loc_quality`
- `dec_loc_quality_head`
- `loss_loc_quality`
- `compose_dual_quality_scores`
- `loc_quality_rerank`
- `loc_quality_power`
- 旧 matched=1 `build_query_semantic_supervision`
- 旧 raw top-20% `pool_query_semantic_evidence`
- `select_final_score_background`
- `final_score_ranking_loss`
- `loss_final_bg_rank`
- 旧 semantic power 乘分路径
- V1/V2 compatibility wrapper

最终全仓符号搜索未发现上述实现残留；测试中仅保留 `assertNotIn("pred_loc_quality", outputs)` 这一反向验收断言。

## 4. 新增文件

### 4.1 模型与配置

- `engine/deim/continuous_semantic_alignment.py`
- `configs/deim_dfine/ablation_sqalign_v3/v0_baseline.yml`
- `configs/deim_dfine/ablation_sqalign_v3/v1_defect_aux.yml`
- `configs/deim_dfine/ablation_sqalign_v3/v2_query_conditioned_mask.yml`
- `configs/deim_dfine/ablation_sqalign_v3/v3_continuous_semantic_quality.yml`
- `configs/deim_dfine/ablation_sqalign_v3/v4_near_gt_candidate_rank.yml`

### 4.2 评估与 smoke

- `tools/wood/sqalign_v3_evaluation.py`
- `tools/wood/evaluate_sqalign_v3.py`
- `tools/wood/sweep_semantic_gate_v3.py`
- `tools/wood/smoke_test_sqalign_v3.py`

### 4.3 测试

- `tests/sqalign_test_utils.py`
- `tests/test_continuous_mask_support.py`
- `tests/test_query_conditioned_roi_mask.py`
- `tests/test_semantic_quality_head.py`
- `tests/test_semantic_suppression_gate.py`
- `tests/test_near_gt_candidate_rank.py`
- `tests/test_sqalign_v3_forward_backward.py`
- `tests/test_unified_evaluator.py`

### 4.4 文档

- `docs/SQ_ALIGN_V3_IMPLEMENTATION_REPORT.md`

## 5. 修改文件

- `engine/deim/__init__.py`：导出 V3 模块；移除旧语义对齐导出。
- `engine/deim/deim.py`：从最高分辨率 encoder feature 构建 defect/pixel 分支，并在最终正常 query 上生成 query mask 和 learned semantic quality。
- `engine/deim/dfine_decoder.py`：只在主输出增加最终正常 query feature；DN query 在返回前剥离；不复制到 aux；Integral/LQE softmax 核在 AMP 下固定为 float32。
- `engine/deim/deim_criterion.py`：删除 V2 loss，增加 defect/query-mask/semantic-quality/semantic-rank/candidate-rank 及日志指标；新 loss 只作用于 main 分支。
- `engine/deim/postprocessor.py`：在 flatten/top-k 前应用 suppression-only semantic gate。
- `engine/solver/det_engine.py`：将 epoch/total_epochs 传给 criterion，支持 candidate rank 的 60% 延迟启用。
- `engine/solver/det_solver.py`：训练入口向 epoch engine 提供 total_epochs。

以下已存在的数据链按 V3 要求保留并通过测试，未改回旧目标：

- `engine/data/dataset/coco_dataset.py` 中的 RLE/polygon mask、`mask_valid` 和 annotation ID；
- `engine/data/transforms/_transforms.py`、`functional.py` 中实例字段同步；
- `engine/data/dataloader.py` 中 mixup/mosaic 的 mask 与 `mask_valid` 链。

## 6. 模块结构与 tensor shape

以 `B=2, Q=300, C=256, D=64, ROI=7` 为例：

| 输出 | Shape | 说明 |
|---|---|---|
| `pred_logits` | `[B,Q,9]` | 原 DEIM MAL 分类 logits |
| `pred_boxes` | `[B,Q,4]` | normalized `cxcywh` |
| `pred_query_features` | `[B,Q,C]` | 最终正常 query，不含 DN |
| `pred_defect_logits` | `[B,1,Hd,Wd]` | 最高分辨率 encoder feature 的 binary defect 输出 |
| semantic pixel embedding | `[B,D,Hs,Ws]` | 默认约 stride 16 |
| query semantic embedding | `[B,Q,D]` | L2 normalized |
| ROI pixel feature | `[B,Q,D,7,7]` | ROIAlign，box 默认 detach |
| `pred_query_mask_logits` | `[B,Q,7,7]` | query-conditioned mask |
| `pred_sem_quality` | `[B,Q,1]` | learned semantic-quality logit |

Smoke（128×128）实际输出：

- `pred_defect_logits=[2,1,16,16]`
- `pred_query_mask_logits=[2,300,7,7]`
- `pred_sem_quality=[2,300,1]`

QCSE mask：

```text
M[b,q,h,w] = einsum(U[b,q,d], R[b,q,d,h,w]) / sqrt(D)
```

ROIAlign 参数：`output_size=7`、`sampling_ratio=2`、`aligned=True`，预测框默认 detach。

## 7. Target 与损失

### 7.1 Binary defect auxiliary

有效 instance mask 并集通过 `adaptive_max_pool2d` 下采样，以保留细线缺陷。

```text
L_defect = BCEWithLogits + DiceLoss
总权重 = 0.5
```

### 7.2 Query mask

每图采样：全部有效 matched、最多 20 个 `class_score × best_IoU` near-unmatched、最多 20 个高 class-score far-background。

```text
L_query_mask = BCEWithLogits + DiceLoss
matched / near / far 组内权重 = 1.0 / 0.25 / 0.10
总权重 = 0.25
```

invalid instance mask 被忽略；far-background 使用全零 ROI target；ROI GT target 保持 soft mask。

### 7.3 Continuous semantic quality

```text
coverage = |M_gt ∩ B_pred| / |M_gt|
pred_density = |M_gt ∩ B_pred| / |B_pred|
gt_density = |M_gt ∩ B_gt| / |B_gt|
relative_density = clip(pred_density / gt_density, 0, 1)
target_sem = sqrt(coverage × relative_density)
```

统计通过 GT mask 积分图批量计算，不逐 query 裁剪原图 mask。target 只用于 semantic quality，不乘入 MAL。

SemanticQualityHead 输入十个 ROI mask 统计量（mean/max/std/top10/top25/阈值比例/中心与边界统计）及 query feature 投影，输出 learned logit。

```text
L_semantic_quality = soft-label BCEWithLogits
matched / near / far 组内权重 = 1.0 / 0.5 / 0.10
总权重 = 0.25
L_semantic_rank 总权重 = 0.05
```

matched、near、far 三组互斥。

### 7.4 Suppression-only gate

```text
q = sigmoid(pred_sem_quality)
g = 1 - lambda × (1-q)^gamma
final_score = class_probability × g
```

`g ∈ [1-lambda, 1]`，只能抑制、不能放大，且在 flatten/top-k 前应用。训练配置默认 gate 关闭；参数只能在 validation sweep 选择。

### 7.5 Per-GT candidate ranking

每个 GT 收集 `IoU >= 0.30` 的候选，按 true-class score 最多取 10 个；matched query 中 IoU 最高者作为 anchor。定位排序只取 target IoU gap 大于 0.10 的 pair，同时在 anchor 上约束 true class 高于 hardest wrong class。

```text
rank_score = class_probability × semantic_gate.detach()
loss_candidate_rank 总权重 = 0.05
start_ratio = 0.60
```

候选选择和 box 均 detach；ranking 只更新分类 score，不更新 semantic map、query mask 或 bbox。

## 8. Branch loss 规则

| 分支 | MAL/bbox/local | defect | query mask | semantic quality/rank | candidate rank |
|---|---:|---:|---:|---:|---:|
| main | 是 | 是 | 是 | 是 | 是 |
| decoder aux | 是 | 否 | 否 | 否 | 否 |
| pre | 是 | 否 | 否 | 否 | 否 |
| encoder aux | 是 | 否 | 否 | 否 | 否 |
| DN | 是 | 否 | 否 | 否 | 否 |

原 `loss_labels_mal` 保持 `IoU^gamma` 正样本目标，没有 semantic target 乘法。

## 9. V0～V4 配置

| 配置 | BDA | query mask | learned semantic quality | candidate rank | inference gate 默认 |
|---|---:|---:|---:|---:|---:|
| V0 | 否 | 否 | 否 | 否 | 关闭 |
| V1 | 是 | 否 | 否 | 否 | 关闭 |
| V2 | 是 | 是 | 否 | 否 | 关闭 |
| V3 | 是 | 是 | 是 | 否 | 关闭 |
| V4 | 是 | 是 | 是 | 是 | 关闭 |

五个配置均已实例化测试通过。

## 10. Unified evaluator

主脚本：`tools/wood/evaluate_sqalign_v3.py`。

只写一个用户指定的主 JSON，顶层固定为：

```text
meta
checkpoint_load
dataset
coco
tide
fixed_thresholds
error_taxonomy
confusion
recall_by_iou
calibration
defect_map
query_mask
semantic_quality
candidate_ranking
score_analysis
runtime
automatic_diagnosis
```

实现内容：

- COCO AP50～AP95、APS/APM/APL、AR1/10/100 和逐类 AP/AP50/AP75/AP80/AP90；
- TIDE 可选依赖降级，不因未安装而崩溃；
- 0.05/0.10/0.25/0.50 固定阈值 TP/FP/FN、precision/recall/F1；
- far-background、duplicate、class confusion、localization FP 分类；
- 空 GT 图像的背景 FP 不会被遗漏；
- confusion matrix、IoU 0.30～0.95 recall；
- overall/per-class ECE、LaECE、Brier、15-bin reliability；
- defect low/original-resolution、boundary F1 和像素统计；
- matched/near/far query mask 指标；
- semantic distribution、AUC 与 target Spearman；
- candidate ranking violation/margin/IoU 指标；
- class-only 与 class×gate 的 AP/AUC/FP 对比；
- latency/FPS/peak CUDA memory；
- 集中式 automatic diagnosis；
- config/checkpoint/annotation SHA256 与 checkpoint load 明细；
- `--max-images` partial evaluation 只评估实际图像；partial 模式明确跳过 TIDE。

## 11. 测试结果

环境：

```text
Conda env: yolov11
Python: D:\Anaconda\envs\yolov11\python.exe
PyTorch: 2.7.1+cu118
Device: cuda:0（forward/backward 和 smoke）
```

单元测试：

```text
python -m unittest discover -s tests -v
Ran 26 tests in 4.160s
OK
```

覆盖：continuous support、empty GT、invalid mask、thin mask、query/ROI 条件化、box detach、soft target、learned quality、gate 上下界、candidate/class rank、mask 数据链、V0～V4 构建、batch=2 forward/backward、官方权重加载、10 图单 JSON evaluator、空 GT 背景 FP。

核心 Python 文件 `py_compile` 通过。

## 12. Smoke 与预训练权重

预训练权重：

```text
weight/deim_dfine_hgnetv2_l_coco_50e.pth
SHA256=A9AC1F9BF51A41A7EC02C373C85E0109F84908DDE11D80759B43248DCB638BAA
matched keys=1156
missing keys=42（新 V3 分支及数据集 head，符合 non-strict tuning 预期）
```

10 iteration：

```text
batch=2, image=128, CUDA AMP, GradScaler init_scale=1
elapsed=5.998s
mean=0.600s/iteration
all_finite=true
```

100 iteration：

```text
batch=2, image=128, CUDA AMP, GradScaler init_scale=1
elapsed=57.114s
mean=0.571s/iteration
last total_loss=90.8997
all_finite=true
```

报告文件：

- `output_2/sqalign_v3/smoke/v4_10iter.json`
- `output_2/sqalign_v3/smoke/v4_100iter.json`

首次用 GradScaler 默认大倍率验证时，聚合后的 D-FINE aux loss 在首步触发 scaled-gradient overflow。定位到 D-FINE Integral softmax 后，已将 Integral/LQE 概率核固定为 fp32；有界 smoke 使用保守 `init_scale=1`，随后 100 步全部有限。生产训练仍由 GradScaler 动态调节，日志中应监控 scale 和 skipped step。

## 13. 运行命令

### 13.1 单元测试

PowerShell 若本机执行策略阻止 `conda activate`，使用等价的 `conda run`：

```powershell
conda run -n yolov11 python -m unittest discover -s tests -v
```

### 13.2 Smoke

```powershell
conda run -n yolov11 python tools/wood/smoke_test_sqalign_v3.py `
  -c configs/deim_dfine/ablation_sqalign_v3/v4_near_gt_candidate_rank.yml `
  -t weight/deim_dfine_hgnetv2_l_coco_50e.pth `
  --iterations 100 --batch-size 2 --image-size 128 `
  --num-classes 9 --device cuda:0 --amp `
  --output-json output_2/sqalign_v3/smoke/v4_100iter.json
```

### 13.3 完整训练命令（仅提供，未执行）

V0 示例：

```bash
mkdir -p output_2/sqalign_v3/v0_baseline
CUDA_VISIBLE_DEVICES=0,2 torchrun --master_port=7840 --nproc_per_node=2 \
  train.py -c configs/deim_dfine/ablation_sqalign_v3/v0_baseline.yml \
  --use-amp --seed=0 \
  -t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
  --output-dir output_2/sqalign_v3/v0_baseline \
  2>&1 | tee output_2/sqalign_v3/v0_baseline/train_console.log
```

V1～V4 对应：

| 实验 | port | config | output-dir |
|---|---:|---|---|
| V1 | 7841 | `v1_defect_aux.yml` | `output_2/sqalign_v3/v1_defect_aux` |
| V2 | 7842 | `v2_query_conditioned_mask.yml` | `output_2/sqalign_v3/v2_query_conditioned_mask` |
| V3 | 7843 | `v3_continuous_semantic_quality.yml` | `output_2/sqalign_v3/v3_continuous_semantic_quality` |
| V4 | 7844 | `v4_near_gt_candidate_rank.yml` | `output_2/sqalign_v3/v4_near_gt_candidate_rank` |

只替换 V0 命令中的 port、config 和 output-dir。

### 13.4 单一 JSON 评估

```bash
python tools/wood/evaluate_sqalign_v3.py \
  -c configs/deim_dfine/ablation_sqalign_v3/v4_near_gt_candidate_rank.yml \
  -r output_2/sqalign_v3/v4_near_gt_candidate_rank/best.pth \
  --images-dir /path/to/val/images \
  --ann-file /path/to/instances_val_sqmal.json \
  --output-json output_2/sqalign_v3/v4_near_gt_candidate_rank/unified_eval.json \
  --device cuda:0 --batch-size 4 --num-workers 4 --max-dets 300 \
  --score-thresholds 0.05 0.10 0.25 0.50 \
  --semantic-gate-lambda 0.10 --semantic-gate-gamma 2.0
```

### 13.5 Validation gate sweep

```bash
python tools/wood/sweep_semantic_gate_v3.py \
  -c configs/deim_dfine/ablation_sqalign_v3/v4_near_gt_candidate_rank.yml \
  -r output_2/sqalign_v3/v4_near_gt_candidate_rank/best.pth \
  --images-dir /path/to/val/images \
  --ann-file /path/to/instances_val_sqmal.json \
  --lambdas 0 0.05 0.10 0.15 0.20 0.30 \
  --gammas 1 2 3 \
  --output-json output_2/sqalign_v3/v4_near_gt_candidate_rank/gate_sweep.json
```

## 14. 风险与未完成项

1. 没有运行完整训练、1 epoch 真实数据训练或测试集评估；没有可报告的 V0～V4 AP 差异。
2. 本机配置中的 train/val 路径指向远程 Linux 数据，无法在本机完成真实数据 epoch/evaluator 闭环。
3. 未用“训练后生成的 V3 checkpoint”跑完整统一 evaluator；当前完成的是 synthetic 10-image schema/数值测试及官方预训练权重兼容加载测试。
4. V1/V2 checkpoint 不兼容是有意设计；V3 新分支从随机初始化开始。
5. Query ROIAlign 增加训练与推理开销；960 输入下的吞吐、显存和 TensorRT/ONNX 支持必须在目标服务器测量。
6. `return_masks: true` 使 V0 也保留同一数据链以保证消融一致性，但会增加 host memory/数据加载开销。
7. TIDE 是可选依赖；未安装时 JSON 会写明 `available=false`，不会崩溃。
8. Gate 默认关闭；必须只在 validation 选择 lambda/gamma，然后固定一次用于 test，不能在 test sweep。
9. Candidate rank 60% 启用依赖 solver 正确传入 total_epochs；当前训练入口已接通，其他自定义训练入口也必须调用 `criterion.set_epoch`。
10. 初始精度目标只有完整训练和统一 evaluator 后才能验收；未达到规范指标时不得宣称方法成功。

## 15. 不应提交的大文件

- `weight/*.pth`
- `output_2/**`
- `**/__pycache__/**`
- `*.pyc`
- 数据集图片、临时 evaluator 输出和训练日志

本次没有执行 Git 提交或推送。
