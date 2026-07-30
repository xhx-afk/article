# SQ-Align V2：Codex 直接重构实现规格书

> 目标工程：用户本地 DEIM-D-FINE 木材缺陷检测代码  
> 当前代码来源：`feature/sq-mal-v1` 的本地版本  
> 目标：直接重构当前代码，解决 SQ-MAL-v1 已暴露的核心问题  
> 重要授权：**无需兼容 SQ-MAL-v1；允许删除、重命名或覆盖 v1 的接口、配置、文件和 checkpoint key。**  
> 禁止事项：不要为了兼容旧配置保留 `use_sqmal`、`loss_sqmal`、旧 hard-bg 或 v1 wrapper；不要同时引入第二方向 MB-FDR。

---

# 1. Codex 必须先理解的实验结论

用户在相同 quick-balanced 测试集上完成了 Q0～Q4：

| 实验              |        AP |      AP50 |      AP75 | Precision@0.25 | Recall@0.25 | 高阈值背景FP | 低阈值背景FP | all-query quality-best-IoU Spearman |
| ----------------- | --------: | --------: | --------: | -------------: | ----------: | -----------: | -----------: | ----------------------------------: |
| Q0 baseline       |     49.79 |     80.58 |     52.44 |          51.88 |       82.97 |         1719 |         4660 |                                  无 |
| Q1 quality        |     50.27 |     80.82 | **53.60** |          56.74 |       82.90 |         1374 |         2868 |                               0.051 |
| **Q2 defect aux** | **50.77** | **81.51** |     53.39 |      **57.75** |       82.28 |         1289 |         2912 |                           **0.229** |
| Q3 SQ-MAL         |     49.83 |     80.78 |     52.20 |          57.85 |       82.16 |         1279 |         3202 |                               0.117 |
| Q4 SQ-MAL+HBG     |     49.82 |     80.40 |     52.44 |          57.51 |       82.32 |         1296 |         2836 |                           **0.004** |

结论：

1. Query Quality Head 和 top-k 前重排序有效；
2. Binary Defectness Auxiliary 有效，Q2 是当前最佳；
3. `IoU^gamma * semantic_support^beta` 直接替换 MAL 正目标无效；
4. `confidence * (1-quality)` 的 hard-background 选择与最终推理排序不一致；
5. 当前 quality head 对 matched query 的相关性较好，但对全部 query 的 best-IoU 相关性很弱；
6. defectness map 只作为辅助 loss，没有进入 query 决策链。

本次重构必须围绕这些痛点，不再尝试修补 v1 兼容性。

---

# 2. 新方法定位

将当前方向重构为：

> **SQ-Align：Semantic Query-Level Quality Alignment**

核心由四部分组成：

1. **All-Query Localization Quality（AQLQ）**  
   对所有 decoder query 监督连续 best-IoU，而不是只给 Hungarian matched query 连续质量、其余全部置 0。

2. **Query Semantic Evidence（QSE）**  
   从预测 defectness map 中，根据每个预测框池化 query 级语义证据，使 semantic map 真正进入 query 决策。

3. **Dual-Quality Score Alignment（DQSA）**  
   最终分数由分类、定位质量和语义证据共同构成，并在 top-k 前完成。

4. **Final-Score-Aligned Background Ranking（FSBR）**  
   按实际最终分数选择最危险的远背景 query，并使用排序损失，而不是重复对低 quality 背景做全零 BCE。

整体公式：

\[
q^{loc}\_i = \sigma(z^{loc}\_i)
\]

\[
q^{sem}_i =
\operatorname{TopKMean}
\left(
\sigma(D)\big|_{B_i}
\right)
\]

\[
S*{i,c}
=
\sigma(l*{i,c})
\cdot
(q^{loc}\_i)^{\eta}
\cdot
(q^{sem}\_i)^{\delta}
\]

其中：

- \(l\_{i,c}\)：原分类 logit；
- \(D\)：二值 defectness logit map；
- \(B_i\)：预测框；
- \(\eta\)：定位质量重排指数；
- \(\delta\)：语义质量重排指数。

主分类损失继续使用原始 DEIM MAL：

\[
L*{cls}=L*{MAL}
\]

**不得再把 semantic support 直接乘进 MAL 正样本目标。**

---

# 3. 直接重构授权

Codex 应直接在用户当前本地版本修改。

允许并建议：

- 删除 `engine/deim/sqmal.py`；
- 新建 `engine/deim/semantic_query_alignment.py`；
- 删除 `loss_labels_sqmal()`；
- 删除 `compute_semantic_support()`；
- 删除 `compute_sq_quality_target()`；
- 删除 `semantic_beta_schedule()`；
- 删除旧 `select_hard_background_queries()`；
- 删除 criterion 中的：
  - `use_sqmal`
  - `semantic_beta`
  - `semantic_warmup_epochs`
  - `semantic_min_mask_area`
  - `sqmal_aux_loss`
- 删除旧 `loss_sqmal`；
- 删除旧 Q0～Q4 配置文件或整体替换 `ablation_sqmal` 目录；
- 将 `pred_quality` 重命名为 `pred_loc_quality`；
- 将旧 checkpoint 的 quality/defect key 视为不兼容；
- 不创建旧接口 alias；
- 不保留 v1 wrapper；
- 不编写自动迁移旧 SQ-MAL checkpoint 的逻辑。

必须保留：

- 原 DEIM MAL；
- D-FINE 原分类、bbox、FGL、DDF、LQE；
- 已验证有效的 semantic RLE 数据链；
- `target["masks"]` 和 `target["mask_valid"]`；
- 原 COCO bbox、类别和 annotation ID；
- 从官方 COCO 预训练权重初始化的能力。

---

# 4. 开始改代码前的预检

Codex 先运行：

```bash
cd /home/zxw4090/hjw/DEIM

git status
git branch --show-current
git rev-parse HEAD
python -V
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

grep -R "class DEIM" -n engine/deim
grep -R "class DFINETransformer" -n engine/deim
grep -R "class DEIMCriterion" -n engine/deim
grep -R "loss_labels_sqmal" -n .
grep -R "pred_quality" -n engine configs
grep -R "compute_semantic_support" -n .
grep -R "select_hard_background_queries" -n .
```

先创建一次安全提交：

```bash
git add -A
git commit -m "backup: state before SQ-Align v2 destructive refactor"
```

之后直接修改当前分支，不要求创建兼容分支。

---

# 5. 目标文件结构

建议最终结构：

```text
engine/deim/
  deim.py
  dfine_decoder.py
  deim_criterion.py
  postprocessor.py
  semantic_query_alignment.py

configs/deim_dfine/ablation_sqalign/
  r0_baseline.yml
  r1_all_query_loc.yml
  r2_loc_defect_aux.yml
  r3_query_semantic.yml
  r4_final_score_rank.yml

tools/wood/
  evaluate_sqalign.py
  visualize_defectness_and_query_scores.py
  sweep_score_powers.py

tests/
  test_all_query_iou_target.py
  test_query_semantic_pooling.py
  test_dual_quality_score.py
  test_background_ranking.py
  test_sqalign_forward_backward.py

docs/
  SQ_ALIGN_V2_IMPLEMENTATION_REPORT.md
```

可保留现有 semantic-map→RLE 数据工具，但删除其中仅为 v1 兼容存在的接口。

---

# 6. 重写 `semantic_query_alignment.py`

新文件只保留 V2 需要的功能。

建议导出：

```python
class QueryLocalizationQualityHead(nn.Module)

def pairwise_box_iou_cxcywh(...)
def build_all_query_best_iou_target(...)
def pool_query_semantic_evidence(...)
def build_query_semantic_supervision(...)
def compose_dual_quality_scores(...)
def select_final_score_background(...)
def final_score_ranking_loss(...)
def build_union_defect_target(...)
```

不再导出 v1 的：

```text
compute_semantic_support
compute_sq_quality_target
semantic_beta_schedule
select_hard_background_queries
apply_quality_rerank
```

---

# 7. All-Query Localization Quality

## 7.1 当前问题

v1 的 target：

```text
matched query：IoU × semantic_support^beta
unmatched query：0
```

它训练的是“是否被 Hungarian 选中”，不是所有 query 的连续定位质量。

## 7.2 新 target

对每张图所有 query：

\[
t_i^{loc}
=
\max_j IoU(B_i,G_j)
\]

无 GT 图像：

\[
t_i^{loc}=0
\]

实现：

```python
def build_all_query_best_iou_target(
    pred_boxes: torch.Tensor,      # [B,Q,4], normalized cxcywh
    targets: Sequence[Dict[str, Tensor]],
) -> torch.Tensor:                 # [B,Q]
```

要求：

- `pred_boxes.detach()`；
- 全 tensor 实现；
- 每图允许 GT 数不同；
- 空 GT 不报错；
- 输出 `[0,1]`；
- 无 NaN/Inf。

## 7.3 Loss

质量 head 输出：

```text
pred_loc_quality: [B,Q,1]
```

采用 BCEWithLogits，但按 query 类型分开平均：

```python
near_mask = best_iou >= loc_near_iou_threshold
far_mask = best_iou < loc_near_iou_threshold
```

默认：

```yaml
loc_near_iou_threshold: 0.10
loc_near_weight: 1.0
loc_far_weight: 0.10
```

matched query 可额外增加权重：

```yaml
loc_matched_extra_weight: 1.0
```

总损失：

\[
L*{locq}
=
w_n \operatorname{BCE}*{near}

- w*f \operatorname{BCE}*{far}
- w*m \operatorname{BCE}*{matched}
  \]

不得把所有 unmatched query 统一以同等权重压到 0。

## 7.4 诊断

每 epoch 输出：

```text
metric_locq_target_mean
metric_locq_pred_mean
metric_locq_pred_near
metric_locq_pred_far
metric_locq_best_iou_spearman_batch
metric_locq_target_ge_05_ratio
```

评估脚本输出：

```text
all-query quality-best-IoU Spearman
top100 quality-best-IoU Spearman
top300 quality-best-IoU Spearman
TP quality mean
background-FP quality mean
duplicate/near-GT quality mean
```

---

# 8. Query Semantic Evidence

## 8.1 保留 defectness head，但改变用途

当前：

```text
encoder feature → defectness map → BCE+Dice
```

新版本：

```text
encoder feature
    ↓
defectness map D
    ├── pixel BCE+Dice
    └── predicted box ROI pooling
              ↓
       per-query q_sem
```

defectness head 在训练和推理时都必须执行，因为 `q_sem` 进入最终分数。

不需要在普通推理结果中返回完整 heatmap，除非可视化开关打开。

## 8.2 池化方式

使用 `torchvision.ops.roi_align`：

```python
def pool_query_semantic_evidence(
    defect_logits: torch.Tensor,    # [B,1,Hd,Wd]
    pred_boxes: torch.Tensor,       # [B,Q,4] normalized cxcywh
    output_size: int = 7,
    topk_ratio: float = 0.20,
    detach_boxes: bool = True,
) -> torch.Tensor:                  # [B,Q,1]
```

步骤：

1. 将归一化 cxcywh 转为 defect map 像素 xyxy；
2. 生成 ROI tensor `[B*Q,5]`；
3. `roi_align(..., output_size=7, aligned=True, sampling_ratio=2)`；
4. 对 `sigmoid(roi_logits)` flatten；
5. 每个 ROI 取最高 `ceil(49*topk_ratio)` 个像素均值；
6. 返回 `[B,Q,1]`。

默认：

```yaml
semantic_roi_size: 7
semantic_topk_ratio: 0.20
semantic_detach_boxes: true
```

为什么不用全框平均：

- Crack、resin 等缺陷细长；
- 正确外接框中也会包含大量背景；
- top-k mean 更适合判断框内是否存在高置信缺陷证据。

## 8.3 Query-level 语义监督

定义：

```text
matched query：target=1
far background（best_iou<0.1）：target=0
near-GT unmatched（best_iou>=0.1）：ignore
```

损失：

\[
L*{semq}
=
w_p BCE(q^{sem}*{pos},1)

- w*b BCE(q^{sem}*{far-bg},0)
  \]

默认：

```yaml
semantic_query_pos_weight: 1.0
semantic_query_bg_weight: 0.25
semantic_query_bg_iou_threshold: 0.10
```

注意：

- `q_sem` 是 defect map 的可微池化结果；
- 梯度主要回传到 defectness head 和 encoder；
- 第一版 box 坐标 detach，不让 semantic loss 改变 bbox；
- 不对 near-GT unmatched query施加0标签，避免打压潜在正确候选。

## 8.4 缺陷图损失

继续使用：

```text
BCEWithLogits + Dice
```

但必须额外提供像素级评估：

```text
Dice
pixel precision
pixel recall
boundary F1（可选）
```

---

# 9. 模型前向重构

修改 `engine/deim/deim.py`：

```python
features = backbone(images)
features = encoder(features)

defect_logits = defectness_head(features[0])

outputs = decoder(features, targets)

pred_sem_quality = pool_query_semantic_evidence(
    defect_logits,
    outputs["pred_boxes"],
    ...
)

outputs["pred_sem_quality"] = pred_sem_quality

if training or return_defect_map_in_eval:
    outputs["pred_defect_logits"] = defect_logits
```

要求：

- defectness head 在启用 query semantic 时推理也运行；
- `pred_sem_quality` 不由 decoder MLP预测；
- semantic score直接来自像素证据；
- 不向 aux decoder 层添加语义 score；
- 不对 DN query计算 query semantic loss；
- 推理结果中不必返回 defect map。

---

# 10. Decoder 重构

将：

```text
QueryQualityHead
pred_quality
dec_quality_head
```

统一重命名为：

```text
QueryLocalizationQualityHead
pred_loc_quality
dec_loc_quality_head
```

允许旧 checkpoint key失效。

`DFINETransformer` 新配置：

```python
use_loc_quality_head: bool = False
loc_quality_hidden_dim: int = 128
loc_quality_aux_last_n: int = 0
```

主输出：

```python
out["pred_loc_quality"] = ...
```

第一版只监督最终 decoder 层：

```yaml
loc_quality_aux_last_n: 0
```

`convert_to_deploy()` 同步处理新 head。

---

# 11. Criterion 重构

## 11.1 删除旧 V1 逻辑

从 `DEIMCriterion` 删除：

```text
use_sqmal
semantic_beta
semantic_warmup_epochs
semantic_min_mask_area
sqmal_aux_loss
loss_labels_sqmal
_current_beta
_matched_quality_components
loss_hard_bg
```

删除所有：

```text
loss_sqmal
metric_semantic_beta_current
metric_mean_semantic_support
```

主分类始终使用原 MAL。

## 11.2 新损失

新增：

```python
loss_loc_quality(...)
loss_defect(...)
loss_query_semantic(...)
loss_final_bg_rank(...)
```

总损失：

\[
L =
L*{MAL}
+L*{box}
+L*{local}
+\lambda*{loc}L*{locq}
+\lambda*{def}L*{def}
+\lambda*{sem}L*{semq}
+\lambda*{rank}L\_{rank}
\]

建议初值：

```yaml
loss_loc_quality: 0.5
loss_defect: 0.5
loss_query_semantic: 0.25
loss_final_bg_rank: 0.05
```

## 11.3 分支规则

| 分支        | MAL | boxes/local | loc quality | defect | query semantic | bg rank |
| ----------- | --: | ----------: | ----------: | -----: | -------------: | ------: |
| main        |   ✓ |           ✓ |           ✓ |      ✓ |              ✓ |       ✓ |
| decoder aux |   ✓ |           ✓ |       默认× |      × |              × |       × |
| pre         |   ✓ |           ✓ |           × |      × |              × |       × |
| encoder aux |   ✓ |           ✓ |           × |      × |              × |       × |
| DN          |   ✓ |           ✓ |           × |      × |              × |       × |

不得把新损失自动复制到所有 auxiliary 分支。

---

# 12. Dual-Quality 最终分数

修改 postprocessor。

新参数：

```python
loc_quality_rerank: bool = False
loc_quality_power: float = 0.25

semantic_quality_rerank: bool = False
semantic_quality_power: float = 0.25
```

计算：

```python
scores = sigmoid(pred_logits)

if loc_quality_rerank:
    scores *= sigmoid(pred_loc_quality).pow(loc_quality_power)

if semantic_quality_rerank:
    scores *= pred_sem_quality.clamp_min(1e-6).pow(semantic_quality_power)

# 之后才 flatten + top-k
```

要求：

- 所有质量融合必须发生在 top-k 前；
- `pred_sem_quality` 已是概率，不重复 sigmoid；
- 支持只开 loc、只开 semantic、两者都开；
- 不使用旧 `quality_rerank` 和 `quality_power`；
- 不创建旧参数兼容 alias。

---

# 13. Final-Score-Aligned Background Ranking

## 13.1 当前 V1 错误

旧选择：

\[
difficulty=p(1-q)
\]

但推理：

\[
S=pq^\eta
\]

它优先挖掘已经会被质量重排压低的背景。

## 13.2 新候选

每图：

```text
unmatched query
且 best_iou_to_any_gt < 0.10
```

## 13.3 新困难度

直接使用最终分数：

\[
h*i=\max_c S*{i,c}
\]

选择最高的 top-k。

默认：

```yaml
final_bg_topk: 5
final_bg_iou_threshold: 0.10
final_bg_start_epoch: 10
```

不要一开始使用20。

## 13.4 排序损失

每图取得：

- matched query 的真实类别最终分数；
- top-k far-background 最终分数。

选最低的 k 个正分数与最高的 k 个背景分数配对：

\[
L*{rank}
=
\frac{1}{k}
\sum
\operatorname{softplus}
\left(
\frac{S*{bg}-S\_{pos}+m}{\tau}
\right)
\]

默认：

```yaml
final_bg_margin: 0.05
final_bg_temperature: 0.10
```

若某图没有正 query或背景候选，返回可求导0。

不得再对 hard background 所有类别做全0 BCE。

## 13.5 选择与反传

- 候选选择和 top-k index 使用 detached final score；
- 排序损失使用原 final score tensor反传；
- final score中 class、loc quality、semantic quality均可获得梯度；
- 第一版 boxes仍 detach，不通过 semantic ROI对坐标反传。

---

# 14. 新配置与干净消融

删除旧 `ablation_sqmal/q0～q4`，创建：

## R0：Baseline

```text
原 MAL
无 loc quality
无 defectness
无 semantic query
无 rank
```

## R1：All-Query Localization Quality

```text
R0
+ pred_loc_quality
+ all-query best-IoU loss
+ loc quality rerank
```

## R2：Localization + Defect Auxiliary

```text
R1
+ defectness map
+ pixel BCE+Dice
不把 semantic score用于推理
```

## R3：Query Semantic Alignment

```text
R2
+ ROI top-k semantic score
+ query semantic loss
+ semantic quality rerank
```

## R4：Final-Score Background Ranking

```text
R3
+ final-score hard background selection
+ pairwise ranking loss
```

所有实验保持：

```text
相同数据
相同 image size=960
相同预训练权重
相同训练周期
相同增强
相同 seed
相同 evaluator
```

---

# 15. 推荐配置初值

```yaml
DFINETransformer:
  use_loc_quality_head: true
  loc_quality_hidden_dim: 128
  loc_quality_aux_last_n: 0

DEIM:
  use_defectness_head: true
  defectness_in_channels: 256
  defectness_hidden_channels: 128
  use_query_semantic: true
  semantic_roi_size: 7
  semantic_topk_ratio: 0.20
  semantic_detach_boxes: true
  return_defect_map_in_eval: false

DEIMCriterion:
  losses:
    - mal
    - boxes
    - local
    - loc_quality
    - defect
    - query_semantic
    - final_bg_rank

  loc_near_iou_threshold: 0.10
  loc_near_weight: 1.0
  loc_far_weight: 0.10
  loc_matched_extra_weight: 1.0

  defect_bce_weight: 1.0
  defect_dice_weight: 1.0

  semantic_query_pos_weight: 1.0
  semantic_query_bg_weight: 0.25
  semantic_query_bg_iou_threshold: 0.10

  use_final_bg_rank: true
  final_bg_topk: 5
  final_bg_iou_threshold: 0.10
  final_bg_start_epoch: 10
  final_bg_margin: 0.05
  final_bg_temperature: 0.10

  weight_dict:
    loss_loc_quality: 0.5
    loss_defect: 0.5
    loss_query_semantic: 0.25
    loss_final_bg_rank: 0.05

PostProcessor:
  loc_quality_rerank: true
  loc_quality_power: 0.25
  semantic_quality_rerank: true
  semantic_quality_power: 0.25
```

真实原 MAL、bbox、GIoU、FGL、DDF 权重继续继承当前 baseline。

---

# 16. 单元测试

## 16.1 All-query IoU target

验证：

1. 完全重合=1；
2. 无交集=0；
3. 一个 query 对多个 GT取最大IoU；
4. 空 GT全0；
5. batch内不同GT数量；
6. AMP无NaN；
7. target不反传到boxes。

## 16.2 Query semantic pooling

构造 defect map：

1. ROI完全落在高值区域，score高；
2. ROI完全在背景，score低；
3. 细线区域使用top-k后仍能得到高score；
4. 改变topk_ratio结果合理；
5. 越界box会被clamp；
6. 空batch/query可处理；
7. box detach时不产生box梯度；
8. defect logits有梯度。

## 16.3 Final score

验证：

```text
class相同，loc更高 → final更高
class相同，semantic更高 → final更高
power=0 → 对应分支不改变排序
融合发生在top-k前
```

## 16.4 Background ranking

验证：

1. 候选只来自 unmatched far-bg；
2. 按 final score选，而不是 `p(1-q)`；
3. topk=5生效；
4. 正分数高于背景+margin时loss很小；
5. 背景分数更高时loss增大；
6. 无正或无候选返回0；
7. 梯度能回到分类、loc quality、defectness。

## 16.5 Forward/backward

检查输出：

```text
pred_logits      [B,Q,9]
pred_boxes       [B,Q,4]
pred_loc_quality [B,Q,1]
pred_sem_quality [B,Q,1]
pred_defect_logits [B,1,Hd,Wd]  # training
```

检查全部新loss有限且有梯度。

---

# 17. 评估与诊断工具

新增统一脚本：

```bash
python tools/wood/evaluate_sqalign.py \
  --config ... \
  --checkpoint ... \
  --output-dir ...
```

必须输出：

```text
COCO AP/AP50/AP75/APM/APL
逐类AP/AP50/AP75
固定阈值混淆矩阵
background FP
Recall@IoU
ECE/LaECE
all-query loc quality-best-IoU Spearman
top100 loc quality-best-IoU Spearman
TP/background FP loc quality分布
TP/background FP semantic score分布
semantic score TP-vs-background ROC-AUC
final score TP-vs-background ROC-AUC
defect map Dice/precision/recall
```

新增 power sweep：

```bash
python tools/wood/sweep_score_powers.py \
  --config ... \
  --checkpoint ... \
  --loc-powers 0 0.1 0.25 0.5 0.75 \
  --sem-powers 0 0.1 0.25 0.5 \
  --split val
```

只在验证集选超参数，测试集只评估一次固定组合。

---

# 18. Codex 必须执行的 Smoke Test

顺序：

```text
1. 单元测试
2. 2张图forward
3. 2张图backward
4. 10 iterations
5. 100 iterations
6. 1 epoch
```

检查：

```text
loss_loc_quality
loss_defect
loss_query_semantic
loss_final_bg_rank
loc quality近目标均值
loc quality远背景均值
semantic正query均值
semantic背景query均值
每图rank背景数量
```

不得直接启动完整训练。

---

# 19. 日志要求

每 epoch 至少记录：

```text
loss_loc_quality
loss_defect
loss_query_semantic
loss_final_bg_rank

metric_best_iou_mean
metric_locq_pred_near
metric_locq_pred_far
metric_locq_pred_matched

metric_semq_pred_pos
metric_semq_pred_far_bg
metric_semq_pos_bg_gap

metric_final_bg_selected_per_image
metric_final_bg_score_mean
metric_matched_final_score_mean
metric_rank_violation_ratio

metric_valid_mask_ratio
```

---

# 20. 工程验收标准

必须满足：

```text
[ ] 旧 SQ-MAL-v1 接口已删除，不存在兼容wrapper
[ ] 主分类使用原 MAL
[ ] 所有query都有best-IoU连续target
[ ] defectness map进入query semantic score
[ ] semantic score在推理中实际参与top-k前排序
[ ] 背景选择使用最终分数
[ ] 背景损失为排序损失，不是旧全0 BCE
[ ] 新loss只作用于main分支
[ ] 单元测试通过
[ ] forward/backward通过
[ ] AMP和双GPU smoke通过
[ ] COCO预训练权重可加载
[ ] 生成完整实现报告
```

---

# 21. 实验初步成功标准

在 quick-balanced 数据集上，相对当前 Q2：

```text
Q2参考：
AP=50.77
AP50=81.51
AP75=53.39
Precision@0.25=57.75
Recall@0.25=82.28
高阈值background FP=1289
低阈值background FP=2912
all-query loc quality-best-IoU Spearman=0.229
```

R4 建议至少达到：

```text
AP >= 51.2
AP75 >= 54.0
AP50下降不超过0.3
Recall@0.5下降不超过0.3个百分点
高阈值background FP <= 1160
低阈值background FP <= 2500
all-query loc quality-best-IoU Spearman >= 0.40
top100 Spearman >= 0.50
semantic TP-vs-background ROC-AUC >= 0.75
TP semantic均值 - background FP semantic均值 >= 0.15
```

若 R3优于R4，则删除 final-bg rank，不强行保留。

---

# 22. Codex 最终报告

生成：

```text
docs/SQ_ALIGN_V2_IMPLEMENTATION_REPORT.md
```

必须包含：

1. 修改前 commit；
2. 修改后 commit；
3. 删除文件和接口；
4. 新增文件；
5. 每个核心函数及shape；
6. 新损失公式；
7. 新配置说明；
8. 单元测试结果；
9. smoke输出；
10. 预训练权重加载结果；
11. 运行命令；
12. 已知风险；
13. R0～R4功能表；
14. 未完成项；
15. 不应提交的模型权重和大文件。

---

# 23. 推荐提交顺序

```text
commit 1: refactor: remove obsolete SQ-MAL v1 paths
commit 2: feat: add all-query localization quality targets
commit 3: feat: connect defectness map to query semantic evidence
commit 4: feat: add dual-quality score composition
commit 5: feat: add final-score background ranking
commit 6: config: add clean SQ-Align ablations
commit 7: test: add unit and smoke tests
commit 8: docs: add SQ-Align implementation report
```

---

## 完成定义

只有以下条件全部满足，才算完成：

```text
v1失效逻辑已真正删除
all-query质量监督已实现
semantic map已真正进入query评分
最终分数与背景挖掘完全一致
原MAL未被替换
R0-R4消融配置可启动
单元测试和smoke全部通过
实现报告完整
```
