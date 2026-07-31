# SQ-Align V3：Codex 破坏性重构与统一评估实现规格书

> 目标仓库：用户本地 `DEIM`，当前代码基于 `https://github.com/xhx-afk/article/tree/sq-mal-v2`  
> 目标：直接重构当前代码，解决 SQ-Align V2 已经被实验确认的失效点  
> 重要授权：**不需要兼容 SQ-MAL-v1 或 SQ-Align-v2；允许删除、重命名、覆盖旧接口、旧配置和旧 checkpoint key。**  
> 本阶段重点：像素语义监督、query 条件化语义证据、连续 mask 质量、近 GT 候选排序、统一 JSON 诊断。  
> 本阶段禁止：重新引入旧 Localization Quality、旧 far-background ranking、旧 semantic top-k 直接乘分。

---

# 0. 最终任务

将当前 SQ-Align V2 重构为：

> **SQ-Align V3 / CMSQ-DEIM**  
> Continuous Mask-Supported Query Alignment for DEIM-D-FINE

新方法由四部分构成：

1. **BDA：Binary Defect Auxiliary**  
   保留已经被实验验证有效的二值缺陷像素监督。

2. **QCSE：Query-Conditioned Semantic Evidence**  
   让 decoder query 与 encoder pixel embedding 在预测框 ROI 内交互，生成每个 query 独立的语义响应，而不是所有 query 共享一张 union defect probability map。

3. **CMSQ：Continuous Mask-Supported Quality**  
   使用 GT instance mask 对预测框的覆盖率与相对密度构造连续 query 语义质量，不再把所有 matched query 的语义 target 固定为 1。

4. **NGCR：Near-GT Candidate Ranking**  
   围绕每个 GT 对邻近候选进行定位排序和类别排序，直接处理当前占多数的 near-GT duplicate/class FP；不再只优化 `best_iou < 0.1` 的远背景。

训练后必须提供一个权重评估脚本，将全部诊断指标汇总到一个 JSON。

---

# 1. 已确认的 V2 问题

## 1.1 Power sweep 结论

当前结果：

```text
R0 baseline AP = 50.238

R1 loc-quality:
最佳 loc_power=0
AP=49.793
任何正 loc_power 都进一步降低 AP 与 Final AUC

R2 loc + defect aux:
最佳 AP=50.123
loc_power=0 与 0.15 几乎相同
说明改善主要来自 defect auxiliary，而不是 loc quality

R3 query semantic:
最佳 loc=0, sem=0.05
AP=50.447
R3 loc=0, sem=0 已达 50.429
说明主要收益来自训练期 semantic supervision，推理 semantic 乘分仅贡献约 0.018 AP

R4 final-score far-background rank:
最佳 AP=49.884
所有 power 组合均明显低于 R3
说明 ranking 训练逻辑本身有害
```

## 1.2 Localization Quality 失效

当前 V2：

```text
all-query loc-quality vs best-IoU Spearman ≈ 0.05
top100 Spearman ≈ 0.02～0.03
```

TP、far-background FP、near-GT FP 的 loc quality 均集中于约 0.74～0.80，不能排序定位质量。

原因：

- head 只读取 decoder query feature；
- 不读取 D-FINE `pred_corners` 四边分布；
- matched query 同时进入 near loss 与 matched loss，被重复计权；
- BCE 回归质量但不直接优化候选排序；
- 推理乘入 loc quality 后，Bkg FP 下降但 AP 与 Final AUC 单调下降。

**V3 必须彻底删除该分支，不做兼容。**

## 1.3 当前 Semantic 分支不充分

有效信号：

```text
Defect map Dice ≈ 0.816
R3 远背景 FP 明显下降
R3 APM 与 AP75 改善
```

失效点：

- 所有 matched query 的 semantic target 固定为 1；
- 7×7 ROI 最高 20% 均值直接作为 semantic probability；
- TP semantic median 接近 0.998；
- far-background FP semantic median仍约 0.92；
- union defect map 与 query 无关；
- semantic AUC 只对 TP vs far-background 计算，未覆盖占多数的 near-GT FP；
- semantic raw power 越大，FP 越少，但 AP 不提升。

## 1.4 当前 R4 排序对象错误

当前只选择：

```text
unmatched query
且 best IoU < 0.1
```

实际错误中 near-GT duplicate/class FP 约占 70%～75%。

当前 ranking 还使用最弱 matched positive 作为正样本，可能抬高 Dense O2O 的低质量匹配，与 MAL 的质量设计冲突。

**V3 必须删除旧 far-background ranking。**

---

# 2. 破坏性重构授权

允许并建议删除：

```text
QueryLocalizationQualityHead
pred_loc_quality
dec_loc_quality_head
loss_loc_quality
build_all_query_best_iou_target  # 可重写为同时返回 best_gt_index
compose_dual_quality_scores
loc_quality_rerank
loc_quality_power

build_query_semantic_supervision  # 当前 matched=1 版本
pool_query_semantic_evidence      # 当前 raw top-20% mean 版本
select_final_score_background
final_score_ranking_loss
loss_final_bg_rank

configs/deim_dfine/ablation_sqalign/r0～r4
旧 sweep/evaluate 中只针对 V2 的字段
所有 V1/V2 compatibility wrapper
```

允许：

- 旧 V2 checkpoint 不兼容；
- 修改 decoder 返回字段；
- 删除旧配置；
- 修改现有 `semantic_query_alignment.py` 或直接替换为新文件；
- 当前本地代码被破坏，只要新版本最终通过测试即可。

必须保留：

```text
原 DEIM MAL
D-FINE bbox/FGL/DDF/LQE
semantic RLE 数据链
target["masks"]
target["mask_valid"]
COCO bbox/category/annotation ID
官方 COCO 预训练权重加载能力
```

---

# 3. 新结构总览

```text
Encoder finest feature
        │
        ├── Binary Defect Head
        │       └── union mask BCE + Dice
        │
        └── Semantic Pixel Projection
                ↓
          Pixel embedding E [B,D,Hs,Ws]

Final decoder query Z [B,Q,C]
        ↓ Query projection
Query embedding U [B,Q,D]

Predicted boxes B [B,Q,4]
        ↓ ROIAlign(E, B)
ROI pixel embedding R [B,Q,D,7,7]

Query-conditioned ROI mask:
M_i = sum_d U_i,d × R_i,d,:,:    → [B,Q,7,7]

ROI mask statistics + query feature:
        ↓
Semantic quality logit q_sem [B,Q,1]

Training:
GT instance masks + predicted boxes
        ↓
continuous mask support target
        ↓
ROI mask BCE/Dice + semantic quality loss

Inference:
class score × semantic suppression gate
        ↓
top-k

Optional final stage:
per-GT near-candidate ranking
```

---

# 4. 目标文件结构

建议最终形成：

```text
engine/deim/
  deim.py
  dfine_decoder.py
  deim_criterion.py
  postprocessor.py
  continuous_semantic_alignment.py

configs/deim_dfine/ablation_sqalign_v3/
  v0_baseline.yml
  v1_defect_aux.yml
  v2_query_conditioned_mask.yml
  v3_continuous_semantic_quality.yml
  v4_near_gt_candidate_rank.yml

tools/wood/
  evaluate_sqalign_v3.py
  sweep_semantic_gate_v3.py
  visualize_sqalign_v3.py
  smoke_test_sqalign_v3.py

tests/
  test_continuous_mask_support.py
  test_query_conditioned_roi_mask.py
  test_semantic_suppression_gate.py
  test_near_gt_candidate_rank.py
  test_sqalign_v3_forward_backward.py

docs/
  SQ_ALIGN_V3_IMPLEMENTATION_REPORT.md
```

---

# 5. 新文件 `continuous_semantic_alignment.py`

至少导出：

```python
class DefectnessHead(nn.Module)
class SemanticPixelProjection(nn.Module)
class QuerySemanticProjection(nn.Module)
class SemanticQualityHead(nn.Module)

def pairwise_box_iou_cxcywh(...)
def build_best_gt_assignment(...)
def roi_align_query_pixel_features(...)
def query_conditioned_mask_logits(...)
def pool_gt_instance_roi_masks(...)
def compute_continuous_mask_support(...)
def build_semantic_quality_targets(...)
def semantic_suppression_gate(...)
def build_per_gt_candidate_groups(...)
def near_gt_candidate_rank_loss(...)
def build_union_defect_target(...)
```

要求：

- 类型注解；
- shape 文档；
- AMP 可运行；
- 空 GT、空 query 不报错；
- 全训练热路径不调用 `.cpu().numpy()`；
- 不使用逐 query Python 裁剪原图 mask；
- 所有 box 格式明确是 normalized `cxcywh`；
- 避免 NaN/Inf；
- 所有主函数有单元测试。

---

# 6. BDA：Binary Defect Auxiliary

保留 V2 中已有效的二值 defectness head，但修正评估含义。

## 6.1 Head

```text
Encoder finest feature
→ 1×1 Conv
→ Depthwise 3×3
→ GroupNorm
→ SiLU
→ 1×1 Conv
→ binary defect logits
```

默认：

```yaml
defect_hidden_channels: 128
```

## 6.2 Union target

继续使用有效 instance mask 的并集。

训练低分辨率 target：

```python
adaptive_max_pool2d
```

以保留 Crack 等细线。

损失：

\[
L_{def}=L_{BCE}+L_{Dice}
\]

默认权重：

```yaml
loss_defect: 0.5
```

## 6.3 评估必须区分

统一评估 JSON 同时输出：

```text
low_resolution_Dice
low_resolution_precision
low_resolution_recall

upsampled_original_resolution_Dice
upsampled_original_resolution_precision
upsampled_original_resolution_recall
boundary_F1
```

不得把 max-pool 后低分辨率 Dice 直接描述为原图分割精度。

---

# 7. QCSE：Query-Conditioned Semantic Evidence

## 7.1 Pixel embedding

从最高分辨率 encoder feature产生：

```python
SemanticPixelProjection(
    in_channels=256,
    embed_dim=64,
    downsample=True,
)
```

建议结构：

```text
1×1 Conv 256→64
GroupNorm
SiLU
3×3 depthwise stride=2
1×1 Conv 64→64
```

默认输出约 stride=16，控制内存。

## 7.2 Query embedding

最终 decoder query：

```python
QuerySemanticProjection(
    query_dim=hidden_dim,
    embed_dim=64,
)
```

结构：

```text
Linear
LayerNorm
SiLU
Linear
L2 normalize
```

## 7.3 Decoder 返回最终 query feature

修改 `DFINETransformer`，主输出增加：

```python
outputs["pred_query_features"]  # [B,Q,C]
```

只返回最终 decoder layer：

- 不向 aux 输出复制；
- 不返回 DN query；
- 去除 DN 部分后再返回正常 query；
- deploy 时仍可使用。

不再创建 loc-quality head。

## 7.4 ROIAlign

对 semantic pixel embedding 和预测框：

```python
roi_align(
    semantic_pixel_embedding,
    rois,
    output_size=(7,7),
    spatial_scale=1.0,
    sampling_ratio=2,
    aligned=True,
)
```

输出：

```text
[B,Q,D,7,7]
```

默认：

```yaml
semantic_roi_size: 7
semantic_detach_boxes: true
```

第一版 box 必须 detach，防止 mask loss反向干扰 bbox。

## 7.5 Query-conditioned mask

\[
M_{i,h,w}
=
\frac{1}{\sqrt D}
\sum_d
U_{i,d}R_{i,d,h,w}
\]

实现：

```python
mask_logits = torch.einsum("bqd,bqdhw->bqhw", query_embed, roi_pixel) / sqrt(D)
```

输出：

```text
pred_query_mask_logits [B,Q,7,7]
```

它必须依赖 query embedding，不能退化为所有 query 共用同一 defect map ROI。

---

# 8. Instance ROI mask target

## 8.1 Best-GT assignment

重写 best-IoU 工具，返回：

```python
best_iou: [B,Q]
best_gt_index: [B,Q]  # 无GT时为-1
```

matched query 必须优先使用 Hungarian 指定的 GT；其余 query 使用 best-IoU GT。

## 8.2 ROI GT mask

对每个 query：

- `best_iou >= 0.1`：取关联 GT instance mask；
- far background：全 0 mask；
- invalid GT mask：该 query mask loss忽略。

使用预测框作为 ROI，在 GT instance mask 上做 ROIAlign 到 7×7。

目标使用 soft mask，不要 threshold 后再训练。

## 8.3 采样

避免全部 300 query 都产生强 mask loss。

每图训练：

```text
全部 matched query
最多 20 个 near-GT unmatched query
最多 20 个 far-background query
```

near-GT query按：

```text
class score × best IoU
```

选择。

far-background按 class score选择。

采样 index detach。

## 8.4 Query mask loss

matched：

```text
weight=1.0
```

near-GT unmatched：

```text
weight=0.25
```

far background：

```text
weight=0.10
target=全0
```

使用：

\[
L_{qmask}=L_{BCE}+L_{Dice}
\]

默认总权重：

```yaml
loss_query_mask: 0.25
```

---

# 9. CMSQ：连续 Mask-Supported Quality

## 9.1 不能再将 matched target 固定为 1

对关联 GT mask \(M_j\) 和预测框 \(B_i\)：

\[
coverage_i=
\frac{|M_j\cap B_i|}{|M_j|+\epsilon}
\]

\[
density_i^{pred}=
\frac{|M_j\cap B_i|}{|B_i|+\epsilon}
\]

\[
density_j^{gt}=
\frac{|M_j\cap B_j^{gt}|}{|B_j^{gt}|+\epsilon}
\]

\[
relative\_density_i=
\operatorname{clip}
\left(
\frac{density_i^{pred}}{density_j^{gt}+\epsilon},
0,1
\right)
\]

\[
t_i^{sem}
=
\sqrt{coverage_i\cdot relative\_density_i}
\]

规则：

```text
matched：连续 target
near-GT unmatched：连续 target
far background：0
invalid mask：ignore
```

这个 target只监督 semantic quality，不乘进 MAL。

## 9.2 高效计算

实现积分图或批量 ROI mask统计：

```text
GT mask area
pred box内 GT mask像素
GT bbox内 GT mask像素
pred box area
GT box area
```

不得逐 query裁剪完整 mask。

## 9.3 Semantic statistics

不能再直接把 raw top-20% mean作为最终分数。

从 `sigmoid(pred_query_mask_logits)` 提取：

```text
roi_mean
roi_max
roi_std
top10_mean
top25_mean
positive_ratio_at_05
positive_ratio_at_075
center_mean
border_mean
center_minus_border
```

拼接 decoder query feature的轻量投影：

```text
semantic stats
+ query feature projection
→ MLP
→ pred_sem_quality_logit
```

输出：

```text
pred_sem_quality [B,Q,1]  # logit
```

## 9.4 Semantic quality loss

使用 soft-label BCE：

```python
BCEWithLogits(pred_sem_quality, continuous_target)
```

互斥分组：

```text
matched
near-unmatched
far-background
```

默认：

```yaml
semantic_matched_weight: 1.0
semantic_near_weight: 0.5
semantic_far_weight: 0.10
loss_semantic_quality: 0.25
```

不得让 matched重复进入 near loss。

## 9.5 排序辅助

只对每图高分类分数 top-30 query，若：

```text
target_sem_i > target_sem_j + 0.15
```

要求：

```text
q_sem_i > q_sem_j
```

默认：

```yaml
semantic_rank_topk: 30
semantic_rank_target_gap: 0.15
semantic_rank_margin: 0.03
semantic_rank_temperature: 0.10
loss_semantic_rank: 0.05
```

这是语义质量内部排序，不是旧 far-background ranking。

---

# 10. Semantic Suppression Gate

## 10.1 不再使用幂乘法

删除：

```python
scores *= semantic_quality ** semantic_power
```

新 gate：

\[
q_i=\sigma(z_i^{sem})
\]

\[
g_i=
1-\lambda(1-q_i)^\gamma
\]

\[
S_{i,c}=P_{i,c}\cdot g_i
\]

特点：

- \(q\) 高时基本不增强；
- \(q\) 低时适度抑制；
- 不让未充分校准的 semantic score重新大幅排列 TP。

默认：

```yaml
semantic_gate_enabled: false
semantic_gate_lambda: 0.20
semantic_gate_gamma: 2.0
```

所有融合在 flatten/top-k 前完成。

## 10.2 验证集 sweep

实现：

```text
lambda:
0
0.05
0.10
0.15
0.20
0.30

gamma:
1
2
3
```

只在 val选择，test固定一次。

---

# 11. NGCR：Near-GT Candidate Ranking

旧 far-background ranking全部删除。

## 11.1 候选构建

对每个 GT \(j\)，收集：

```text
IoU(query_i, GT_j) >= 0.30
```

每个 GT 最多保留：

```text
top 10 candidates by true-class score
```

正 query：

- 优先选择与该 GT匹配且 IoU最高的 query；
- 若一个 GT有多个 Dense O2O matched query，只选择 IoU最高者作为 rank anchor；
- 其他 matched query也可作为候选负例，但不强制提升。

## 11.2 定位排序

对于同一 GT 的候选 \(i,k\)，若：

\[
IoU_i > IoU_k + 0.10
\]

要求 true-class final score：

\[
S_{i,y}>S_{k,y}+m_{loc}
\]

默认：

```yaml
candidate_iou_threshold: 0.30
candidate_iou_gap: 0.10
candidate_rank_margin: 0.03
candidate_rank_topk_per_gt: 3
```

## 11.3 类别混淆排序

在 anchor query 上：

\[
S_{anchor,y}
>
\max_{c\neq y}S_{anchor,c}+m_{cls}
\]

默认：

```yaml
class_rank_margin: 0.05
```

## 11.4 梯度范围

第一版：

```python
rank_score =
class_probability
* semantic_gate.detach()
```

ranking loss只更新分类排序，不反向更新 semantic map、query mask或 bbox。

## 11.5 启用时机

```yaml
candidate_rank_start_ratio: 0.60
loss_candidate_rank: 0.05
```

前60%训练不启用。

不得选择最弱 matched positive与最强远背景配对。

---

# 12. Criterion 重构

删除：

```text
loss_loc_quality
loss_final_bg_rank
旧 loss_query_semantic
旧 V2 diagnostics
```

新增：

```python
loss_defect(...)
loss_query_mask(...)
loss_semantic_quality(...)
loss_semantic_rank(...)
loss_candidate_rank(...)
```

主损失：

\[
L=
L_{MAL}
+L_{bbox}
+L_{FGL}
+L_{DDF}
+\lambda_dL_{def}
+\lambda_mL_{qmask}
+\lambda_sL_{semq}
+\lambda_{sr}L_{sem-rank}
+\lambda_{cr}L_{candidate-rank}
\]

分支规则：

| 分支 | MAL | bbox/local | defect | query mask | semantic quality | semantic rank | candidate rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| main | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| decoder aux | ✓ | ✓ | × | × | × | × | × |
| pre | ✓ | ✓ | × | × | × | × | × |
| encoder aux | ✓ | ✓ | × | × | × | × | × |
| DN | ✓ | ✓ | × | × | × | × | × |

不得自动复制新 loss 到 aux/DN。

---

# 13. 干净消融配置

## V0：Baseline

```text
原 DEIM-D-FINE
MAL
无 semantic模块
```

## V1：Defect Auxiliary

```text
V0
+ binary defectness head
+ loss_defect
```

无 query semantic、无 gate、无 ranking。

## V2：Query-Conditioned Mask

```text
V1
+ semantic pixel embedding
+ query embedding
+ query-conditioned ROI mask
+ loss_query_mask
```

无 semantic quality gate。

## V3：Continuous Semantic Quality

```text
V2
+ continuous mask support target
+ semantic quality head
+ loss_semantic_quality
+ loss_semantic_rank
```

训练配置默认 gate关闭；训练后在验证集 sweep gate。

## V4：Near-GT Candidate Ranking

```text
V3
+ per-GT localization/class ranking
```

训练期间 gate可参与 rank score但必须 detach；建议使用 val已选的 gate参数，或训练配置固定 lambda=0.1。

所有实验保持：

```text
相同数据
相同 image size=960
相同预训练权重
相同 epoch
相同增强
相同 batch size
相同 seed
相同评估脚本
```

---

# 14. 统一权重评估脚本

必须实现：

```text
tools/wood/evaluate_sqalign_v3.py
```

运行：

```bash
python tools/wood/evaluate_sqalign_v3.py \
  -c CONFIG \
  -r CHECKPOINT \
  --images-dir IMAGE_DIR \
  --ann-file ANN_JSON \
  --output-json RESULT_JSON \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 4 \
  --max-dets 300 \
  --score-thresholds 0.05 0.10 0.25 0.50 \
  --semantic-gate-lambda 0.10 \
  --semantic-gate-gamma 2.0
```

## 14.1 强制只生成一个主 JSON

所有数值指标必须写入：

```text
RESULT_JSON
```

可以额外保存可视化图片，但不允许把关键数值拆散在多个 JSON/CSV/TXT 中。

## 14.2 JSON 顶层结构

```json
{
  "meta": {},
  "checkpoint_load": {},
  "dataset": {},
  "coco": {},
  "tide": {},
  "fixed_thresholds": {},
  "error_taxonomy": {},
  "confusion": {},
  "recall_by_iou": {},
  "calibration": {},
  "defect_map": {},
  "query_mask": {},
  "semantic_quality": {},
  "candidate_ranking": {},
  "score_analysis": {},
  "runtime": {},
  "automatic_diagnosis": {}
}
```

---

# 15. 统一 JSON 必须包含的指标

## 15.1 Meta

```text
timestamp
git_branch
git_commit
dirty_worktree
config_path
config_sha256
checkpoint_path
checkpoint_sha256
annotation_path
annotation_sha256
image_dir
seed
device
torch_version
cuda_version
num_parameters
trainable_parameters
```

## 15.2 Dataset

```text
num_images
num_annotations
num_classes
per_class_GT
valid_mask_count
invalid_mask_count
valid_mask_ratio
```

## 15.3 COCO

必须输出：

```text
AP@[0.50:0.95]
AP50
AP55
AP60
AP65
AP70
AP75
AP80
AP85
AP90
AP95

APS
APM
APL
AR1
AR10
AR100
ARS
ARM
ARL
```

逐类输出同样的：

```text
AP
AP50
AP75
AP80
AP90
GT count
```

## 15.4 TIDE

若安装 `tidecv`：

```text
Cls dAP
Loc dAP
Both dAP
Dupe dAP
Bkg dAP
Miss dAP
FalsePos dAP
FalseNeg dAP
```

若未安装：

```json
{
  "available": false,
  "reason": "tidecv is not installed"
}
```

评估脚本不能因此崩溃。

## 15.5 Fixed thresholds

对每个：

```text
0.05
0.10
0.25
0.50
```

输出：

```text
TP
FP
FN
precision
recall
F1
background_FP
near_GT_FP
duplicate_same_class_FP
class_confusion_FP
localization_FP
predictions
```

## 15.6 Error taxonomy

明确区分：

```text
TP：
same class，未重复使用GT，IoU>=0.5

far_background_FP：
best any-class IoU<0.1

duplicate_same_class_FP：
与同类GT IoU>=0.5，但GT已被更高分检测占用

class_confusion_FP：
与任意GT IoU>=0.5，但预测类别错误

localization_FP：
0.1<=best IoU<0.5

FN：
没有被same-class IoU>=0.5检测匹配
```

输出 overall 与 per-class 数量。

## 15.7 Confusion

输出：

```text
labels
count_matrix
row_normalized_matrix
column_normalized_matrix
```

全部嵌入 JSON。

## 15.8 Recall

```text
Recall@IoU 0.30/0.40/0.50/0.60/0.70/0.75/0.80/0.85/0.90/0.95
overall
per_class
```

## 15.9 Calibration

overall 与 per-class：

```text
ECE
LaECE
Brier score
mean confidence
accuracy
mean matched IoU
15-bin reliability data
```

## 15.10 Defect map

```text
low_res Dice/precision/recall
original_res Dice/precision/recall
boundary F1
per-class instance-pixel recall
foreground saturation ratio
background activation mean
```

## 15.11 Query mask

matched、near-unmatched、far-background：

```text
ROI mask BCE
ROI mask Dice
ROI mask IoU
positive pixel precision
positive pixel recall
```

逐类输出 matched query mask Dice。

## 15.12 Semantic quality

分组：

```text
TP
far_background_FP
duplicate_same_class_FP
class_confusion_FP
localization_FP
all_FP
```

每组：

```text
count
mean
std
p10
p25
median
p75
p90
ratio_gt_0.90
ratio_gt_0.95
ratio_lt_0.10
```

AUC：

```text
TP vs far background ROC-AUC/PR-AUC
TP vs duplicate ROC-AUC/PR-AUC
TP vs class confusion ROC-AUC/PR-AUC
TP vs localization FP ROC-AUC/PR-AUC
TP vs all FP ROC-AUC/PR-AUC
```

质量 target相关性：

```text
all valid queries Spearman
top10 per image Spearman
top30 per image Spearman
matched Spearman
near-unmatched Spearman
```

## 15.13 Candidate ranking

```text
GT with >=2 candidates
candidate count mean
anchor true-class score
hard candidate true-class score
ranking violation ratio
mean anchor-hard margin
top-score candidate mean IoU
highest-IoU candidate selected ratio
per-class violation ratio
```

## 15.14 Score analysis

分别评估：

```text
class_score
class_score × gate
```

输出：

```text
TP vs all-FP ROC-AUC
TP vs all-FP PR-AUC
AP
AP50
AP75
background FP
near-GT FP
```

这样可以判断 gate 是否真正改善排序，而不是只减少固定阈值预测数。

## 15.15 Runtime

```text
warmup iterations
timed iterations
batch size
mean latency ms/image
p50 latency
p95 latency
FPS
peak CUDA memory
```

## 15.16 Automatic diagnosis

根据指标自动写入字符串数组，例如：

```json
{
  "warnings": [
    "semantic_quality_saturated",
    "semantic_gate_reduces_fp_but_hurts_ap",
    "near_gt_fp_dominates",
    "query_mask_not_learning",
    "candidate_rank_violation_high"
  ],
  "summary": [
    "..."
  ]
}
```

判定规则集中定义，不要散落。

---

# 16. Gate sweep 脚本

实现：

```text
tools/wood/sweep_semantic_gate_v3.py
```

命令：

```bash
python tools/wood/sweep_semantic_gate_v3.py \
  -c CONFIG \
  -r CHECKPOINT \
  --images-dir VAL_IMAGES \
  --ann-file VAL_JSON \
  --lambdas 0 0.05 0.10 0.15 0.20 0.30 \
  --gammas 1 2 3 \
  --output-json gate_sweep.json
```

gate sweep JSON 至少包含：

```text
lambda
gamma
AP
AP50
AP75
AP80
AP90
background FP
near-GT FP
LaECE
TP-vs-all-FP AUC
```

排序优先级：

```text
1. AP
2. AP75
3. TP-vs-all-FP AUC
4. background FP
```

只在 val选择。

---

# 17. 单元测试

## 17.1 Continuous support

测试：

```text
pred box=GT box → target接近1
只覆盖半个mask → target下降
过大box → relative density下降
纯背景 → 0
细线mask正确box不应接近0
invalid mask → ignore
空GT不报错
```

## 17.2 Query-conditioned mask

测试：

```text
不同query embedding产生不同mask
相同query+不同ROI产生不同mask
梯度回到pixel projection与query projection
box detach后无box梯度
输出shape正确
```

## 17.3 Semantic statistics/head

测试：

```text
全背景ROI quality低
完整前景ROI quality高
少数孤立高点不能直接使quality接近1
soft target BCE正常
```

## 17.4 Gate

测试：

```text
q=1 → gate=1
q=0 → gate=1-lambda
lambda=0 →原分数
gate只抑制不放大
发生在top-k前
```

## 17.5 Candidate ranking

测试：

```text
高IoU候选应排在低IoU候选前
错误类别分数过高时loss增大
无足够候选返回0
只更新class score
```

## 17.6 Unified evaluator

使用10张图测试：

```text
生成一个JSON
所有顶层字段存在
JSON可序列化
无NaN/Infinity
checkpoint hash存在
COCO和fixed threshold数值存在
```

---

# 18. Smoke Test

顺序：

```text
1. 单元测试
2. batch=2 forward
3. batch=2 backward
4. 10 iterations
5. 100 iterations
6. 1 epoch
7. 用生成的权重运行统一 evaluator
```

输出检查：

```text
pred_logits [B,Q,9]
pred_boxes [B,Q,4]
pred_query_features [B,Q,C]
pred_defect_logits [B,1,Hd,Wd]
pred_query_mask_logits [B,Q,7,7]
pred_sem_quality [B,Q,1]
```

loss：

```text
loss_mal
loss_bbox
loss_giou
loss_fgl
loss_ddf
loss_defect
loss_query_mask
loss_semantic_quality
loss_semantic_rank
loss_candidate_rank
```

---

# 19. 日志要求

每 epoch 至少记录：

```text
loss_defect
loss_query_mask
loss_semantic_quality
loss_semantic_rank
loss_candidate_rank

metric_defect_dice
metric_query_mask_dice_matched
metric_query_mask_dice_near
metric_sem_target_mean
metric_sem_pred_matched
metric_sem_pred_near
metric_sem_pred_far
metric_sem_target_spearman
metric_sem_saturation_gt_09

metric_candidate_gt_count
metric_candidate_count_mean
metric_candidate_violation_ratio
metric_candidate_anchor_margin
```

---

# 20. 训练命令配置

Codex 只创建配置与命令说明，不启动完整训练。

示例：

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7840 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign_v3/v0_baseline.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output_2/sqalign_v3/v0_baseline
```

V1～V4只替换配置、端口和输出目录。

---

# 21. 初步验收目标

以 quick-balanced R0约50.24 AP为参考。

## V1

```text
AP不低于R0
AP50/AP75至少一个提升
Defect map原图Dice有效
```

## V2

```text
V2 >= V1 + 0.2 AP
matched query mask Dice >= 0.65
不同query mask不是完全相同
```

## V3

```text
V3 gate关闭时 >= V2
semantic target Spearman >= 0.35
TP vs all-FP semantic ROC-AUC >= 0.75
background FP semantic median不再接近1
ratio_gt_0.95(background FP)显著下降
```

## V4

```text
V4 >= V3
near-GT FP下降
candidate violation ratio下降
AP与AP75不下降
```

最终候选：

```text
AP >= 51.0
AP75 >= 54.0
AP50下降 <= 0.3
TP-vs-all-FP AUC高于class-only
near-GT FP与far-background FP均下降
```

未达到时不得宣称成功。

---

# 22. Codex 实现报告

生成：

```text
docs/SQ_ALIGN_V3_IMPLEMENTATION_REPORT.md
```

包含：

1. 重构前后 commit；
2. 删除的 V2 文件/接口；
3. 新增与修改文件；
4. 模块结构和 tensor shape；
5. 损失公式；
6. evaluator JSON schema；
7. 单元测试结果；
8. smoke结果；
9. 预训练权重加载结果；
10. 运行命令；
11. 已知风险；
12. 未完成项；
13. 不应提交的大文件。

---

# 23. 推荐 Git 提交顺序

```text
commit 1: refactor: remove failed SQ-Align V2 quality and far-bg rank
commit 2: feat: add query-conditioned semantic ROI masks
commit 3: feat: add continuous mask-supported semantic quality
commit 4: feat: add semantic suppression gate
commit 5: feat: add per-GT near-candidate ranking
commit 6: feat: add unified single-JSON evaluator
commit 7: config: add clean V0-V4 ablations
commit 8: test: add unit and smoke tests
commit 9: docs: add implementation report
```

---

# 24. 直接发送给 Codex 的提示词

```text
请读取仓库根目录的 SQ_ALIGN_V3_Codex_Destructive_Refactor_Spec.md，并直接重构当前本地 sq-mal-v2 代码。

重要要求：
1. 不需要兼容SQ-MAL-v1或SQ-Align-v2。
2. 可以删除、重命名和覆盖旧接口、旧配置、旧checkpoint key。
3. 必须删除当前无效的Localization Quality和far-background ranking。
4. 主分类继续使用原DEIM MAL。
5. 保留已验证有效的binary defect auxiliary。
6. 按文档实现：
   - query-conditioned semantic ROI mask；
   - 连续mask-supported semantic target；
   - learned semantic quality，而不是raw top-20% mean；
   - 只抑制低语义证据的semantic gate；
   - per-GT near-candidate ranking。
7. 实现训练后权重统一评估脚本，所有排错指标写入一个JSON。
8. 统一JSON必须包含COCO全阈值、TIDE、固定阈值错误分类、校准、defect map、query mask、semantic质量、candidate ranking、runtime和自动诊断。
9. 先完成测试、smoke、配置和实现报告，不要启动完整训练。
10. 最终列出删除/修改/新增文件、测试结果、命令和风险。
```

---

## 完成定义

只有同时满足：

```text
旧loc-quality彻底删除
旧far-bg ranking彻底删除
query-conditioned semantic evidence已实现
continuous mask target已实现
gate只抑制不放大
near-GT ranking已实现
V0-V4可启动
权重评估可生成单一完整JSON
单元测试与smoke全部通过
实现报告完整
```

任务才算完成。
