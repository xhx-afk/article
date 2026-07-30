# SQ-Align V2：用户验收与效果排查手册

> 用途：Codex 完成直接重构后，用户按本手册检查代码、数据链、训练行为和最终效果。  
> 配套文档：`SQ_ALIGN_V2_Codex_Refactor_Spec.md`  
> 当前参考最优：Q2 defect auxiliary，AP=50.77。

---

# 1. 本次不再验收什么

本次不要求：

- SQ-MAL-v1 checkpoint继续加载；
- Q1～Q4旧配置继续运行；
- `use_sqmal` 或 `loss_sqmal` 保留；
- 旧 `pred_quality` key保留；
- 旧 `sqmal.py` 保留；
- 旧 hard-bg 结果可复现。

若 Codex为了兼容v1保留大量旧接口，应要求删除后再验收。

本次真正需要验收：

```text
all-query localization quality
query-level semantic evidence
dual-quality final score
final-score-aligned background ranking
```

---

# 2. 先备份当前版本

```bash
cd /home/zxw4090/hjw/DEIM

git status
git branch --show-current
git rev-parse HEAD

git add -A
git commit -m "backup: before SQ-Align v2 refactor"
```

备份当前Q0～Q4结果：

```bash
mkdir -p /home/zxw4090/hjw/DEIM/result_backups/sqmal_v1

cp -r output_2/sqmal \
  /home/zxw4090/hjw/DEIM/result_backups/sqmal_v1/
```

重点保存：

```text
q0_baseline(3).json
q1_quality(3).json
q2_defect_aux(3).json
q3_sqmal(3).json
q4_sqmal_hbg(3).json
```

当前Q2参考：

```text
AP=50.7716
AP50=81.5111
AP75=53.3877
Precision@0.25=57.7453%
Recall@0.25=82.2804%
F1=67.8634%
高阈值background FP=1289
低阈值background FP=2912
LaECE=8.2554%
all-query quality-best-IoU Spearman=0.2288
```

---

# 3. 将文档发送给 Codex

将：

```text
SQ_ALIGN_V2_Codex_Refactor_Spec.md
```

放到：

```text
/home/zxw4090/hjw/DEIM/
```

发送提示词：

```text
请读取仓库根目录的 SQ_ALIGN_V2_Codex_Refactor_Spec.md，并直接重构当前本地代码。

不需要兼容SQ-MAL-v1，可以删除或覆盖旧接口、配置和文件。
不要保留旧use_sqmal、loss_sqmal、semantic support乘法目标和旧hard-bg。
主分类继续使用原DEIM MAL。
先完成代码、单元测试、smoke test、R0-R4配置和实现报告，不要直接完整训练。
```

---

# 4. Codex 完成后检查文件

预期存在：

```text
engine/deim/semantic_query_alignment.py
configs/deim_dfine/ablation_sqalign/
tests/test_all_query_iou_target.py
tests/test_query_semantic_pooling.py
tests/test_dual_quality_score.py
tests/test_background_ranking.py
tests/test_sqalign_forward_backward.py
docs/SQ_ALIGN_V2_IMPLEMENTATION_REPORT.md
```

检查旧接口是否删除：

```bash
grep -R "loss_labels_sqmal" -n engine configs || true
grep -R "use_sqmal" -n engine configs || true
grep -R "semantic_beta" -n engine configs || true
grep -R "compute_semantic_support" -n engine || true
grep -R "compute_sq_quality_target" -n engine || true
grep -R "confidence.*1.*quality" -n engine || true
```

这些命令原则上不应返回有效代码。

检查新接口：

```bash
grep -R "pred_loc_quality" -n engine
grep -R "pred_sem_quality" -n engine
grep -R "build_all_query_best_iou_target" -n engine
grep -R "pool_query_semantic_evidence" -n engine
grep -R "final_score_ranking_loss" -n engine
```

---

# 5. 代码级关键验收

## 5.1 主分类必须回到原 MAL

检查配置：

```bash
grep -R 'losses:' -A15 configs/deim_dfine/ablation_sqalign
```

R1～R4应包含：

```text
mal
boxes
local
```

不应包含：

```text
sqmal
```

## 5.2 all-query quality target

打开 criterion 或工具函数，确认 target是：

```text
每个query与所有GT的最大IoU
```

不是：

```text
matched query=IoU
unmatched query=0
```

运行单元测试后，应明确出现：

```text
duplicate/near-GT unmatched query可以得到非零quality target
far background query target接近0
```

## 5.3 defectness map必须进入query决策

确认前向中存在：

```text
defect logits
+ pred boxes
→ ROIAlign/top-k
→ pred_sem_quality
```

不能只是：

```text
defect logits
→ loss_defect
```

## 5.4 最终分数

确认 postprocessor：

```text
class score
× loc quality^eta
× semantic quality^delta
→ flatten/top-k
```

必须在 top-k 前执行。

## 5.5 背景挖掘

确认候选排序使用：

```text
最终分数
```

不能再使用：

```text
class confidence × (1-quality)
```

确认 loss 是正负最终分数排序，不是旧全类别全0 BCE。

---

# 6. 运行单元测试

```bash
cd /home/zxw4090/hjw/DEIM
conda activate deim_mamba   # 按实际DEIM环境修改
```

运行：

```bash
pytest -q tests/test_all_query_iou_target.py
pytest -q tests/test_query_semantic_pooling.py
pytest -q tests/test_dual_quality_score.py
pytest -q tests/test_background_ranking.py
pytest -q tests/test_sqalign_forward_backward.py
```

若环境没有pytest，Codex应提供统一脚本。

任何测试失败都不要进入训练。

---

# 7. 数据链复查

现有 semantic RLE 数据链若已验证，不必重新生成。

当前 quick 数据：

```text
images/test：794张
instances_test_sqmal.json：2596个annotation
```

仍需随机检查：

```bash
python tools/wood/visualize_sqmal_masks.py \
  --images-dir /home/zxw4090/hjw/D-FINE/data/WoodDefect/wood_coco_all_only_defect_quick_balanced_4000/images/train \
  --coco-json /home/zxw4090/hjw/DEIM/quick_data_debug/instances_train_sqmal.json \
  --output-dir /home/zxw4090/hjw/DEIM/quick_data_debug/visual_recheck \
  --num-images 100 \
  --seed 42
```

重点检查：

- Crack细线没有消失；
- resin多个连通域没有错配；
- knot类实例没有合并；
- `mask_valid` 与 boxes数量一致；
- Mosaic/Resize后没有错位。

---

# 8. Forward/Backward Smoke

Codex应提供：

```bash
python tools/wood/smoke_test_sqalign.py \
  -c configs/deim_dfine/ablation_sqalign/r4_final_score_rank.yml \
  -t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
  --batch-size 2 \
  --device cuda
```

检查输出：

```text
pred_logits       [2,Q,9]
pred_boxes        [2,Q,4]
pred_loc_quality  [2,Q,1]
pred_sem_quality  [2,Q,1]
pred_defect_logits[2,1,Hd,Wd]
```

检查loss：

```text
loss_mal
loss_bbox
loss_giou
loss_fgl
loss_ddf
loss_loc_quality
loss_defect
loss_query_semantic
loss_final_bg_rank
```

检查梯度：

```text
classification head：非零
bbox head：非零
loc quality head：非零
defectness head：非零
encoder：非零
```

---

# 9. 先运行100 iteration

依次跑 R1～R4 debug。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
  -c configs/deim_dfine/ablation_sqalign/r1_all_query_loc_debug.yml \
  --use-amp \
  --seed=0 \
  -t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
  --output-dir output_2/sqalign_v2/debug_r1
```

观察：

```text
loss_loc_quality是否下降
near query loc quality是否高于far query
semantic正query是否高于背景query
rank violation是否下降
loss是否出现NaN
显存是否异常增长
```

健康趋势参考：

```text
locq_pred_near > locq_pred_far
semq_pred_pos > semq_pred_far_bg
matched_final_score > final_bg_score
rank_violation_ratio逐渐下降
```

---

# 10. 正式消融顺序

输出根目录：

```bash
mkdir -p output_2/sqalign_v2
```

## R0 Baseline

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7830 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign/r0_baseline.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output_2/sqalign_v2/r0_baseline \
2>&1 | tee output_2/sqalign_v2/r0_baseline/train_console.log
```

## R1 All-Query Localization

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7831 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign/r1_all_query_loc.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output_2/sqalign_v2/r1_all_query_loc \
2>&1 | tee output_2/sqalign_v2/r1_all_query_loc/train_console.log
```

## R2 Loc + Defect Auxiliary

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7832 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign/r2_loc_defect_aux.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output_2/sqalign_v2/r2_loc_defect_aux \
2>&1 | tee output_2/sqalign_v2/r2_loc_defect_aux/train_console.log
```

## R3 Query Semantic

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7833 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign/r3_query_semantic.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output_2/sqalign_v2/r3_query_semantic \
2>&1 | tee output_2/sqalign_v2/r3_query_semantic/train_console.log
```

## R4 Final-Score Rank

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7834 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign/r4_final_score_rank.yml \
--use-amp \
--seed=0 \
-t ./weight/deim_dfine_hgnetv2_l_coco_50e.pth \
--output-dir output_2/sqalign_v2/r4_final_score_rank \
2>&1 | tee output_2/sqalign_v2/r4_final_score_rank/train_console.log
```

实际epoch、数据路径、batch size继承当前baseline，不额外修改。

---

# 11. 每个实验统一评估

```bash
python tools/wood/evaluate_sqalign.py \
  -c configs/deim_dfine/ablation_sqalign/r4_final_score_rank.yml \
  -r output_2/sqalign_v2/r4_final_score_rank/best_stg2.pth \
  --output-dir output_2/sqalign_v2/eval_test/r4_final_score_rank
```

每个实验必须输出：

```text
result.json
confusion_matrix.png
confusion_matrix_with_fp_fn.png
per_class_metrics.json
background_fp.json
recall_by_iou.json
calibration.json
query_quality.json
semantic_query.json
defectness_metrics.json
```

---

# 12. Power Sweep

对 R1：

```text
loc_power = 0, 0.1, 0.25, 0.5, 0.75
sem_power = 0
```

对 R3/R4：

```text
loc_power = 0.1, 0.25, 0.5
sem_power = 0, 0.1, 0.25, 0.5
```

命令：

```bash
python tools/wood/sweep_score_powers.py \
  -c configs/deim_dfine/ablation_sqalign/r3_query_semantic.yml \
  -r output_2/sqalign_v2/r3_query_semantic/best_stg2.pth \
  --loc-powers 0.1 0.25 0.5 \
  --sem-powers 0 0.1 0.25 0.5 \
  --split val \
  --output-dir output_2/sqalign_v2/r3_query_semantic/power_sweep
```

只能用验证集选 power。

选定后固定配置，再在 test 评估一次。

---

# 13. R1 验收：all-query定位质量是否真正改善

Q2旧值：

```text
all-query Spearman=0.2288
```

R1目标：

```text
all-query Spearman >= 0.40
top100 Spearman >= 0.50
TP loc quality均值明显高于background FP
```

若仍低于0.30，优先排查：

1. 是否真的对全部query计算best IoU；
2. 是否误把 unmatched target继续置0；
3. far background权重是否过大；
4. head输入是否用了错误decoder层；
5. 评估是否在重排序后query上错误配对；
6. boxes是否使用正确cxcywh归一化格式。

调参顺序：

```text
loc_far_weight: 0.10 → 0.05
loc_matched_extra_weight: 1.0 → 2.0
loss_loc_quality: 0.5 → 1.0
loc_quality_hidden_dim: 128 → 256
```

一次只改一个变量。

---

# 14. R2 验收：defectness辅助是否仍有效

检查像素指标：

```text
Dice
precision
recall
```

查看可视化：

```bash
python tools/wood/visualize_defectness_and_query_scores.py \
  -c configs/deim_dfine/ablation_sqalign/r2_loc_defect_aux.yml \
  -r output_2/sqalign_v2/r2_loc_defect_aux/best_stg2.pth \
  --output-dir output_2/sqalign_v2/r2_loc_defect_aux/visuals \
  --num-images 100
```

重点查看：

- 正常木纹是否大面积高响应；
- Crack细线是否连续；
- resin是否只响应框中心而漏掉长条区域；
- Quartzity是否与Marrow混淆；
- Blue_stain是否因样本少而完全不响应。

R2应至少不低于R1。

若R2下降：

```text
loss_defect: 0.5 → 0.25
Dice权重: 1.0 → 0.5
检查union mask降采样
检查Mosaic后的mask
```

---

# 15. R3 验收：semantic map是否真正进入query

必须输出：

```text
semantic score TP均值
semantic score background FP均值
semantic score far-bg均值
TP-vs-background ROC-AUC
TP-vs-background PR-AUC
```

目标：

```text
ROC-AUC >= 0.75
TP均值 - background FP均值 >= 0.15
```

若 defect map像素指标好，但query semantic区分差：

1. 检查ROI坐标；
2. 检查box是否从归一化正确映射到defect map；
3. 调整：
   - roi_size 7→9；
   - topk_ratio 0.20→0.10；
4. 检查细长缺陷是否需要更小top-k；
5. 不要使用全框平均；
6. 检查pred_sem_quality是否误做两次sigmoid。

R3的关键判定：

```text
R3必须优于R2，才能证明query级semantic alignment有效。
```

若只减少FP但AP下降，先降低：

```text
semantic_quality_power: 0.25 → 0.10
loss_query_semantic: 0.25 → 0.10
semantic_query_bg_weight: 0.25 → 0.10
```

---

# 16. R4 验收：背景排序是否真正对齐AP

确认选出的背景是：

```text
最终分数最高
且max IoU<0.1
```

查看：

```text
selected final-bg score
matched positive final score
rank violation ratio
```

目标：

```text
rank violation逐步下降
高阈值background FP进一步下降
Recall基本不变
AP高于R3
```

若低阈值FP下降但AP不升，说明仍在压不重要的低排序背景。

调整顺序：

```text
final_bg_topk: 5 → 3
loss_final_bg_rank: 0.05 → 0.025
margin: 0.05 → 0.02
start_epoch: 10 → 15
```

若R4不如R3：

> 直接删除背景排序模块，以R3作为最终方法。不要为了“完整四模块”强行保留。

---

# 17. 统一结果表

| 实验     |    AP |  AP50 |  AP75 |   APM |   APL | Precision | Recall | 高分Bkg FP | 低分Bkg FP | Loc Spearman | Sem AUC |
| -------- | ----: | ----: | ----: | ----: | ----: | --------: | -----: | ---------: | ---------: | -----------: | ------: |
| Q2旧参考 | 50.77 | 81.51 | 53.39 | 33.90 | 55.81 |     57.75 |  82.28 |       1289 |       2912 |        0.229 |      无 |
| R0       |       |       |       |       |       |           |        |            |            |              |         |
| R1       |       |       |       |       |       |           |        |            |            |              |         |
| R2       |       |       |       |       |       |           |        |            |            |              |         |
| R3       |       |       |       |       |       |           |        |            |            |              |         |
| R4       |       |       |       |       |       |           |        |            |            |              |         |

逐类：

| 实验     | Live AP | Dead AP | resin AP | Crack AP | Quartzity AP | Knot_missing AP | Blue AP |
| -------- | ------: | ------: | -------: | -------: | -----------: | --------------: | ------: |
| Q2旧参考 |   41.13 |   46.54 |    45.36 |    47.69 |        51.80 |           52.15 |   44.45 |
| R0       |         |         |          |          |              |                 |         |
| R1       |         |         |          |          |              |                 |         |
| R2       |         |         |          |          |              |                 |         |
| R3       |         |         |          |          |              |                 |         |
| R4       |         |         |          |          |              |                 |         |

---

# 18. 初步成功标准

相对旧Q2，最终候选模型建议满足：

```text
AP >= 51.2
AP75 >= 54.0
AP50下降 <= 0.3
Recall@0.5下降 <= 0.3个百分点
高阈值background FP <= 1160
低阈值background FP <= 2500
all-query loc Spearman >= 0.40
top100 loc Spearman >= 0.50
semantic ROC-AUC >= 0.75
```

单seed达到后，再跑3个seed。

---

# 19. 三随机种子

仅对：

```text
R0
最佳R3或R4
```

运行：

```text
seed=0
seed=1
seed=2
```

报告：

```text
mean ± std
```

建议论文级标准：

```text
AP均值提升 >= 0.8
3个seed方向一致
AP75均值提升 >= 1.0
background FP稳定下降
```

---

# 20. 完整数据集复验

quick subset只用于验证机制。

最终必须回到完整数据：

```text
完整train
完整val
完整test
```

不得只用794张测试图作为论文主结果。

完整数据复验时保持：

```text
image size=960
数据划分不变
训练周期不变
预训练权重不变
只替换模型模块
```

之后再进行第二方向 MB-FDR，不能同时混入本次实验。

---

# 21. 最终保留规则

## 保留R1

满足：

```text
all-query相关性明显提高
AP不低于R0
```

## 保留R2

满足：

```text
R2不低于R1
像素缺陷图有效
```

## 保留R3

满足：

```text
R3稳定优于R2
semantic score能区分TP与背景FP
```

## 保留R4

满足：

```text
R4稳定优于R3
background FP进一步下降
Recall无明显损失
```

任何模块不满足，就从最终模型删除。

---

# 22. 常见故障速查

| 现象                      | 优先排查                               |
| ------------------------- | -------------------------------------- |
| all-query Spearman仍接近0 | target仍是matched-only；box格式错误    |
| semantic score全部很高    | defect map背景过亮；ROI top-k过激      |
| semantic score全部很低    | ROI坐标错误；重复sigmoid；细线被下采样 |
| AP50下降                  | loc/sem power过大；背景loss过强        |
| AP75不升                  | loc quality未学到IoU；只学到前景性     |
| FP降但AP不升              | 压的是低排名FP，不是top-ranked FP      |
| Recall下降                | near-GT unmatched被错误标为背景        |
| resin/Crack下降           | 使用全框平均；top-k比例过大            |
| Blue_stain波动大          | 样本只有20个，需多seed和置信区间       |
| 显存大幅上涨              | 对aux/DN重复ROI pooling或新loss        |

---

# 23. 最终验收清单

```text
[ ] 旧SQ-MAL-v1接口删除
[ ] 原MAL作为主分类损失
[ ] all-query best-IoU target正确
[ ] near-GT unmatched target非零
[ ] defectness map进入query semantic score
[ ] ROI top-k对细长缺陷有效
[ ] loc和semantic分数在top-k前融合
[ ] 背景选择按最终分数
[ ] 背景使用排序loss
[ ] 新loss未复制到aux/DN
[ ] 单元测试全部通过
[ ] 100 iteration无NaN/OOM
[ ] R0～R4可独立训练
[ ] 统一评估工具输出完整
[ ] power只在val选择
[ ] quick subset达到初步标准
[ ] 3 seed结果稳定
[ ] 完整数据集复验完成
```

---

## 最终决策原则

本次目标不是保住“SQ-MAL”这个名字，而是得到真正有效的机制。

最终方法可能是：

```text
R3：All-query localization quality
  + binary defectness
  + query semantic alignment
```

也可能是：

```text
R4：R3 + final-score background ranking
```

以实验结果决定，不为了模块数量保留无效设计。
