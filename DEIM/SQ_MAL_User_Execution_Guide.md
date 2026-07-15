# SQ-MAL 用户实现与实验操作手册

> 适用对象：负责运行 DEIM-D-FINE 木材缺陷实验的用户  
> 目标：按安全顺序让 Codex 完成 SQ-MAL 代码实现，并由用户完成数据准备、检查、短训、完整训练和实验验收。  
> 配套文件：`SQ_MAL_Codex_Implementation_Spec.md`

---

# 1. 这次具体要实现什么

本次只实现论文第一个模块方向：

> **SQ-MAL：Semantic Quality-aware Matchability-Aware Loss**

它使用数据集已有 semantic map，帮助 DEIM-D-FINE 判断：

1. 预测框是否与 GT 框对齐；
2. 预测框内是否真的有缺陷像素；
3. query 的分类分数是否应当被信任；
4. 哪些高分 query 实际是木纹背景假阳性。

模块包含四个逐步实验：

```text
Q1：Query Quality Head
Q2：Q1 + Binary Defectness Head
Q3：Q2 + SQ-MAL
Q4：Q3 + Hard Background Mining
```

为了保证增益能够归因，本阶段：

- 继续使用 image size=960；
- 不改为矩形输入；
- 不增加 P2；
- 不实现 MB-FDR；
- 不加长尾重采样；
- 不修改 matcher；
- 不和 TSEM 混用。

---

# 2. 开始前需要准备的路径

请先填写下面的路径表，后续会直接发给 Codex。

```text
DEIM_REPO=
CONDA_ENV=
TRAIN_IMAGE_DIR=
VAL_IMAGE_DIR=
TEST_IMAGE_DIR=

SEMANTIC_MAP_DIR=
SEMANTIC_SPEC_TXT=

TRAIN_COCO_JSON=
VAL_COCO_JSON=
TEST_COCO_JSON=

BASELINE_CONFIG=
BASELINE_CHECKPOINT=
PRETRAINED_CHECKPOINT=

OUTPUT_ROOT=
REPORT_ROOT=
```

你的实际目录可能类似：

```text
DEIM_REPO=/home/zxw4090/hjw/DEIM
CONDA_ENV=dfine
SEMANTIC_MAP_DIR=/home/.../Semantic_Maps
SEMANTIC_SPEC_TXT=/home/.../Semantic_Map_Specification.txt
BASELINE_CONFIG=configs/deim_dfine/deim_hgnetv2_l_wood_960.yml
PRETRAINED_CHECKPOINT=./weight/deim_dfine_hgnetv2_l_coco_50e.pth
OUTPUT_ROOT=output/sqmal
```

不要直接复制示例，必须换成自己的路径。

---

# 3. 备份当前工程

进入仓库：

```bash
cd "$DEIM_REPO"
```

记录当前状态：

```bash
git status
git branch --show-current
git rev-parse HEAD
```

如果有未提交修改：

```bash
git add -A
git commit -m "backup: save state before SQ-MAL"
```

也可以先创建 patch：

```bash
git diff > before_sqmal.patch
```

创建分支：

```bash
git checkout -b feature/sq-mal-v1
```

已有该分支时：

```bash
git checkout feature/sq-mal-v1
```

---

# 4. 将实现规格交给 Codex

把文件：

```text
SQ_MAL_Codex_Implementation_Spec.md
```

复制到 DEIM 仓库根目录：

```bash
cp /path/to/SQ_MAL_Codex_Implementation_Spec.md "$DEIM_REPO/"
```

确认：

```bash
ls -lh SQ_MAL_Codex_Implementation_Spec.md
```

发送给 Codex 的提示词：

```text
请读取仓库根目录的 SQ_MAL_Codex_Implementation_Spec.md，并严格按照文档分阶段实现。

我的路径如下：
DEIM_REPO=...
TRAIN_IMAGE_DIR=...
VAL_IMAGE_DIR=...
TEST_IMAGE_DIR=...
SEMANTIC_MAP_DIR=...
SEMANTIC_SPEC_TXT=...
TRAIN_COCO_JSON=...
VAL_COCO_JSON=...
TEST_COCO_JSON=...
BASELINE_CONFIG=...
PRETRAINED_CHECKPOINT=...
OUTPUT_ROOT=...
REPORT_ROOT=...

要求：
1. 先检查当前分支源码，不要假定它与官方 main 完全一致。
2. 不覆盖 baseline 配置。
3. 所有新增功能默认关闭。
4. 先只完成代码、转换工具、测试、Q0-Q4配置和报告。
5. 不要直接开始完整训练。
6. 每完成一个阶段运行单元测试和 smoke test。
7. 最后列出所有修改文件、命令、测试结果和风险。
```

---

# 5. Codex 完成后先看什么

不要立即运行完整训练。先检查 Codex 是否交付：

```text
engine/deim/sqmal.py
semantic map 转换脚本
mask 验证脚本
Q0-Q4 配置
单元测试
smoke test
docs/SQ_MAL_IMPLEMENTATION_REPORT.md
```

查看 Git 变化：

```bash
git status
git diff --stat
git diff -- engine/deim/deim_criterion.py
git diff -- engine/deim/dfine_decoder.py
git diff -- engine/deim/postprocessor.py
git diff -- engine/data/dataset/coco_dataset.py
```

重点确认：

- 原 `loss_labels_mal()` 还在；
- 原 baseline 配置没有被覆盖；
- 新开关默认是 false；
- postprocessor 在 top-k 前乘 quality；
- criterion 没有把 defect/hard-bg 重复算到所有 aux、encoder 和 DN 分支；
- `mask_valid` 与 boxes 同步筛选；
- 旧 checkpoint 使用 strict=False 或项目原有兼容加载方式。

---

# 6. 第一步：转换训练集 Semantic Maps

## 6.1 先只转换小样本

建议 Codex 的转换脚本支持：

```text
--max-images 50
```

先运行50张：

```bash
python tools/wood/augment_coco_with_semantic_masks.py \
  --images-dir "$TRAIN_IMAGE_DIR" \
  --semantic-maps-dir "$SEMANTIC_MAP_DIR" \
  --semantic-spec "$SEMANTIC_SPEC_TXT" \
  --input-coco "$TRAIN_COCO_JSON" \
  --output-coco "$OUTPUT_ROOT/data_debug/instances_train_sqmal_50.json" \
  --report-dir "$REPORT_ROOT/data_debug/train50" \
  --semantic-suffix _segm \
  --max-images 50
```

实际参数名以 Codex 最终实现为准。

## 6.2 检查报告

查看：

```bash
cat "$REPORT_ROOT/data_debug/train50/summary.json"
column -s, -t < "$REPORT_ROOT/data_debug/train50/per_class_mask_valid.csv" | less -S
```

要求：

```text
bbox_changed = 0
category_changed = 0
annotation_id_changed = 0
```

`valid_ratio` 不应异常低。若低于90%，先停止，检查：

- semantic map 类别映射是否正确；
- semantic map 与图片尺寸是否一致；
- `_segm` 命名是否正确；
- 类别名大小写映射；
- bbox 与 semantic map 是否来自同一数据版本。

## 6.3 查看可视化

至少人工检查50张，特别看：

```text
Live_knot
Dead_knot
resin
Crack
Quartzity
Knot_missing
Blue_stain
```

每张图检查：

- mask 是否落在正确 bbox 中；
- 同类多个实例是否被错误合并；
- 裂纹是否因 polygon 简化消失；
- mask 是否与图像上下/左右颠倒；
- semantic map 是否用了错误颜色索引；
- overgrown 是否被重新加入。

发现错位时，不要继续全量转换。

---

# 7. 第二步：全量生成带 mask 的 COCO JSON

建议为 train、val、test 分别生成新 JSON，不覆盖原文件。

```bash
mkdir -p "$OUTPUT_ROOT/coco_sqmal"
mkdir -p "$REPORT_ROOT/coco_sqmal"
```

训练集：

```bash
python tools/wood/augment_coco_with_semantic_masks.py \
  --images-dir "$TRAIN_IMAGE_DIR" \
  --semantic-maps-dir "$SEMANTIC_MAP_DIR" \
  --semantic-spec "$SEMANTIC_SPEC_TXT" \
  --input-coco "$TRAIN_COCO_JSON" \
  --output-coco "$OUTPUT_ROOT/coco_sqmal/instances_train_sqmal.json" \
  --report-dir "$REPORT_ROOT/coco_sqmal/train" \
  --semantic-suffix _segm
```

验证集和测试集同理：

```bash
python tools/wood/augment_coco_with_semantic_masks.py \
  --images-dir "$VAL_IMAGE_DIR" \
  --semantic-maps-dir "$SEMANTIC_MAP_DIR" \
  --semantic-spec "$SEMANTIC_SPEC_TXT" \
  --input-coco "$VAL_COCO_JSON" \
  --output-coco "$OUTPUT_ROOT/coco_sqmal/instances_val_sqmal.json" \
  --report-dir "$REPORT_ROOT/coco_sqmal/val" \
  --semantic-suffix _segm
```

```bash
python tools/wood/augment_coco_with_semantic_masks.py \
  --images-dir "$TEST_IMAGE_DIR" \
  --semantic-maps-dir "$SEMANTIC_MAP_DIR" \
  --semantic-spec "$SEMANTIC_SPEC_TXT" \
  --input-coco "$TEST_COCO_JSON" \
  --output-coco "$OUTPUT_ROOT/coco_sqmal/instances_test_sqmal.json" \
  --report-dir "$REPORT_ROOT/coco_sqmal/test" \
  --semantic-suffix _segm
```

注意：

- 若 train/val/test 图片实际都在同一个目录，三个 `--images-dir` 可以相同；
- 不要覆盖原 COCO JSON；
- 评估仍是 bbox AP，semantic mask 只作为训练监督；
- test 加 mask 是为了诊断和可视化，不是必须参与训练。

---

# 8. 第三步：运行数据集验证

运行 Codex 提供的验证工具：

```bash
python tools/wood/validate_sqmal_dataset.py \
  --images-dir "$TRAIN_IMAGE_DIR" \
  --coco-json "$OUTPUT_ROOT/coco_sqmal/instances_train_sqmal.json" \
  --num-samples 200 \
  --output-dir "$REPORT_ROOT/validate_train"
```

必须检查：

```text
boxes数量 == labels数量 == masks数量 == mask_valid数量
RLE能成功decode
增强后mask与bbox对齐
无NaN/Inf
无负宽高框
无越界索引
```

再运行可视化：

```bash
python tools/wood/visualize_sqmal_masks.py \
  --images-dir "$TRAIN_IMAGE_DIR" \
  --coco-json "$OUTPUT_ROOT/coco_sqmal/instances_train_sqmal.json" \
  --output-dir "$REPORT_ROOT/visual_train" \
  --num-images 200 \
  --seed 42
```

人工抽查至少100张。

---

# 9. 第四步：运行单元测试

进入环境：

```bash
conda activate "$CONDA_ENV"
cd "$DEIM_REPO"
```

Codex 应提供统一命令，例如：

```bash
pytest -q tests/test_sqmal_semantic_support.py
pytest -q tests/test_sqmal_losses.py
pytest -q tests/test_sqmal_forward.py
```

或：

```bash
python tools/tests_sqmal/run_all.py
```

不能忽略失败测试。

重点查看：

- 细线 mask 的 semantic support 不会接近0；
- invalid mask 时退化成原 MAL；
- quality_power=0 时排序不变；
- 所有新开关关闭时 baseline 输出不变；
- AMP 前向/反向正常。

---

# 10. 第五步：运行模型 Smoke Test

## 10.1 只前向

使用2张图：

```bash
python tools/wood/smoke_test_sqmal.py \
  -c configs/deim_dfine/ablation_sqmal/q4_sqmal_hbg.yml \
  --coco-json "$OUTPUT_ROOT/coco_sqmal/instances_train_sqmal.json" \
  --images-dir "$TRAIN_IMAGE_DIR" \
  --batch-size 2 \
  --device cuda
```

应看到：

```text
pred_logits: [B,Q,9]
pred_boxes: [B,Q,4]
pred_quality: [B,Q,1]
pred_defect_logits: [B,1,Hd,Wd]
```

## 10.2 反向

应打印：

```text
loss_sqmal
loss_quality
loss_defect
loss_hard_bg
total_loss
```

并确认新 head 有非零梯度。

## 10.3 旧 checkpoint 加载

用当前预训练权重运行：

```bash
python tools/wood/smoke_test_sqmal.py \
  -c configs/deim_dfine/ablation_sqmal/q4_sqmal_hbg.yml \
  -t "$PRETRAINED_CHECKPOINT" \
  --coco-json "$OUTPUT_ROOT/coco_sqmal/instances_train_sqmal.json" \
  --images-dir "$TRAIN_IMAGE_DIR" \
  --batch-size 2 \
  --device cuda
```

允许 missing keys 仅包含：

```text
quality head
defectness head
```

若出现大量 backbone/encoder/decoder missing keys，停止并修复。

---

# 11. 第六步：先做100 iteration短训

不要先跑完整 epoch。

根据 Codex 提供的 debug 配置，运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
  -c configs/deim_dfine/ablation_sqmal/q1_quality_debug.yml \
  --use-amp \
  --seed=0 \
  -t "$PRETRAINED_CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT/debug_q1"
```

Q2、Q3、Q4分别短训。

观察：

```text
loss 是否有限
显存是否稳定
速度是否异常下降
quality 正样本均值是否高于负样本
semantic support 是否在合理范围
hard-bg 每图是否接近TopK而非始终为0
valid_mask_ratio 是否稳定
```

异常信号：

| 现象 | 可能原因 |
|---|---|
| quality全部接近0 | 负样本权重过大或正负未分别归一 |
| semantic support大量为0 | mask映射、bbox坐标或积分图错误 |
| defect loss很低但AP下降 | mask与图像增强错位 |
| hard-bg始终为0 | IoU条件、quality条件或匹配mask错误 |
| hard-bg始终非常大 | top-k未按每张图限制 |
| AP50骤降 | semantic beta或rerank power过大 |
| CUDA OOM | aux层重复输出quality或保留过多mask图 |

---

# 12. 第七步：先复现Q0基线

Q0 必须与当前 baseline 功能一致，只使用新数据 JSON 但所有 SQ-MAL 功能关闭。

训练命令示例：

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun \
  --master_port=7811 \
  --nproc_per_node=2 \
  train.py \
  -c configs/deim_dfine/ablation_sqmal/q0_baseline.yml \
  --use-amp \
  --seed=0 \
  -t "$PRETRAINED_CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT/q0_baseline" \
  | tee "$OUTPUT_ROOT/q0_baseline/train_console.log"
```

为什么必须跑Q0：

- 验证换成带 segmentation 的 JSON 没有改变 bbox；
- 验证 loader/transform 修改没有破坏 baseline；
- 验证新代码关闭时结果仍可复现；
- 后续Q1～Q4必须与Q0对比，而不是与另一套旧评估脚本对比。

Q0至少应接近当前正确基线：

```text
AP约50.72
AP50约80.04
AP75约54.41
```

允许随机种子与训练波动，但若差异大于1 AP，先排查，不继续模块实验。

---

# 13. 第八步：按顺序训练Q1～Q4

保持以下条件完全一致：

```text
相同数据划分
相同960输入
相同预训练权重
相同epoch
相同batch size
相同学习率
相同增强
相同seed
相同评估脚本
```

## 13.1 Q1：Quality Head

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7812 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqmal/q1_quality.yml \
--use-amp \
--seed=0 \
-t "$PRETRAINED_CHECKPOINT" \
--output-dir "$OUTPUT_ROOT/q1_quality" \
| tee "$OUTPUT_ROOT/q1_quality/train_console.log"
```

功能：

```text
原 MAL
+ query quality head
+ quality loss
+ 推理重排序
```

主要验证分类—定位质量错位是否存在。

## 13.2 Q2：Defectness Auxiliary

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7813 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqmal/q2_defect_aux.yml \
--use-amp \
--seed=0 \
-t "$PRETRAINED_CHECKPOINT" \
--output-dir "$OUTPUT_ROOT/q2_defect_aux" \
| tee "$OUTPUT_ROOT/q2_defect_aux/train_console.log"
```

功能：

```text
Q1
+ 二值缺陷度辅助分支
```

仍使用原 MAL，判断 semantic auxiliary supervision 自身是否有效。

## 13.3 Q3：完整SQ-MAL

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7814 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqmal/q3_sqmal.yml \
--use-amp \
--seed=0 \
-t "$PRETRAINED_CHECKPOINT" \
--output-dir "$OUTPUT_ROOT/q3_sqmal" \
| tee "$OUTPUT_ROOT/q3_sqmal/train_console.log"
```

功能：

```text
Q2
+ IoU × semantic support 的SQ-MAL target
```

## 13.4 Q4：Hard Background

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7815 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqmal/q4_sqmal_hbg.yml \
--use-amp \
--seed=0 \
-t "$PRETRAINED_CHECKPOINT" \
--output-dir "$OUTPUT_ROOT/q4_sqmal_hbg" \
| tee "$OUTPUT_ROOT/q4_sqmal_hbg/train_console.log"
```

功能：

```text
Q3
+ hard background query mining
```

---

# 14. 每次训练后必须统一评估

每个实验使用同一测试 JSON、同一 checkpoint 选择原则和同一 score floor。

至少输出：

```text
COCO AP/AP50/AP75
逐类AP
TIDE
混淆矩阵
FP/FN矩阵
每类PR曲线
ECE或LaECE
IoU分段召回
每类背景FP数量
```

建议输出目录：

```text
$OUTPUT_ROOT/qX_xxx/eval/
  class_and_all_metrics.json
  per_class_ap.json
  tide_summary.txt
  confusion_matrix.png
  confusion_matrix_with_fp_fn.png
  ece_summary.json
  reliability_bins.csv
  recall_by_iou_threshold.json
  quality_diagnostics.json
```

新增质量诊断：

```text
quality与真实IoU的Spearman
quality与联合target的Spearman
正query质量分布
负query质量分布
背景FP质量分布
TP质量分布
```

---

# 15. 如何判断每一步是否有效

## Q1有效

应看到：

- quality 与真实 IoU 正相关；
- 质量重排序后 FP dAP 下降；
- AP50和召回基本不下降；
- 若 AP 提升小但 FP明显下降，仍可进入Q2。

## Q2有效

应看到：

- defectness map 可视化覆盖真实缺陷；
- Background dAP 下降；
- Crack、Quartzity 的背景纹理响应减少；
- defect loss 不是快速塌缩到0。

## Q3有效

应看到：

- Q3优于Q2，证明 semantic support 进入 MAL 后有额外价值；
- AP或TIDE FP指标进一步改善；
- matched semantic support 的均值不能异常低；
- Live_knot、Dead_knot、resin 的AP50不应明显下降。

## Q4有效

应看到：

- Background dAP相对Q3继续下降；
- False Positive dAP下降；
- False Negative dAP和Recall@0.5基本不变；
- Quartzity、Crack、Blue_stain FP/GT下降。

---

# 16. 推荐的初始超参数

```yaml
semantic_beta: 0.5
semantic_warmup_epochs: 5

loss_quality: 0.5
quality_pos_weight: 1.0
quality_neg_weight: 0.25
quality_power: 0.5

loss_defect: 0.5
defect_bce_weight: 1.0
defect_dice_weight: 1.0

loss_hard_bg: 0.25
hard_bg_topk: 20
hard_bg_iou_threshold: 0.1
hard_bg_start_epoch: 5
```

第一轮不要大规模网格搜索。

调参顺序：

```text
1. quality_power
2. semantic_beta
3. loss_quality
4. loss_defect
5. loss_hard_bg
6. hard_bg_topk
```

---

# 17. 常见失败与处理

## 17.1 AP50明显下降

按顺序调整：

```text
quality_power: 0.5 → 0.25
semantic_beta: 0.5 → 0.25
semantic_warmup_epochs: 5 → 10
loss_hard_bg: 0.25 → 0.1
hard_bg_start_epoch: 5 → 10
```

## 17.2 背景FP没下降

检查：

- postprocessor 是否在 top-k 前重排序；
- quality head 是否只学到 IoU，没有学到 semantic support；
- Q3是否真的使用 `sqmal` 而不是仍使用 `mal`；
- hard-bg 是否选中了远离GT的query；
- quality target 是否被错误设为全1；
- semantic mask 是否大面积覆盖正常木纹背景。

## 17.3 Crack性能下降

常见原因：

- 用普通“缺陷像素/预测框面积”作为 support；
- 细裂纹在 mask 降采样中消失；
- semantic map 被错误简化成 polygon；
- 使用 bilinear 将细线降采样掉。

正确做法：

```text
coverage + relative density
RLE保存mask
adaptive max pooling生成defect target
```

## 17.4 显存增长过大

检查：

- 是否给每个decoder aux层都输出quality；
- 是否把完整原分辨率mask放进model outputs；
- 是否在每层重复构造积分图；
- 是否对DN/encoder分支重复计算semantic loss。

第一版应只在main output计算质量、defect和hard-bg。

## 17.5 旧模型无法推理

检查：

- `quality_rerank=False` 时是否仍强制读取 `pred_quality`；
- quality head 是否默认关闭；
- constructor是否新增了无默认值参数；
- checkpoint是否使用 strict=True；
- export脚本是否假定输出dict只有pred_logits和pred_boxes。

---

# 18. 多随机种子复验

Q0～Q4先用seed=0筛选。

最终只对最优方案和Q0运行：

```text
seed=0
seed=1
seed=2
```

报告：

```text
mean ± std
```

Blue_stain 测试实例很少，不能只依据一次 AP 判断模块效果。还应查看：

- 背景FP数量；
- PR曲线；
- bootstrap置信区间；
- 多seed方向是否一致。

---

# 19. 实验表格模板

| 实验 | Quality | Defect Aux | SQ-MAL | Hard BG | AP | AP50 | AP75 | Bkg dAP | FP dAP | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q0 |  |  |  |  |  |  |  |  |  |  |
| Q1 | ✓ |  |  |  |  |  |  |  |  |  |
| Q2 | ✓ | ✓ |  |  |  |  |  |  |  |  |
| Q3 | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |
| Q4 | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |

逐类重点表：

| 实验 | Live AP | Dead AP | resin AP | Crack AP | Quartzity AP | Blue FP | Quartzity FP | Crack FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q0 |  |  |  |  |  |  |  |  |
| Q1 |  |  |  |  |  |  |  |  |
| Q2 |  |  |  |  |  |  |  |  |
| Q3 |  |  |  |  |  |  |  |  |
| Q4 |  |  |  |  |  |  |  |  |

---

# 20. 完成后的验收清单

在进入第二个创新模块前，逐项确认：

```text
[ ] Q0复现成功
[ ] mask有效率和可视化通过
[ ] 所有单元测试通过
[ ] AMP与双GPU训练正常
[ ] 旧checkpoint可加载
[ ] Q1-Q4均有独立配置
[ ] quality正负分布可区分
[ ] Background dAP下降
[ ] False Positive dAP下降
[ ] Recall@0.5没有明显下降
[ ] AP50没有明显下降
[ ] 至少一个重点类别明确提升
[ ] 三个seed方向一致
[ ] Git提交和实现报告完整
```

若 Q4 不如 Q3，则论文中可以使用 Q3，不应为了模块数量强行保留 hard background。

若 Q2 不如 Q1，但 Q3有效，应重新检查 defect auxiliary 权重，或仅将 semantic mask用于SQ target而删除辅助分割分支。

---

# 21. 回滚方法

查看提交：

```bash
git log --oneline --decorate -20
```

回退某个实验提交：

```bash
git revert <commit_hash>
```

不要直接使用：

```bash
git reset --hard
```

除非已经确认所有用户改动都已备份。

切回原分支：

```bash
git checkout <原分支名>
```

应用备份 patch：

```bash
git apply before_sqmal.patch
```

---

# 22. 本阶段最终交付物

用户最终应保存：

```text
1. 带RLE mask的train/val/test COCO JSON
2. mask转换与可视化报告
3. SQ-MAL完整代码
4. Q0-Q4配置
5. 单元测试与smoke test结果
6. Q0-Q4训练日志
7. 统一评估结果
8. 三随机种子结果
9. 实现报告
10. Git commit记录
```

这些材料可直接用于论文中的：

- 方法实现细节；
- 消融实验；
- 错误分析；
- 可视化；
- 复现说明。
