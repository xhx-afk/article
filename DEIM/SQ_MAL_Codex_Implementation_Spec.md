# SQ-MAL Codex 工程实现规格书

> 目标仓库：DEIM / DEIM-D-FINE  
> 目标模块：SQ-MAL（Semantic Quality-aware Matchability-Aware Loss）  
> 文档用途：将本文件放到目标仓库根目录后，直接交给 Codex 按阶段完成实现、测试和交付。  
> 重要原则：**先检查当前分支源码，再按符号定位修改点，不得仅依赖本文中的行号。**

---

## 0. Codex 的最终任务

在用户当前的 DEIM-D-FINE-L 木材缺陷检测工程中，实现一个默认关闭、向后兼容的 SQ-MAL 模块，使用数据集已有的 semantic map 监督以下三个部分：

1. **二值缺陷度辅助分支**：让最高分辨率检测特征学习“缺陷/背景”像素区分；
2. **Query 质量分支**：预测每个 decoder query 的联合定位—语义质量；
3. **SQ-MAL 与困难背景挖掘**：使用 bbox IoU 和 GT mask 语义支持度联合构造软分类目标，并重点抑制远离 GT 的高分背景 query。

推理时，在 top-k 之前使用 query quality 对分类分数重排序：

```text
final_score = sigmoid(class_logit) * sigmoid(quality_logit) ** quality_power
```

第一版不修改 Hungarian matcher，不实现 MB-FDR，不修改输入尺寸，不进行类别重采样。

---

# 1. 背景与问题证据

用户当前正确基线为 DEIM-D-FINE-L，训练/测试图像尺寸设置为 960。诊断结果表明：

- bbox AP@[0.50:0.95] 约 50.72；
- AP50 约 80.04，AP75 约 54.41；
- IoU=0.5 时 GT 覆盖召回约 97.48%；
- TIDE Background dAP 为 5.18；
- TIDE Classification dAP 为 4.66；
- False Positive dAP 为 18.53；
- False Negative dAP 仅为 0.88；
- 大量错误不是“没有候选框”，而是背景假阳性、类别排序和质量排序不可靠。

因此 SQ-MAL 的研发目标不是增加 query 或扩大正样本数量，而是：

```text
分类置信度
    +
bbox 定位质量
    +
缺陷像素支持度
    ↓
更可靠的 query 匹配质量与最终排序
```

---

# 2. 必须遵守的工程约束

## 2.1 向后兼容

所有新增功能必须由配置开关控制，且默认关闭。关闭全部开关后：

- 模型输出、loss、postprocessor 行为与原基线一致；
- 原配置文件不需要增加任何字段；
- 原预训练权重可以加载；
- 原推理、导出和评估流程可以运行；
- 不允许直接覆盖用户已经训练稳定的 baseline 配置。

推荐开关：

```yaml
use_defectness_head: false
use_quality_head: false
use_sqmal: false
use_hard_bg: false
quality_rerank: false
```

并对字段给出详细中文注释

## 2.2 第一版禁止项

第一版禁止同时加入以下内容：

- 不修改 Hungarian matcher 代价；
- 不实现 semantic matcher；
- 不实现 MB-FDR；
- 不修改 backbone；
- 不增加 P2；
- 不修改训练图像尺寸；
- 不增加 class-balanced loss；
- 不增加尾类 oversampling；
- 不加入 copy-paste；
- 不删除原 MAL；
- 不删除 D-FINE 原有 LQE；
- 不将九类 semantic map 作为九分类分割任务。

## 2.3 可复现实验

至少生成以下五个独立配置：

```text
Q0_baseline
Q1_quality
Q2_defect_aux
Q3_sqmal
Q4_sqmal_hbg
```

每个配置只比前一个增加明确功能，便于消融。

---

# 3. 开始编码前的仓库预检

Codex 必须先执行并记录：

```bash
pwd
git status
git branch --show-current
git rev-parse HEAD
python -V
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
find engine/deim -maxdepth 1 -type f | sort
find engine/data -maxdepth 3 -type f | sort
find configs -maxdepth 3 -type f | grep -E "deim|dfine|wood" | sort
```

然后按符号搜索，不得假定用户分支与官方 main 完全相同：

```bash
grep -R "class DEIM" -n engine
grep -R "class DFINETransformer" -n engine
grep -R "class DEIMCriterion" -n engine
grep -R "def loss_labels_mal" -n engine
grep -R "class PostProcessor" -n engine
grep -R "return_masks" -n engine/data
grep -R "aux_outputs" -n engine/deim
grep -R "set_epoch" -n engine | head -100
```

创建研发分支：

```bash
git checkout -b feature/sq-mal-v1
```

如分支已存在，则切换到该分支，不得强制覆盖用户改动。

---

# 4. 官方代码中的关键接入点

当前官方 DEIM 结构中通常存在以下接入点，用户分支应通过符号确认：

```text
engine/deim/deim.py
engine/deim/dfine_decoder.py
engine/deim/deim_criterion.py
engine/deim/postprocessor.py
engine/data/dataset/coco_dataset.py
```

官方实现中的典型行为：

1. `DEIM.forward()` 按 backbone → encoder → decoder 顺序执行；
2. `DFINETransformer` 内含 `dec_score_head`、`dec_bbox_head` 和 D-FINE LQE；
3. `DEIMCriterion.loss_labels_mal()` 使用 matched bbox IoU 构造 MAL 软目标；
4. criterion 会重复计算 main、aux、pre、encoder 和 DN 分支损失；
5. `PostProcessor` 在 sigmoid 后先 flatten，再执行 top-k；
6. COCO dataset 已有 `return_masks` 能力，但只读取 `segmentation`，默认不会保留自定义 `mask_valid` 字段。

Codex 必须以当前分支实际代码为准适配。

---

# 5. 交付文件清单

最终至少新增或修改以下内容。

## 5.1 建议新增

```text
engine/deim/sqmal.py
tools/wood/augment_coco_with_semantic_masks.py
tools/wood/validate_sqmal_dataset.py
tools/wood/visualize_sqmal_masks.py
tests/test_sqmal_semantic_support.py
tests/test_sqmal_losses.py
tests/test_sqmal_forward.py
configs/deim_dfine/ablation_sqmal/
  q0_baseline.yml
  q1_quality.yml
  q2_defect_aux.yml
  q3_sqmal.yml
  q4_sqmal_hbg.yml
docs/SQ_MAL_IMPLEMENTATION_REPORT.md
```

如果仓库没有 `tests/`，允许创建 `tools/tests_sqmal/`，但应提供一条统一测试命令。

## 5.2 预计修改

```text
engine/data/dataset/coco_dataset.py
engine/deim/__init__.py
engine/deim/deim.py
engine/deim/dfine_decoder.py
engine/deim/deim_criterion.py
engine/deim/postprocessor.py
```

如当前分支文件名不同，通过类名和函数名定位。

---

# 6. 第一阶段：将 Semantic Maps 加入 COCO annotation

## 6.1 用户提供的数据输入

转换工具必须使用命令行参数，不得把路径写死：

```bash
python tools/wood/augment_coco_with_semantic_masks.py \
  --images-dir /path/to/images \
  --semantic-maps-dir /path/to/Semantic_Maps \
  --semantic-spec /path/to/Semantic_Map_Specification.txt \
  --input-coco /path/to/instances_train.json \
  --output-coco /path/to/instances_train_sqmal.json \
  --report-dir /path/to/reports/train \
  --semantic-suffix _segm
```

支持的图像名关系：

```text
原图：99100003.jpg / bmp / png
semantic map：99100003_segm.png / bmp / tif
```

扩展名不应写死，应根据 stem 搜索。

## 6.2 类别约束

有效九类及顺序必须从 input COCO `categories` 读取，禁止自行重新编号。

预期类别：

```text
Live_knot
Dead_knot
resin
knot_with_crack
Crack
Marrow
Quartzity
Knot_missing
Blue_stain
```

`overgrown` 必须保持删除状态。转换脚本不得重新加入它。

类别名应通过 canonical map 兼容大小写和历史拼写，但最终输出必须保持 input COCO 的 `category_id` 与 `name` 不变。

## 6.3 解析 Semantic_Map_Specification

实现独立函数：

```python
parse_semantic_spec(path) -> SemanticSpec
```

至少支持：

- 灰度索引图；
- RGB 颜色图；
- 文本中以空格、冒号、逗号或制表符分隔；
- 类别名大小写差异；
- 背景值；
- overgrown 值但不输出 overgrown annotation。

如果无法自动解析，程序必须：

1. 输出已读取的所有行；
2. 给出明确报错；
3. 提示用户通过 `--class-map-json` 提供显式映射；
4. 不得静默猜测颜色或索引。

## 6.4 图像尺寸校验

每张图必须检查：

```text
image_width == semantic_width
image_height == semantic_height
```

不一致时：

- 默认将该图所有 annotation 标记 `mask_valid=0`；
- 不自动缩放 semantic map；
- 将文件写入 `size_mismatch.txt`；
- 继续处理其他图像；
- 提供 `--resize-semantic-nearest` 可选参数，但默认关闭。

## 6.5 实例 mask 提取策略

semantic map 很可能是类别级语义图，而 COCO annotation 是实例级 bbox。第一版按每个 annotation 提取局部实例 mask。

对每个 annotation：

1. 获得该 category 的二值语义图；
2. 将 bbox 按宽高分别向外扩展 `expand_ratio=0.03`；
3. 截取扩展 bbox 中的该类别像素；
4. 对局部区域做 connected components；
5. 过滤面积 `< min_component_area` 的连通域，默认 2 像素；
6. 按以下优先级选择一个或多个合理连通域：
   - 与原 bbox 的像素重叠率；
   - 连通域 bbox 与 annotation bbox 的 IoU；
   - 连通域中心到 annotation 中心的归一化距离；
7. 若多个连通域都与 bbox 明显重叠，允许合并；
8. 将结果限制在扩展 bbox 内；
9. 若结果为空，设置 `mask_valid=0`；
10. 若结果有效，设置 `mask_valid=1`。

建议连通域评分：

```text
score =
0.50 * intersection_over_component
+ 0.35 * bbox_iou
+ 0.15 * center_score
```

但必须将权重做成 CLI 参数或常量集中定义，不能散落在代码中。

## 6.6 使用 RLE，不用简化 polygon

为保护 Crack、resin 等细长区域，输出 COCO RLE：

```json
"segmentation": {
  "size": [height, width],
  "counts": "..."
}
```

使用 Fortran-order mask 编码。JSON 写入前将 `counts` 的 bytes 转为 ASCII string。

修改 `convert_coco_poly_to_mask()`，兼容：

- polygon list；
- uncompressed RLE；
- compressed RLE。

不得破坏原 polygon 数据集。

## 6.7 mask_valid 字段

每个 annotation 增加：

```json
"mask_valid": 1
```

无有效 mask 时：

```json
"mask_valid": 0,
"segmentation": []
```

COCO loader 必须读取该字段，形成：

```python
target["mask_valid"]  # BoolTensor[N]
```

并与 boxes、labels、masks 使用相同的 `keep` 过滤。

## 6.8 输出报告

转换完成后必须生成：

```text
summary.json
per_class_mask_valid.csv
size_mismatch.txt
missing_semantic_map.txt
empty_class_pixels.txt
invalid_masks.jsonl
sample_visualizations/
```

`summary.json` 至少包含：

```json
{
  "images_total": 0,
  "images_processed": 0,
  "annotations_total": 0,
  "valid_masks": 0,
  "invalid_masks": 0,
  "valid_ratio": 0.0,
  "bbox_changed": 0,
  "category_changed": 0,
  "annotation_id_changed": 0
}
```

要求：

```text
bbox_changed == 0
category_changed == 0
annotation_id_changed == 0
```

---

# 7. 第二阶段：数据加载和 transform 完整性

## 7.1 COCO loader

在 `ConvertCocoPolysToMask` 中：

```python
mask_valid = torch.tensor(
    [bool(obj.get("mask_valid", bool(obj.get("segmentation")))) for obj in anno],
    dtype=torch.bool,
)
```

对 `keep` 同步过滤：

```python
mask_valid = mask_valid[keep]
target["mask_valid"] = mask_valid
```

## 7.2 transform 字段同步

搜索所有可能按 box 有效性过滤实例的 transform，例如：

```bash
grep -R "target.*boxes" -n engine/data/transforms
grep -R "keep =" -n engine/data/transforms
grep -R "\"masks\"" -n engine/data/transforms
```

任何对 boxes、labels、masks、area、iscrowd 的筛选，都必须同步筛选 `mask_valid`。

检查至少以下增强：

- Resize；
- RandomHorizontalFlip；
- RandomZoomOut；
- RandomCrop；
- Mosaic；
- MixUp；
- Pad；
- SanitizeBoundingBoxes。

如果 Mosaic 或 MixUp 不支持 RLE，因为进入 transform 前已经 decode 为 tensor mask，确保 tensor mask 与 boxes 一起拼接。

## 7.3 空实例处理

当一张图增强后没有有效 box：

```text
boxes.shape == [0,4]
labels.shape == [0]
masks.shape == [0,H,W]
mask_valid.shape == [0]
```

不得报错。

---

# 8. 第三阶段：新增 `engine/deim/sqmal.py`

建议集中放置以下组件和纯函数：

```python
class DefectnessHead(nn.Module)
class QueryQualityHead(nn.Module)  # 如最终放在 decoder 文件，可不重复
build_union_defect_target(...)
compute_box_mask_sums(...)
compute_semantic_support(...)
compute_sq_quality_target(...)
select_hard_background_queries(...)
semantic_beta_schedule(...)
```

所有函数必须：

- 有类型注解；
- 有 shape 注释；
- 对空 tensor 可运行；
- 对 NaN/Inf 做防护；
- 支持 AMP；
- 不调用 `.cpu().numpy()`；
- 不在训练热路径使用逐框 `.item()`；
- 不依赖 OpenCV，训练路径只使用 PyTorch/torchvision。

---

# 9. Semantic Support 的精确定义

## 9.1 输入

对 matched query \(i\)：

```text
pred_box_i：归一化 cxcywh
gt_box_i：归一化 cxcywh
gt_mask_i：[H,W] bool/uint8
mask_valid_i：bool
```

## 9.2 框转换

将归一化 cxcywh 转为像素 xyxy：

```text
x1 = floor(clamp(x1_norm * W, 0, W))
y1 = floor(clamp(y1_norm * H, 0, H))
x2 = ceil(clamp(x2_norm * W, 0, W))
y2 = ceil(clamp(y2_norm * H, 0, H))
```

保证：

```text
x2 >= x1 + 1
y2 >= y1 + 1
```

## 9.3 使用积分图精确求矩形 mask 像素和

不得逐 query 对大 mask 做 Python slicing。

对每张图中 matched masks：

```python
integral = F.pad(
    masks.float().cumsum(-2).cumsum(-1),
    (1, 0, 1, 0),
)
```

使用高级索引计算：

```text
sum = I[y2,x2] - I[y1,x2] - I[y2,x1] + I[y1,x1]
```

每张图处理后再 concat，允许不同图像 H/W 不同。

## 9.4 语义覆盖率

\[
r_i =
\operatorname{clamp}
\left(
\frac{|M_i \cap B_i^{pred}|}{|M_i|+\epsilon},
0,1
\right)
\]

## 9.5 相对语义密度

\[
d_i^{pred} =
\frac{|M_i \cap B_i^{pred}|}{|B_i^{pred}|+\epsilon}
\]

\[
d_i^{gt} =
\frac{|M_i \cap B_i^{gt}|}{|B_i^{gt}|+\epsilon}
\]

\[
\hat d_i =
\operatorname{clamp}
\left(
\frac{d_i^{pred}}{d_i^{gt}+\epsilon},
0,1
\right)
\]

## 9.6 Semantic support

\[
s_i = \sqrt{r_i \cdot \hat d_i}
\]

Fallback：

```python
s_i = 1.0
```

当任意条件成立：

- `mask_valid=False`；
- GT mask area 小于 `min_mask_area`；
- GT box area 小于 1；
- mask shape 无效；
- 产生 NaN/Inf。

Fallback 为 1 的原因：使 SQ-MAL 自动退化为原 MAL，而不是删除该正样本。

## 9.7 Detach

`semantic_support` 只作为监督目标：

```python
semantic_support = semantic_support.detach()
iou = iou.detach()
```

不得通过框离散索引反传梯度。

---

# 10. 第四阶段：Quality Head

## 10.1 保留原 LQE

D-FINE decoder 已有 LQE，根据 `pred_corners` 的分布统计修正分类 logits。不得删除或替换它。

新 quality head 表达：

```text
bbox IoU × semantic support
```

它与 LQE 功能不同。

## 10.2 Decoder 修改

在 `DFINETransformer` 中新增配置字段：

```python
use_quality_head: bool = False
quality_hidden_dim: int = 128
quality_aux_last_n: int = 0
```

新增每层 head：

```python
nn.Sequential(
    nn.Linear(layer_dim, quality_hidden_dim),
    nn.ReLU(inplace=True),
    nn.Linear(quality_hidden_dim, 1),
)
```

注意 D-FINE 在 `eval_idx` 前后可能使用 `hidden_dim` 与 `scaled_dim`，quality head 输入维度必须与对应 decoder layer output 一致。

## 10.3 Decoder 返回值

在每个有效 decoder 层计算：

```python
quality_logit = quality_head[i](output)
```

新增返回 stack：

```text
out_quality: [num_layers, B, Q_total, 1]
```

DN 开启时，按与 logits、boxes 相同的 `dn_num_split` 在 query 维拆分。

主输出：

```python
out["pred_quality"] = out_quality[-1]
```

aux 输出仅在 `quality_aux_last_n > 0` 时加入：

```python
aux_dict["pred_quality"] = ...
```

第一版默认：

```yaml
quality_aux_last_n: 0
```

即只监督最后一层。

## 10.4 部署转换

`convert_to_deploy()` 必须像 score/bbox head 一样只保留 `eval_idx` 对应的 quality head。

旧 checkpoint 加载时，只允许出现新模块的 missing keys，不允许出现大量 unexpected keys。

---

# 11. 第五阶段：Defectness Head

## 11.1 接入位置

优先在 `DEIM.forward()` 中，对 encoder 输出的最高分辨率特征计算：

```python
features = self.backbone(x)
features = self.encoder(features)
defect_logits = self.defectness_head(features[0])
outputs = self.decoder(features, targets)
```

如果当前分支的 encoder 输出结构不同，Codex 必须检查实际 shape 后适配。

新增可选注入：

```python
defectness_head: Optional[nn.Module] = None
```

若当前注册系统不支持可选 inject，允许创建新的注册类：

```python
DEIMWithSQMAL
```

但优先保持原 `DEIM` 向后兼容。

## 11.2 Head 结构

```python
Conv1x1(in_channels -> hidden)
DepthwiseConv3x3(hidden)
BatchNorm2d或GroupNorm
SiLU
Conv1x1(hidden -> 1)
```

默认：

```yaml
in_channels: 256
hidden_channels: 128
```

不得使用大型 decoder。

## 11.3 输出

训练时：

```python
outputs["pred_defect_logits"] = defect_logits
```

推理时默认不返回 defect map，除非：

```yaml
return_defect_map_in_eval: true
```

## 11.4 GT union mask

将所有 `mask_valid=True` 的实例 mask 取并集。

降采样到 defect logits 尺度时使用：

```python
F.adaptive_max_pool2d
```

以保留细裂纹。

损失：

```text
BCEWithLogits + Dice
```

空有效 mask 图像的 union target 为全 0，仍参与背景监督。

---

# 12. 第六阶段：新增 SQ-MAL，不覆盖原 MAL

## 12.1 Criterion 参数

新增但默认无效：

```python
use_sqmal=False
semantic_beta=0.5
semantic_warmup_epochs=0
semantic_min_mask_area=2.0
quality_pos_weight=1.0
quality_neg_weight=0.25
defect_bce_weight=1.0
defect_dice_weight=1.0
hard_bg_topk=20
hard_bg_iou_threshold=0.1
hard_bg_start_epoch=0
```

保留：

```python
loss_labels_mal()
```

新增：

```python
loss_labels_sqmal()
loss_quality()
loss_defect()
loss_hard_bg()
```

`get_loss()` 增加映射，但 baseline 配置仍使用 `mal`。

## 12.2 SQ-MAL target

原 MAL：

\[
t_i = IoU_i^\gamma
\]

SQ-MAL：

\[
t_i =
IoU_i^\gamma \cdot s_i^{\beta_t}
\]

其中：

\[
\beta*t =
\beta*{max}
\cdot
\min\left(1,\frac{epoch}{warmup_epochs}\right)
\]

若 warmup_epochs=0：

```text
beta_t = semantic_beta
```

建议初值：

```yaml
semantic_beta: 0.5
semantic_warmup_epochs: 5
```

## 12.3 epoch 注入

在 criterion 增加：

```python
def set_epoch(self, epoch: int, total_epochs: int | None = None):
    self.current_epoch = int(epoch)
```

在训练 epoch 开始处：

```python
if hasattr(criterion, "set_epoch"):
    criterion.set_epoch(epoch, total_epochs)
```

如果 criterion 被 DDP/wrapper 包裹，正确取得内部对象。

不得通过全局变量读取 epoch。

## 12.4 loss key

返回：

```python
{"loss_sqmal": loss}
```

配置 weight_dict 单独使用：

```yaml
loss_sqmal: 1.0
```

不要复用 `loss_mal` key，以便日志和消融清晰。

## 12.5 分支计算规则

第一版：

| 分支            |           sqmal |  quality | defect | hard_bg |
| --------------- | --------------: | -------: | -----: | ------: |
| main            |               ✓ |        ✓ |      ✓ |       ✓ |
| decoder aux     |      可选 sqmal | 默认关闭 |      × |       × |
| pre_outputs     | 原 mal 或 focal |        × |      × |       × |
| enc_aux_outputs | 原 mal 或 focal |        × |      × |       × |
| dn_outputs      |          原 mal |        × |      × |       × |
| dn_pre_outputs  |          原 mal |        × |      × |       × |

必须在 criterion forward 中显式控制，不能因为 `self.losses` 遍历而把 quality/defect/hard_bg 重复施加到所有分支。

推荐实现：

```python
def _loss_allowed(self, loss_name, branch_name, aux_index=None) -> bool:
    ...
```

---

# 13. Quality Loss

## 13.1 Target

matched query：

\[
q_i^\* = IoU_i \cdot s_i^{\beta_t}
\]

unmatched query：

\[
q_i^\* = 0
\]

## 13.2 Loss

使用 BCEWithLogits，并分别平均正负部分：

```python
pos_loss = bce[pos_mask].mean() if pos_mask.any() else zero
neg_loss = bce[neg_mask].mean() if neg_mask.any() else zero
loss_quality = quality_pos_weight * pos_loss + quality_neg_weight * neg_loss
```

默认：

```yaml
quality_pos_weight: 1.0
quality_neg_weight: 0.25
loss_quality: 0.5
```

不得直接对全部 query 求平均，否则负样本数量会使 head 倾向全 0。

---

# 14. Hard Background Mining

## 14.1 候选

仅使用 unmatched query。

计算每个 query 与所有 GT 的最大 IoU：

```text
max_iou_to_any_gt
```

候选条件：

```text
max_iou_to_any_gt < hard_bg_iou_threshold
```

默认 0.1，以排除目标附近重复框。

## 14.2 困难度

\[
h*i =
\max_c sigmoid(logit*{i,c})
\cdot
(1-sigmoid(quality_i))
\]

对每张图选 top-k：

```yaml
hard_bg_topk: 20
```

如果候选不足，使用实际数量。

## 14.3 Loss

对选中 query 的全部类别 logits 使用全 0 BCE target：

```python
loss_hard_bg = BCEWithLogits(
    selected_logits,
    zeros_like(selected_logits),
).mean()
```

默认权重：

```yaml
loss_hard_bg: 0.25
hard_bg_start_epoch: 5
```

warmup 前返回可求导的 0：

```python
outputs["pred_logits"].sum() * 0.0
```

---

# 15. PostProcessor 质量重排序

## 15.1 配置

```python
quality_rerank=False
quality_power=0.5
```

## 15.2 focal/sigmoid 分支

在 flatten 和 top-k **之前**：

```python
scores = torch.sigmoid(logits)
if quality_rerank:
    quality = torch.sigmoid(outputs["pred_quality"])
    scores = scores * quality.pow(quality_power)
scores, index = torch.topk(scores.flatten(1), ...)
```

## 15.3 softmax 分支

在每个 query 获得 class score 后、query top-k 前乘 quality。

## 15.4 兼容性

当任意条件成立时不重排序：

- `quality_rerank=False`；
- outputs 不含 `pred_quality`。

`quality_power=0` 应等效于不重排序。

---

# 16. 配置设计

不要复制整份大配置。每个 ablation config 应 include 用户当前稳定的 wood baseline。

示例：

```yaml
__include__: ["../deim_hgnetv2_l_wood_960.yml"]

output_dir: ./output/sqmal/q1_quality

DFINETransformer:
  use_quality_head: true
  quality_hidden_dim: 128
  quality_aux_last_n: 0

DEIMCriterion:
  losses: ["boxes", "local", "mal", "quality"]
  weight_dict:
    loss_mal: 1.0
    loss_bbox: 5.0
    loss_giou: 2.0
    loss_fgl: 0.15
    loss_ddf: 1.5
    loss_quality: 0.5

PostProcessor:
  quality_rerank: true
  quality_power: 0.5
```

真实 baseline 权重以用户当前配置为准，不得用示例覆盖。

五组配置：

## Q0

完全复用 baseline，只改 output_dir。

## Q1

```text
quality head
quality loss
quality rerank
原 MAL
```

## Q2

```text
Q1
+ defectness head
+ defect loss
仍使用原 MAL
```

## Q3

```text
Q2
+ SQ-MAL 替换 main MAL
不启用 hard background
```

## Q4

```text
Q3
+ hard background mining
```

---

# 17. 单元测试

## 17.1 Semantic support

构造 64×64 mask 和 box，验证：

1. pred box=GT box，support 接近 1；
2. pred box只覆盖半个 mask，support 降低；
3. pred box大面积背景，support 降低；
4. 细线 mask 在正确 bbox 中 support 不应接近 0；
5. invalid mask 返回 1；
6. 空 matched query 返回空 tensor；
7. 不产生 NaN/Inf；
8. CPU 与 CUDA 结果误差可接受；
9. AMP 下可运行。

## 17.2 Quality loss

验证：

- 正负分开归一化；
- 无正样本时不报错；
- 无负样本时不报错；
- invalid mask 时 target 等于 IoU；
- `semantic_beta=0` 时 SQ target 等于原 MAL target 的 IoU 部分。

## 17.3 Postprocessor

验证：

- 质量较低的高分类分数 query 在 rerank 后可以下降；
- `quality_power=0` 与 baseline 排序相同；
- 关闭开关时输出完全一致；
- focal 与 softmax 两条分支都可运行。

## 17.4 Dataset

验证：

- polygon 和 RLE 都能读取；
- `mask_valid` 与 boxes 长度一致；
- resize/flip 后 mask 与 box 对齐；
- 空标注图像可读取。

---

# 18. Smoke Test

新增脚本或命令完成以下检查。

## 18.1 Dataset smoke

读取 8 张训练图：

```text
image tensor shape
boxes shape
labels shape
masks shape
mask_valid shape
```

断言：

```text
N_boxes == N_labels == N_masks == N_mask_valid
```

输出可视化。

## 18.2 Model forward smoke

用 batch size=2 前向：

```text
pred_logits: [B,Q,9]
pred_boxes: [B,Q,4]
pred_quality: [B,Q,1]
pred_defect_logits: [B,1,Hd,Wd]
```

## 18.3 Backward smoke

计算总 loss 并反向，检查：

- loss 有限；
- quality head 有非零梯度；
- defectness head 有非零梯度；
- classifier 有非零梯度；
- bbox head 有非零梯度；
- 无 inplace/autograd 报错。

## 18.4 短训练

先运行：

```text
10 iterations
100 iterations
1 epoch
```

再允许全量训练。

---

# 19. 日志要求

每个 epoch 至少记录：

```text
loss_sqmal
loss_quality
loss_defect
loss_hard_bg
semantic_beta_current
mean_matched_iou
mean_semantic_support
mean_quality_target
mean_quality_pred_pos
mean_quality_pred_neg
hard_bg_selected_per_image
valid_mask_ratio_in_batch
```

每次验证额外输出：

```text
quality_vs_iou_spearman
quality_vs_joint_target_spearman
```

不得每个 iteration 打印大 tensor。

---

# 20. 验收标准

## 20.1 工程验收

必须满足：

- 原 baseline 配置可运行；
- 所有新开关关闭时与 baseline 输出一致；
- 原 checkpoint 可加载；
- 新 checkpoint 可保存和恢复；
- 训练、验证、推理、导出至少完成 smoke；
- 单元测试通过；
- 无 NaN/Inf；
- 不改变 COCO bbox、category 和 annotation ID；
- 有完整实现报告。

## 20.2 实验初步验收

相对于 Q0，Q4 的建议目标：

```text
整体 AP 提升 >= 0.8
AP50 下降不超过 0.5
Recall@0.5 下降不超过 0.3 个百分点
TIDE Background dAP 相对下降 >= 15%
TIDE False Positive dAP 明显下降
Quartzity、Crack、Blue_stain 中至少两个类别 FP/GT 下降
```

未达到时不允许直接宣称有效，应根据日志定位：

- quality head 是否学到；
- semantic support 是否过低；
- invalid mask 比例是否过高；
- hard background 是否过强；
- rerank power 是否过大。

---

# 21. 必须生成的实现报告

`docs/SQ_MAL_IMPLEMENTATION_REPORT.md` 必须包含：

1. 当前 commit 和分支；
2. 修改文件列表；
3. 每个文件修改摘要；
4. 新配置参数说明；
5. 数据转换报告；
6. mask valid 比例；
7. 单元测试结果；
8. smoke test 输出 shape；
9. 运行命令；
10. 已知限制；
11. 回滚方式；
12. Q0～Q4 对应功能表；
13. 不应提交的大文件列表。

---

# 22. 推荐 Git 提交顺序

```text
commit 1: feat(data): add semantic-map to COCO RLE conversion
commit 2: feat(data): load masks and mask_valid safely
commit 3: feat(sqmal): add semantic support utilities and tests
commit 4: feat(model): add query quality head and reranking
commit 5: feat(model): add binary defectness auxiliary head
commit 6: feat(loss): add SQ-MAL and hard-background mining
commit 7: config: add SQ-MAL ablation configs
commit 8: test: add smoke tests and implementation report
```

每个 commit 必须可独立阅读，禁止将所有改动压在一次提交中。

---

# 23. Codex 执行顺序

Codex 应严格按以下顺序工作：

```text
1. 仓库预检
2. 找到用户实际 wood baseline config
3. 运行 baseline forward/eval smoke
4. 实现 semantic map → COCO RLE 工具
5. 生成小样本 COCO 并验证
6. 修改 loader 和 transforms
7. 单元测试 dataset
8. 实现 semantic support 纯函数
9. 单元测试 semantic support
10. 实现 Q1 quality head + loss + rerank
11. 完成 Q1 smoke
12. 实现 Q2 defectness head + loss
13. 完成 Q2 smoke
14. 实现 Q3 SQ-MAL
15. 完成 Q3 smoke
16. 实现 Q4 hard background
17. 完成 Q4 smoke
18. 生成 Q0～Q4 configs
19. 运行 100 iteration 对比
20. 生成实现报告
```

不得跳过数据可视化验证直接开始全量训练。

---

# 24. 交给 Codex 的执行提示词

将本文件放在仓库根目录后，向 Codex 发送：

```text
请读取仓库根目录的 SQ_MAL_Codex_Implementation_Spec.md，并严格按照文档分阶段实现。

要求：
1. 先检查当前分支源码，不要假定它与官方 main 完全一致。
2. 不要覆盖我的 baseline 配置和现有实验代码。
3. 所有新增功能必须默认关闭并保持向后兼容。
4. 每完成一个阶段就运行对应单元测试和 smoke test。
5. 不要开始完整训练，只完成代码、数据转换工具、Q0-Q4 配置、测试和实现报告。
6. 遇到路径未知时，将其做成 CLI 参数或配置项，不要写死。
7. 最终给出修改文件、运行命令、测试结果、未完成项和风险说明。
```

---

# 25. 参考资料

- DEIM 官方仓库：`https://github.com/Intellindust-AI-Lab/DEIM`
- DEIM CVPR 2025：`https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html`
- Wood Defect Detection：`https://datasetninja.com/wood-defect-detection`
- D-FINE：`https://arxiv.org/abs/2410.13842`

---

## 完成定义

只有同时满足以下条件，任务才算完成：

```text
数据能生成有效 RLE mask
loader/transform 保持实例字段对齐
Q0 baseline 未被破坏
Q1-Q4 均可独立启动
新 loss 可前向和反向
质量重排序发生在 top-k 前
单元测试与 smoke test 通过
实现报告完整
```
