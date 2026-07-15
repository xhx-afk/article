# SQ-MAL 实现报告

## 1. 分支与版本

- 开发分支：`feature/sq-mal-v1`
- 起点提交：`40b02e6c46a4aabed9b2d0e49c4bd531f4a943a2`
- 完成主要代码与 smoke 的验证提交：`419c4fc`
- 最终文档提交：以本文件所在分支的 `git rev-parse HEAD` 为准
- 本地测试环境：Windows PowerShell、Python 3.11.13、CUDA、PyTorch/torchvision 来自 `yolov11` Conda 环境

开发时保留了工作区原有 TSEM、训练结果、配置修改和输出文件；SQ-MAL 提交均使用路径限定，没有提交或回退这些无关改动。

## 2. 实现范围

第一版严格保持以下边界：

- 未修改 Hungarian matcher；
- 未修改 backbone、P2、query 数量或输入尺寸；
- 未实现 MB-FDR、类别重采样、oversampling 或 copy-paste；
- 保留原 MAL 与 D-FINE LQE；
- 所有新功能默认关闭；
- quality 重排发生在 flatten/top-k 之前。

## 3. 文件与修改摘要

### 数据工具

- `tools/wood/sqmal_data_utils.py`：semantic specification、类别 canonical map、RLE 编解码共享函数。
- `tools/wood/augment_coco_with_semantic_masks.py`：semantic map 到实例 RLE，尺寸校验、connected components 评分、报告和抽样可视化。
- `tools/wood/validate_sqmal_dataset.py`：RLE、mask_valid、bbox/category/annotation ID 不变性校验。
- `tools/wood/visualize_sqmal_masks.py`：bbox 与实例 mask 抽样叠加。
- `tools/wood/smoke_test_sqmal.py`：真实 COCO batch 的 forward/backward/checkpoint smoke。

### 数据加载

- `engine/data/dataset/coco_dataset.py`：兼容 polygon、uncompressed RLE、compressed RLE 和空 segmentation；输出 `mask_valid`。
- `engine/data/transforms/_transforms.py`：sanitation 同步 labels、area、iscrowd、masks、mask_valid、mixup。
- `engine/data/transforms/functional.py`：legacy crop 同步 mask_valid。
- `engine/data/dataloader.py`：MixUp 和多尺度 resize 同步 masks/mask_valid。

### 模型与损失

- `engine/deim/sqmal.py`：DefectnessHead、QueryQualityHead、积分图 semantic support、union defect target、quality target、hard-background 选择。
- `engine/deim/dfine_decoder.py`：可选 quality head、DN 拆分、main/可选 aux 输出、deploy 裁剪。
- `engine/deim/deim.py`：可选 defectness head，训练态输出 defect logits。
- `engine/deim/deim_criterion.py`：SQ-MAL、quality、defect、hard-background loss，以及 main/aux/pre/enc/DN 分支控制。
- `engine/deim/postprocessor.py`：sigmoid/softmax 路径 top-k 前 quality 重排。
- `engine/solver/det_engine.py`：epoch 注入；`metric_*` 只记录、不进入总 loss。
- `engine/core/yaml_utils.py`：UTF-8 YAML 和独立顶层配置字典，避免 Q0-Q4 连续加载污染。

## 4. 配置参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `DEIM.use_defectness_head` | false | 构建二值 defectness head |
| `DEIM.return_defect_map_in_eval` | false | 评估时是否返回 defect map |
| `DFINETransformer.use_quality_head` | false | 构建 query quality head |
| `DFINETransformer.quality_hidden_dim` | 128 | quality head 隐藏维度 |
| `DFINETransformer.quality_aux_last_n` | 0 | 最后多少个 aux 层输出 quality |
| `DEIMCriterion.use_sqmal` | false | main 分类使用 SQ-MAL |
| `semantic_beta` | 0.5 | semantic support 指数上限 |
| `semantic_warmup_epochs` | 0 | beta warmup epoch 数 |
| `semantic_min_mask_area` | 2.0 | 有效 GT mask 最小面积 |
| `use_hard_bg` | false | 启用困难背景挖掘 |
| `hard_bg_topk` | 20 | 每图困难背景 query 数 |
| `hard_bg_iou_threshold` | 0.1 | 远离 GT 的最大 IoU 阈值 |
| `hard_bg_start_epoch` | 0 | hard-bg 开始 epoch |
| `PostProcessor.quality_rerank` | false | top-k 前质量重排 |
| `PostProcessor.quality_power` | 0.5 | quality 对最终分数的指数 |

## 5. Q0-Q4 功能表

| 配置 | Quality | Defect Aux | SQ-MAL | Hard BG |
|---|---:|---:|---:|---:|
| `q0_baseline.yml` |  |  |  |  |
| `q1_quality.yml` | ✓ |  |  |  |
| `q2_defect_aux.yml` | ✓ | ✓ |  |  |
| `q3_sqmal.yml` | ✓ | ✓ | ✓ |  |
| `q4_sqmal_hbg.yml` | ✓ | ✓ | ✓ | ✓ |

五个配置均 include `configs/deim_dfine/deim_hgnetv2_l_wood.yml`，没有复制或覆盖 baseline。训练前必须把 baseline 的 train/val annotation 路径指向转换后的 `*_sqmal.json`。

## 6. 数据转换验证

本地合成九类灰度 semantic map 测试结果：

- images：1；
- annotations：9；
- valid masks：9；
- mask valid ratio：100%；
- bbox changed：0；
- category changed：0；
- annotation ID changed：0；
- 9 个 compressed RLE 均成功解码。

真实 Wood semantic maps 未位于当前工作区，因此真实 train/val/test 的 valid ratio 尚未生成。必须在服务器执行转换、可视化和校验后再填写论文数据。

## 7. 单元测试结果

统一命令：

```bash
python -m unittest \
  tests.test_sqmal_dataset_tools \
  tests.test_sqmal_coco_loader \
  tests.test_sqmal_semantic_support \
  tests.test_sqmal_losses \
  tests.test_sqmal_postprocessor \
  tests.test_sqmal_forward -v
```

结果：`Ran 13 tests ... OK`。

已覆盖：

- 灰度 map 到九类实例 compressed RLE；
- polygon、compressed RLE、空 segmentation；
- mask_valid 与 box sanitation 对齐；
- full/half/background/细线 semantic support；
- invalid mask fallback=1；
- CUDA AMP；
- SQ-MAL、quality、defect、hard-bg 前向与反向；
- focal/softmax quality rerank；
- quality power=0 和缺少 quality 输出的兼容性；
- Q0 state 加载到 Q4，只缺 quality/defectness 新键；
- Q0/Q4 baseline bbox/logits 一致；
- Q4 deploy 前向。

## 8. Smoke 输出 shape

真实 YAML 测试将输入缩小为 `B=2, 128x128, num_classes=3`，只用于结构测试：

```text
pred_logits:        [2, 300, 3]
pred_boxes:         [2, 300, 4]
pred_quality:       [2, 300, 1]
pred_defect_logits: [2, 1, 16, 16]
```

Q4 的 classifier、bbox head、quality head、defectness head 均验证有非零梯度，全部新 loss 有限。deploy 模式输出 logits/boxes/quality，默认不返回 defect map。

## 9. 运行命令

转换 train 示例：

```bash
python tools/wood/augment_coco_with_semantic_masks.py \
  --images-dir /path/images/train \
  --semantic-maps-dir /path/Semantic_Maps/train \
  --semantic-spec /path/Semantic_Map_Specification.txt \
  --input-coco /path/instances_train.json \
  --output-coco /path/instances_train_sqmal.json \
  --report-dir /path/reports/train \
  --semantic-suffix _segm
```

校验与可视化：

```bash
python tools/wood/validate_sqmal_dataset.py \
  --sqmal-coco /path/instances_train_sqmal.json \
  --original-coco /path/instances_train.json \
  --report /path/reports/train/validation.json

python tools/wood/visualize_sqmal_masks.py \
  --coco /path/instances_train_sqmal.json \
  --images-dir /path/images/train \
  --output-dir /path/reports/train/manual_review
```

真实数据 smoke：

```bash
python tools/wood/smoke_test_sqmal.py \
  -c configs/deim_dfine/ablation_sqmal/q4_sqmal_hbg.yml \
  --coco-json /path/instances_train_sqmal.json \
  --images-dir /path/images/train \
  --batch-size 2 \
  --device cuda
```

训练命令见 `SQ_MAL_User_Execution_Guide.md`，必须先 Q0，再按 Q1、Q2、Q3、Q4 顺序执行。

## 10. 未完成项与限制

- 未访问真实 Semantic_Maps，未生成人工抽查通过的 train/val/test RLE JSON。
- 未执行真实数据 10 iteration、100 iteration、1 epoch 或完整 Q0-Q4 训练。
- 未执行双 GPU DDP smoke。
- 未执行 ONNX/TensorRT 文件导出；仅完成 `model.deploy()` 前向。
- 未生成真实 valid mask ratio、TIDE、AP、ECE 和多随机种子结果。
- 当前 softmax postprocessor 保留了基线原有隐式 softmax 维度行为；默认 DEIM 使用 focal/sigmoid 路径。

在上述实验完成前，不应宣称 SQ-MAL 已带来 AP 或 TIDE 改善。

## 11. Git 提交与回滚

```text
cc59279 feat(data): add semantic map to COCO RLE conversion
f4a9339 feat(data): load masks and mask_valid safely
9b8f1b3 feat(sqmal): add semantic support utilities and tests
2eb2d6c feat(model): add query quality head and reranking
b183a18 feat(model): add binary defectness auxiliary head
e4a59e2 feat(loss): add SQ-MAL and hard-background mining
96b072d config: add SQ-MAL ablation configs
419c4fc test: add SQ-MAL loader and model smoke tests
```

回滚单阶段：

```bash
git revert <commit_hash>
```

切回原开发分支：

```bash
git checkout feature/tsem-v1
```

不要使用 `git reset --hard`，当前工作区仍包含用户未提交改动。

## 12. 不应提交的大文件

- `*.pth`、`*.pt`、ONNX/TensorRT engine；
- `output/`、TensorBoard events、训练日志和可视化批量图片；
- 原始图像、Semantic_Maps；
- 转换后的完整 train/val/test COCO JSON（应存数据盘或制品存储）；
- cache、`__pycache__`、临时 debug 数据。

