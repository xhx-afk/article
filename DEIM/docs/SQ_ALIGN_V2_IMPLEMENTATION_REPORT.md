# SQ-Align V2 实现报告

## 1. 结论

本地项目已从旧联合质量方案直接重构为 SQ-Align V2，不提供旧接口、旧配置或 checkpoint key 兼容层。主分类损失始终使用原 DEIM MAL。

已完成并验证：

- all-query best-IoU localization quality；
- defectness map 的 ROI top-k query semantic evidence；
- `class × loc quality × semantic quality` 的 top-k 前融合；
- 按同一最终分数选择远背景并使用 pairwise ranking loss；
- R0～R4 干净消融配置；
- 19 项单元/数据链/完整模型测试；
- 2 图 CUDA AMP forward/backward；
- 10 iteration 和 100 iteration 有界合成优化 smoke；
- COCO 预训练共享权重加载检查；
- 统一评估、可视化和 power sweep 工具。

未运行完整训练、1 epoch 真实数据训练或精度评估，符合“不要开始完整训练”的限制。当前本机只有 1 张 CUDA GPU，因此双 GPU smoke 留待远程服务器执行。

## 2. Git 状态

- 父仓库根目录：`E:\article`
- 当前分支：`feature/sq-mal-v1`
- 修改前本地 HEAD：`2830352a9d72bd05e0cd202401b76e67a162c95a`
- 修改后本地 HEAD：`2830352a9d72bd05e0cd202401b76e67a162c95a`
- 已知重构前远程快照参考：`39b9e10f3bfc11738b12442a13f9b5ed44ee5dc0`

按用户要求没有创建 commit，也没有 push。父仓库当前把整个 `DEIM_sqmal_v1/` 显示为未跟踪目录，因此不能把当前 HEAD 解释为该子目录内容的独立提交。

## 3. 核心实现

### 3.1 All-query localization quality

`build_all_query_best_iou_target(pred_boxes, targets) -> [B,Q]` 对每个 query 计算：

\[
t_i^{loc}=\max_j IoU(B_i,G_j)
\]

无 GT 时目标为 0。预测框和 GT 均在 target 构造中 detach，计算使用 float32，支持 batch 内不同 GT 数、空 GT、空 query 和 AMP。

`pred_loc_quality: [B,Q,1]` 是 logit。损失为分组 BCEWithLogits：

\[
L_{locq}=w_n\,BCE_{IoU\ge0.1}+w_f\,BCE_{IoU<0.1}+w_m\,BCE_{matched}
\]

默认 `w_n=1.0`、`w_f=0.1`、`w_m=1.0`。它不是“matched=IoU、unmatched=0”。

### 3.2 Query semantic evidence

`pool_query_semantic_evidence(defect_logits, pred_boxes) -> [B,Q,1]` 执行：

1. detach normalized `cxcywh` boxes；
2. 映射并 clamp 到 defect map 坐标；
3. `roi_align(output_size=7, aligned=True, sampling_ratio=2)`；
4. 对 ROI logits 做 sigmoid；
5. 对 `ceil(49 × 0.20)=10` 个最高响应像素求均值。

语义监督：matched query 为 1，unmatched 且 best-IoU `<0.1` 为 0，unmatched near-GT query 忽略。梯度进入 defectness head 和 encoder，但不通过 ROI boxes 进入 bbox 坐标。

### 3.3 Dual-quality final score

`compose_dual_quality_scores()` 在 flatten/top-k 前计算：

\[
S_{i,c}=\sigma(l_{i,c})\cdot\sigma(z_i^{loc})^{\eta}\cdot(q_i^{sem})^{\delta}
\]

默认 `η=0.25`、`δ=0.25`。`pred_sem_quality` 已是概率，不重复 sigmoid。PostProcessor 可独立开启 loc、semantic 或同时开启。

### 3.4 Final-score-aligned background ranking

候选必须同时满足：

- 未被 Hungarian matcher 选中；
- 与任意 GT 的 best-IoU `<0.1`。

选择依据为 detached `max_c S[i,c]`，每图默认 top-5。正样本使用 matched query 的真实类别最终分数，最弱正样本与最强背景配对：

\[
L_{rank}=mean\left[softplus\left(\frac{S_{bg}-S_{pos}+0.05}{0.10}\right)\right]
\]

top-k index 使用 detached score；损失使用原 score，因此分类、loc quality、semantic/defectness 三条分支均可获得梯度。

### 3.5 分支规则

新损失只在 main decoder 输出计算。decoder aux、pre、encoder aux 和 DN 继续保留原 MAL/bbox/local 路径，不复制 loc/defect/query-semantic/rank 损失。

## 4. 接口与 shape

| 接口 | 输入 | 输出 |
|---|---|---|
| `QueryLocalizationQualityHead` | `[B,Q,C]` | `[B,Q,1]` logits |
| `DefectnessHead` | `[B,C,H,W]` | `[B,1,H,W]` logits |
| `pairwise_box_iou_cxcywh` | `[N,4]`, `[M,4]` | `[N,M]` |
| `build_all_query_best_iou_target` | `[B,Q,4]`, targets | `[B,Q]` |
| `pool_query_semantic_evidence` | `[B,1,Hd,Wd]`, `[B,Q,4]` | `[B,Q,1]` probability |
| `build_query_semantic_supervision` | `[B,Q]`, indices | target/valid/positive/far-bg `[B,Q]` |
| `compose_dual_quality_scores` | `[B,Q,C]` + qualities | `[B,Q,C]` |
| `select_final_score_background` | `[B,Q,C]`, `[B,Q]`, indices | selected `[B,Q]`, counts `[B]` |
| `final_score_ranking_loss` | final scores + matches | scalar |
| `build_union_defect_target` | targets + `(Hd,Wd)` | `[B,1,Hd,Wd]` |

训练态 R4 主输出：

```text
pred_logits        [B,Q,num_classes]
pred_boxes         [B,Q,4]
pred_loc_quality   [B,Q,1]
pred_sem_quality   [B,Q,1]
pred_defect_logits [B,1,Hd,Wd]
```

## 5. R0～R4

| 配置 | MAL | loc head/loss/rerank | defect pixel loss | semantic ROI/loss/rerank | final-bg rank |
|---|---:|---:|---:|---:|---:|
| R0 `r0_baseline.yml` | ✓ | – | – | – | – |
| R1 `r1_all_query_loc.yml` | ✓ | ✓ | – | – | – |
| R2 `r2_loc_defect_aux.yml` | ✓ | ✓ | ✓ | – | – |
| R3 `r3_query_semantic.yml` | ✓ | ✓ | ✓ | ✓ | – |
| R4 `r4_final_score_rank.yml` | ✓ | ✓ | ✓ | ✓ | ✓ |

继承链保持同一 Wood baseline、960 输入、训练周期、增强、优化器和数据链，仅覆盖对应消融功能。

## 6. 测试结果

环境：Python 3.11.13、PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118、NVIDIA GeForce RTX 2060。

完整测试命令：

```powershell
conda run -n yolov11 python -m unittest discover -s tests -p "test_*.py" -v
```

最终结果：`Ran 19 tests in 4.131s — OK`。

覆盖内容包括：

- 完全重合、无交集、多 GT 最大 IoU、空 GT、空 query、batch 变长 GT、AMP 和 target detach；
- ROI 高/低响应、细线 top-k、top-k ratio、越界 clamp、空 batch/query、defect gradient、box detach；
- near-GT unmatched semantic ignore；
- loc/semantic power 与 top-k 前融合；
- final-score far-bg 候选、top-5、正负顺序损失、无正/无背景、三分支梯度；
- semantic RLE、mask_valid、SanitizeBoundingBoxes、Mosaic mask canvas；
- R0～R4 构建、COCO checkpoint 共享权重、2 图完整模型 CUDA AMP forward/backward；
- 新 loss 仅 main 分支；分类、bbox、loc head、defect head、encoder 梯度非零。

CLI 启动检查通过：

```powershell
conda run -n yolov11 python tools/wood/evaluate_sqalign.py --help
conda run -n yolov11 python tools/wood/visualize_defectness_and_query_scores.py --help
conda run -n yolov11 python tools/wood/sweep_score_powers.py --help
conda run -n yolov11 python tools/wood/smoke_test_sqalign.py --help
```

## 7. Forward/backward 与短程优化

2 图完整模型 CUDA AMP forward/backward 通过；输出 shape 正确，4 个新 loss 均有限，所有要求的梯度组非零。

10 iteration 命令：

```powershell
conda run -n yolov11 python tools/wood/smoke_test_sqalign.py `
  --iterations 10 --amp --num-classes 3 `
  -t weight/deim_dfine_hgnetv2_l_coco_50e.pth `
  --output-json output_2/sqalign_smoke/10_iterations.json
```

最终复跑结果：4.75 s、2.10 iter/s、峰值显存 642.74 MB，无 NaN/OOM，且 `nonfinite_parameters=[]`；平均新损失：loc `0.72780`、defect `0.69639`、semantic `0.17930`、rank `0.05028`。

100 iteration 命令同上，将 `--iterations` 改为 100。结果：44.44 s、2.25 iter/s、峰值显存 882.61 MB，无 NaN/OOM；平均新损失：

| 指标 | 值 |
|---|---:|
| `loss_loc_quality` | 0.72729 |
| `loss_defect` | 0.67301 |
| `loss_query_semantic` | 0.19035 |
| `loss_final_bg_rank` | 0.05059 |
| locq-best-IoU batch Spearman | 0.31931 |
| locq near / far | 0.49896 / 0.49867 |
| semq pos / far-bg | 0.56410 / 0.52648 |
| semantic gap | +0.03763 |
| selected backgrounds/image | 5.0 |

这是 128×128、batch=2、随机合成图像/目标的数值与梯度 smoke，不是收敛或 AP 结论。`rank_violation_ratio=1.0` 表明 100 个随机短步不足以训练好排序，真实数据实验仍需验证。

## 8. 预训练权重

本地 `weight/deim_dfine_hgnetv2_l_coco_50e.pth` 已检查。测试可加载超过 100 个共享参数；100-step smoke 实际加载 1156 个 shape-compatible tensor。

预期缺失包括：

- 新 `decoder.dec_loc_quality_head.*`；
- 新 `defectness_head.*`；
- smoke 将类别数临时设为 3 时，分类/denoising heads 因 shape 不同而不加载。

这是允许的非兼容行为。远程真实九类训练仍应使用现有 `train.py ... -t checkpoint` tuning 入口处理类别 head。

## 9. 评估工具

```bash
python tools/wood/evaluate_sqalign.py \
  -c configs/deim_dfine/ablation_sqalign/r4_final_score_rank.yml \
  -r output_2/sqalign_v2/r4_final_score_rank/best_stg2.pth \
  --output-dir output_2/sqalign_v2/r4_final_score_rank/eval
```

输出包含 COCO AP/AP50/AP75/APM/APL、逐类 AP、固定阈值 precision/recall/混淆矩阵、background FP、Recall@IoU、ECE/LaECE、all/top100/top300 loc Spearman、TP/background/near-GT loc 分布、semantic TP/background 分布、semantic ROC-AUC/PR-AUC、final-score ROC-AUC、defect Dice/precision/recall。

Power sweep 强制 `--split val`，不会允许用 test 选择 power。

## 10. 删除文件

- `engine/deim/sqmal.py`
- `configs/deim_dfine/ablation_sqmal/q0_baseline.yml`
- `configs/deim_dfine/ablation_sqmal/q1_quality.yml`
- `configs/deim_dfine/ablation_sqmal/q2_defect_aux.yml`
- `configs/deim_dfine/ablation_sqmal/q3_sqmal.yml`
- `configs/deim_dfine/ablation_sqmal/q4_sqmal_hbg.yml`
- `tests/test_sqmal_semantic_support.py`
- `tests/test_sqmal_postprocessor.py`
- `tests/test_sqmal_losses.py`
- `tests/test_sqmal_forward.py`
- `tests/test_sqmal_evaluator.py`
- `tests/test_sqmal_dataset_tools.py`
- `tools/wood/smoke_test_sqmal.py`
- `docs/SQ_MAL_IMPLEMENTATION_REPORT.md`
- 对应旧 `.pyc` cache artifacts

旧 evaluator/data/test 文件中仍有价值的 semantic RLE 能力已用新名称替换，不保留旧 wrapper。

## 11. 修改文件

- `engine/deim/__init__.py`
- `engine/deim/deim.py`
- `engine/deim/dfine_decoder.py`
- `engine/deim/deim_criterion.py`
- `engine/deim/postprocessor.py`
- `tools/wood/augment_coco_with_semantic_masks.py`

## 12. 新增或重命名后的文件

- `engine/deim/semantic_query_alignment.py`
- `configs/deim_dfine/ablation_sqalign/r0_baseline.yml`
- `configs/deim_dfine/ablation_sqalign/r1_all_query_loc.yml`
- `configs/deim_dfine/ablation_sqalign/r2_loc_defect_aux.yml`
- `configs/deim_dfine/ablation_sqalign/r3_query_semantic.yml`
- `configs/deim_dfine/ablation_sqalign/r4_final_score_rank.yml`
- `tests/sqalign_test_utils.py`
- `tests/test_all_query_iou_target.py`
- `tests/test_query_semantic_pooling.py`
- `tests/test_dual_quality_score.py`
- `tests/test_background_ranking.py`
- `tests/test_sqalign_forward_backward.py`
- `tests/test_semantic_mask_data_chain.py`
- `tools/wood/evaluate_sqalign.py`
- `tools/wood/visualize_defectness_and_query_scores.py`
- `tools/wood/sweep_score_powers.py`
- `tools/wood/smoke_test_sqalign.py`
- `tools/wood/optional_dependency_stubs.py`
- `tools/wood/semantic_mask_data_utils.py`
- `tools/wood/validate_semantic_dataset.py`
- `tools/wood/visualize_semantic_masks.py`
- `tools/wood/requirements_sqalign.txt`
- `docs/SQ_ALIGN_V2_IMPLEMENTATION_REPORT.md`

生成但不属于源码的 smoke 结果：

- `output_2/sqalign_smoke/10_iterations.json`
- `output_2/sqalign_smoke/100_iterations.json`

## 13. 已删除接口

- `use_sqmal`
- `loss_labels_sqmal()` / `loss_sqmal`
- `compute_semantic_support()`
- `compute_sq_quality_target()`
- `semantic_beta_schedule()` 及 semantic beta/warmup 配置
- `select_hard_background_queries()`
- `loss_hard_bg` 及旧全 0 BCE
- `QueryQualityHead` / `pred_quality` / `dec_quality_head`
- `quality_rerank` / `quality_power`
- 旧 evaluator、test 和 checkpoint wrapper

源代码/配置/测试/工具中已执行无结果审计；只在规范、验收指南和本报告的删除说明中保留文字引用。

## 14. 风险和未完成项

1. 本机没有 Wood 真实 train/val 数据；100-step 是合成 smoke，没有验证真实数据收敛、AP 或 background FP。
2. 本机只有 1 张 RTX 2060；双 GPU torchrun smoke 未执行。
3. 按用户要求未运行 1 epoch、完整训练、quick-balanced 评估、3 seeds 或完整数据集复验。
4. 新 head 不兼容旧 checkpoint key，这是明确授权行为；恢复训练必须使用 V2 checkpoint，旧权重只适合作为 tuning 初始化。
5. ROI top-k 对非常细的缺陷依赖 defect feature 分辨率；真实数据需检查 Crack/resin 可视化并在 val 上选择 ratio/power。
6. 排名 loss 在合成 100-step 中 violation 未下降；R4 必须以真实 R3/R4 对照决定是否保留，不能仅凭 smoke 判断有效。
7. 当前评估工具已通过编译和 CLI 启动检查，但因缺少本地真实数据/V2 checkpoint，尚未完成端到端指标文件实跑。
8. pycocotools 在 mask 测试中产生 NumPy `copy` 参数 deprecation warning，不影响测试结果。

## 15. 不应提交的文件

- `weight/*.pth`、训练 checkpoint、EMA/optimizer state；
- `output_2/` 下 smoke、训练、评估缓存和图像；
- 数据集图片、COCO JSON 副本、semantic maps；
- `__pycache__/`、`.pyc`、临时日志。

本次没有提交或上传任何权重、大文件或 GitHub 内容。
