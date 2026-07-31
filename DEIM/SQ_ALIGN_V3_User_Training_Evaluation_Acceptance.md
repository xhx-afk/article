# SQ-Align V3：用户训练、统一评估与验收手册

> 配套文件：`SQ_ALIGN_V3_Codex_Destructive_Refactor_Spec.md`  
> 目标：将文档发送给 Codex 完成破坏性重构后，用户按本手册验证代码、训练 V0～V4，并使用权重生成单一诊断 JSON。  
> 当前代码基础：`sq-mal-v2`  
> 当前最佳但不足：R3，AP约50.447，仅比baseline高约0.209。

---

# 1. 本次重构的判断标准

本次不是继续调 V2 power，而是替换失效结构。

必须删除：

```text
pred_loc_quality
loss_loc_quality
loc_quality_power
当前raw top-20% semantic score
当前matched semantic target=1
当前far-background ranking
```

新版本应围绕：

```text
Defect Auxiliary
Query-Conditioned Mask
Continuous Semantic Quality
Semantic Suppression Gate
Near-GT Candidate Ranking
```

---

# 2. 路径变量

按实际路径修改：

```bash
export REPO=/home/zxw4090/hjw/DEIM
export ENV_NAME=deim_mamba

export PRETRAIN=$REPO/weight/deim_dfine_hgnetv2_l_coco_50e.pth
export OUT=$REPO/output_2/sqalign_v3

export TRAIN_IMG=/path/to/images/train
export VAL_IMG=/path/to/images/val
export TEST_IMG=/path/to/images/test

export TRAIN_JSON=/path/to/instances_train_sqmal.json
export VAL_JSON=/path/to/instances_val_sqmal.json
export TEST_JSON=/path/to/instances_test_sqmal.json
```

进入环境：

```bash
cd "$REPO"
conda activate "$ENV_NAME"
mkdir -p "$OUT"
```

---

# 3. 备份 V2

```bash
git status
git branch --show-current
git rev-parse HEAD

git add -A
git commit -m "backup: SQ-Align V2 before destructive V3 refactor"
```

备份结果：

```bash
mkdir -p result_backups/sqalign_v2
cp -r output_2/sqalign_v2 result_backups/sqalign_v2/
```

保存 power sweep 汇总。

---

# 4. 将文档发送给 Codex

把：

```text
SQ_ALIGN_V3_Codex_Destructive_Refactor_Spec.md
```

复制到仓库根目录。

发送：

```text
请读取仓库根目录的 SQ_ALIGN_V3_Codex_Destructive_Refactor_Spec.md，并直接重构当前本地sq-mal-v2代码。

无需兼容v1/v2，允许删除、重命名和覆盖旧接口与配置。
删除当前Localization Quality、raw semantic top-k乘分和far-background ranking。
保留原DEIM MAL与已有效的defect auxiliary。
实现query-conditioned ROI mask、连续mask quality、semantic suppression gate、near-GT ranking，以及单一JSON统一评估脚本。
先完成测试、smoke、V0-V4配置和实现报告，不启动完整训练。
```

---

# 5. Codex 完成后的文件检查

预期：

```text
engine/deim/continuous_semantic_alignment.py
tools/wood/evaluate_sqalign_v3.py
tools/wood/sweep_semantic_gate_v3.py
tools/wood/visualize_sqalign_v3.py
tools/wood/smoke_test_sqalign_v3.py

configs/deim_dfine/ablation_sqalign_v3/v0_baseline.yml
configs/deim_dfine/ablation_sqalign_v3/v1_defect_aux.yml
configs/deim_dfine/ablation_sqalign_v3/v2_query_conditioned_mask.yml
configs/deim_dfine/ablation_sqalign_v3/v3_continuous_semantic_quality.yml
configs/deim_dfine/ablation_sqalign_v3/v4_near_gt_candidate_rank.yml

docs/SQ_ALIGN_V3_IMPLEMENTATION_REPORT.md
```

检查旧接口：

```bash
grep -R "pred_loc_quality" -n engine configs || true
grep -R "loss_loc_quality" -n engine configs || true
grep -R "loc_quality_power" -n engine configs || true
grep -R "select_final_score_background" -n engine || true
grep -R "final_score_ranking_loss" -n engine || true
```

原则上不应返回有效代码。

检查新接口：

```bash
grep -R "pred_query_features" -n engine
grep -R "pred_query_mask_logits" -n engine
grep -R "pred_sem_quality" -n engine
grep -R "compute_continuous_mask_support" -n engine
grep -R "near_gt_candidate_rank_loss" -n engine
grep -R "semantic_suppression_gate" -n engine
```

---

# 6. 单元测试

```bash
pytest -q tests/test_continuous_mask_support.py
pytest -q tests/test_query_conditioned_roi_mask.py
pytest -q tests/test_semantic_suppression_gate.py
pytest -q tests/test_near_gt_candidate_rank.py
pytest -q tests/test_sqalign_v3_forward_backward.py
```

若没有 pytest，使用 Codex 提供的统一测试脚本。

任何失败都不要进入训练。

---

# 7. 数据链复查

V3继续使用现有 RLE instance masks。

运行：

```bash
python tools/wood/validate_sqmal_dataset.py \
  --images-dir "$TRAIN_IMG" \
  --coco-json "$TRAIN_JSON" \
  --num-samples 200 \
  --output-dir "$OUT/data_validation"
```

随机可视化：

```bash
python tools/wood/visualize_sqmal_masks.py \
  --images-dir "$TRAIN_IMG" \
  --coco-json "$TRAIN_JSON" \
  --output-dir "$OUT/data_visuals" \
  --num-images 200 \
  --seed 42
```

重点：

```text
Crack细线
resin多连通域
knot类实例分离
mask_valid同步
Mosaic/Resize后对齐
```

---

# 8. Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/wood/smoke_test_sqalign_v3.py \
  -c configs/deim_dfine/ablation_sqalign_v3/v4_near_gt_candidate_rank.yml \
  -t "$PRETRAIN" \
  --batch-size 2 \
  --iterations 100 \
  --device cuda:0 \
  --amp \
  --output-json "$OUT/smoke_v4.json"
```

检查输出 shape：

```text
pred_logits [B,Q,9]
pred_boxes [B,Q,4]
pred_query_features [B,Q,C]
pred_defect_logits [B,1,Hd,Wd]
pred_query_mask_logits [B,Q,7,7]
pred_sem_quality [B,Q,1]
```

检查新 loss 有限且有梯度：

```text
loss_defect
loss_query_mask
loss_semantic_quality
loss_semantic_rank
loss_candidate_rank
```

---

# 9. 先做短训

每组先运行：

```text
10 iterations
100 iterations
1 epoch
```

观察：

```text
query mask Dice是否上升
semantic target/pred相关性是否上升
背景semantic是否不再全部接近1
candidate rank violation是否下降
是否OOM/NaN
```

健康信号：

```text
matched query mask Dice > near query mask Dice > far background
semantic pred matched > near > far
background ratio_gt_0.95逐渐下降
candidate anchor margin逐渐增大
```

---

# 10. 正式训练命令

## V0 Baseline

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7850 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign_v3/v0_baseline.yml \
--use-amp \
--seed=0 \
-t "$PRETRAIN" \
--output-dir "$OUT/v0_baseline" \
2>&1 | tee "$OUT/v0_baseline/train_console.log"
```

## V1 Defect Auxiliary

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7851 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign_v3/v1_defect_aux.yml \
--use-amp \
--seed=0 \
-t "$PRETRAIN" \
--output-dir "$OUT/v1_defect_aux" \
2>&1 | tee "$OUT/v1_defect_aux/train_console.log"
```

## V2 Query-Conditioned Mask

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7852 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign_v3/v2_query_conditioned_mask.yml \
--use-amp \
--seed=0 \
-t "$PRETRAIN" \
--output-dir "$OUT/v2_query_conditioned_mask" \
2>&1 | tee "$OUT/v2_query_conditioned_mask/train_console.log"
```

## V3 Continuous Semantic Quality

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7853 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign_v3/v3_continuous_semantic_quality.yml \
--use-amp \
--seed=0 \
-t "$PRETRAIN" \
--output-dir "$OUT/v3_continuous_semantic_quality" \
2>&1 | tee "$OUT/v3_continuous_semantic_quality/train_console.log"
```

## V4 Near-GT Candidate Ranking

```bash
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --master_port=7854 --nproc_per_node=2 \
train.py \
-c configs/deim_dfine/ablation_sqalign_v3/v4_near_gt_candidate_rank.yml \
--use-amp \
--seed=0 \
-t "$PRETRAIN" \
--output-dir "$OUT/v4_near_gt_candidate_rank" \
2>&1 | tee "$OUT/v4_near_gt_candidate_rank/train_console.log"
```

---

# 11. 使用权重生成单一诊断 JSON

## V0

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/wood/evaluate_sqalign_v3.py \
  -c configs/deim_dfine/ablation_sqalign_v3/v0_baseline.yml \
  -r "$OUT/v0_baseline/best_stg2.pth" \
  --images-dir "$TEST_IMG" \
  --ann-file "$TEST_JSON" \
  --output-json "$OUT/v0_baseline/evaluation.json" \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 4 \
  --max-dets 300 \
  --score-thresholds 0.05 0.10 0.25 0.50
```

V1～V4只替换 config、checkpoint和output-json。

## V3 示例

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/wood/evaluate_sqalign_v3.py \
  -c configs/deim_dfine/ablation_sqalign_v3/v3_continuous_semantic_quality.yml \
  -r "$OUT/v3_continuous_semantic_quality/best_stg2.pth" \
  --images-dir "$TEST_IMG" \
  --ann-file "$TEST_JSON" \
  --output-json "$OUT/v3_continuous_semantic_quality/evaluation.json" \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 4 \
  --max-dets 300 \
  --score-thresholds 0.05 0.10 0.25 0.50 \
  --semantic-gate-lambda 0 \
  --semantic-gate-gamma 2
```

---

# 12. 验证集 Gate Sweep

只对 V3、V4。

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/wood/sweep_semantic_gate_v3.py \
  -c configs/deim_dfine/ablation_sqalign_v3/v3_continuous_semantic_quality.yml \
  -r "$OUT/v3_continuous_semantic_quality/best_stg2.pth" \
  --images-dir "$VAL_IMG" \
  --ann-file "$VAL_JSON" \
  --lambdas 0 0.05 0.10 0.15 0.20 0.30 \
  --gammas 1 2 3 \
  --output-json "$OUT/v3_continuous_semantic_quality/gate_sweep_val.json" \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 4
```

选择顺序：

```text
AP最高
→ AP75
→ TP-vs-all-FP AUC
→ near-GT FP
→ background FP
```

选定后，仅在 test运行一次。

例如 val选择：

```text
lambda=0.10
gamma=2
```

test：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/wood/evaluate_sqalign_v3.py \
  -c configs/deim_dfine/ablation_sqalign_v3/v3_continuous_semantic_quality.yml \
  -r "$OUT/v3_continuous_semantic_quality/best_stg2.pth" \
  --images-dir "$TEST_IMG" \
  --ann-file "$TEST_JSON" \
  --output-json "$OUT/v3_continuous_semantic_quality/evaluation_gate_selected.json" \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 4 \
  --max-dets 300 \
  --score-thresholds 0.05 0.10 0.25 0.50 \
  --semantic-gate-lambda 0.10 \
  --semantic-gate-gamma 2
```

---

# 13. 检查单一 JSON

```bash
python -m json.tool \
  "$OUT/v3_continuous_semantic_quality/evaluation.json" \
  > /dev/null && echo "[OK] valid JSON"
```

查看关键字段：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/home/zxw4090/hjw/DEIM/output_2/sqalign_v3/v3_continuous_semantic_quality/evaluation.json")
d = json.loads(path.read_text())

print("AP:", d["coco"]["AP"])
print("AP50:", d["coco"]["AP50"])
print("AP75:", d["coco"]["AP75"])
print("AP80:", d["coco"]["AP80"])
print("AP90:", d["coco"]["AP90"])

print("TIDE:", d["tide"])
print("Fixed 0.25:", d["fixed_thresholds"]["0.25"])
print("Semantic all-FP AUC:", d["semantic_quality"]["auc"]["TP_vs_all_FP"])
print("Semantic Spearman:", d["semantic_quality"]["correlation"])
print("Candidate ranking:", d["candidate_ranking"])
print("Warnings:", d["automatic_diagnosis"]["warnings"])
PY
```

---

# 14. 一次汇总 V0～V4

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/home/zxw4090/hjw/DEIM/output_2/sqalign_v3")
experiments = [
    "v0_baseline",
    "v1_defect_aux",
    "v2_query_conditioned_mask",
    "v3_continuous_semantic_quality",
    "v4_near_gt_candidate_rank",
]

header = (
    f"{'experiment':32s} {'AP':>7s} {'AP50':>7s} {'AP75':>7s} "
    f"{'APM':>7s} {'APL':>7s} {'BkgFP':>7s} {'NearFP':>7s} "
    f"{'SemAUC':>8s} {'SemRho':>8s} {'QMask':>8s} {'RankV':>8s}"
)
print(header)

for exp in experiments:
    p = root / exp / "evaluation.json"
    if not p.exists():
        print(f"{exp:32s} MISSING")
        continue
    d = json.loads(p.read_text())
    c = d["coco"]
    fixed = d["fixed_thresholds"]["0.25"]
    sem = d.get("semantic_quality", {})
    auc = sem.get("auc", {}).get("TP_vs_all_FP", None)
    rho = sem.get("correlation", {}).get("top30_per_image_spearman", None)
    qmask = d.get("query_mask", {}).get("matched", {}).get("Dice", None)
    rankv = d.get("candidate_ranking", {}).get("violation_ratio", None)

    def fmt(v):
        return "null" if v is None else f"{v:.4f}"

    print(
        f"{exp:32s} "
        f"{c['AP']*100:7.3f} "
        f"{c['AP50']*100:7.3f} "
        f"{c['AP75']*100:7.3f} "
        f"{c['APM']*100:7.3f} "
        f"{c['APL']*100:7.3f} "
        f"{fixed['background_FP']:7d} "
        f"{fixed['near_GT_FP']:7d} "
        f"{fmt(auc):>8s} "
        f"{fmt(rho):>8s} "
        f"{fmt(qmask):>8s} "
        f"{fmt(rankv):>8s}"
    )
PY
```

---

# 15. V1 验收

V1回答：

> 纯 defect auxiliary 是否有效？

要求：

```text
V1不能包含query mask、semantic quality、gate或ranking。
```

成功信号：

```text
AP不低于V0
AP50/AP75至少一个提升
TIDE Bkg dAP下降
原图 defect Dice有效
```

若 V1低于 V0：

```text
loss_defect 0.5 → 0.25
检查max-pool target膨胀
检查Mosaic mask
检查背景activation
```

---

# 16. V2 验收

V2回答：

> query 与 pixel feature交互的实例语义 mask 是否学到？

检查 JSON：

```text
query_mask.matched.Dice
query_mask.near_unmatched.Dice
query_mask.far_background
per_class matched Dice
```

目标：

```text
matched Dice >= 0.65
matched > near > far
Crack/resin不塌缩
```

可视化：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/wood/visualize_sqalign_v3.py \
  -c configs/deim_dfine/ablation_sqalign_v3/v2_query_conditioned_mask.yml \
  -r "$OUT/v2_query_conditioned_mask/best_stg2.pth" \
  --images-dir "$TEST_IMG" \
  --ann-file "$TEST_JSON" \
  --output-dir "$OUT/v2_query_conditioned_mask/visuals" \
  --num-images 100 \
  --device cuda:0
```

若所有 query mask几乎相同：

```text
检查pred_query_features是否为最终正常query
检查query projection是否真的参与einsum
检查query embedding是否被错误broadcast
```

---

# 17. V3 验收

V3回答：

> 连续语义质量是否能区分TP和全部FP，而不仅是远背景？

重点：

```text
TP vs all-FP AUC
TP vs duplicate AUC
TP vs class-confusion AUC
TP vs localization-FP AUC
top10/top30 Spearman
background saturation
```

建议：

```text
TP-vs-all-FP ROC-AUC >= 0.75
top30 Spearman >= 0.35
background ratio_gt_0.95明显低于V2 raw semantic
```

若 far-background AUC高但 all-FP AUC低：

```text
semantic仍不能处理near-GT错误
进入V4前先确认query mask是否query-conditioned
```

若 gate降低FP但AP下降：

```text
lambda减小
检查class-only AUC是否高于gated AUC
不强行启用gate
```

---

# 18. V4 验收

V4回答：

> per-GT ranking是否减少主要的near-GT错误？

比较 V3 与 V4：

```text
near_GT_FP
duplicate_same_class_FP
class_confusion_FP
localization_FP
candidate violation ratio
AP/AP75
```

成功：

```text
near-GT FP下降
violation ratio下降
AP与AP75不下降
```

若 V4不如V3：

```text
loss_candidate_rank 0.05 → 0.025
start_ratio 0.60 → 0.75
topk_per_gt 3 → 2
margin 0.03 → 0.01
```

仍无效则删除V4，以V3为最终模型。

---

# 19. 自动诊断字段的使用

统一 JSON 应自动给出警告。

常见：

```text
semantic_quality_saturated
```

处理：

```text
检查matched连续target
降低far权重
检查quality head输入
```

```text
query_masks_not_query_conditioned
```

处理：

```text
检查不同query mask余弦相似度
检查query projection
```

```text
gate_reduces_fp_but_hurts_ap
```

处理：

```text
关闭gate或减小lambda
```

```text
near_gt_fp_dominates
```

处理：

```text
检查V4候选覆盖率
```

```text
candidate_rank_hurts_high_iou_ap
```

处理：

```text
降低rank权重，推迟启用
```

---

# 20. 初步成功标准

quick-balanced 单seed：

```text
AP >= 51.0
AP75 >= 54.0
AP50下降 <= 0.3
TP-vs-all-FP AUC高于class-only
near-GT FP下降
far-background FP下降
```

达到后才进行3 seed。

---

# 21. 三随机种子

只运行：

```text
V0
最终V3或V4
```

```text
seed=0
seed=1
seed=2
```

报告：

```text
mean ± std
```

建议：

```text
AP均值提升 >= 0.8
AP75均值提升 >= 1.0
3个seed方向一致
```

---

# 22. 完整数据复验

quick subset只验证机制。

最终回到完整：

```text
train
val
test
```

保持：

```text
image size=960
预训练权重相同
训练周期相同
数据划分相同
```

使用同一个 `evaluate_sqalign_v3.py` 生成完整 test JSON。

---

# 23. 最终模块保留规则

```text
V1 > V0 → 保留defect auxiliary
V2 > V1 → 保留query-conditioned mask
V3 > V2 → 保留continuous semantic quality
gate提高AP → 启用gate
gate只降低FP但AP下降 → 关闭gate
V4 > V3 → 保留near-GT ranking
V4 <= V3 → 删除ranking
```

不为了模块数量保留无效部分。

---

# 24. 最终验收清单

```text
[ ] 旧loc-quality删除
[ ] 旧far-bg ranking删除
[ ] 原MAL保留
[ ] V0复现
[ ] V1是纯defect消融
[ ] query mask依赖query feature
[ ] matched semantic target为连续值
[ ] raw top20 mean不再直接当最终概率
[ ] gate只抑制不增强
[ ] ranking围绕每个GT构造
[ ] 统一权重评估输出单一JSON
[ ] JSON含COCO完整阈值
[ ] JSON含TIDE
[ ] JSON含错误分类与混淆矩阵
[ ] JSON含ECE/LaECE
[ ] JSON含defect/query mask指标
[ ] JSON含TP vs全部FP语义AUC
[ ] JSON含candidate ranking指标
[ ] JSON含runtime和hash
[ ] 单元测试通过
[ ] 100 iteration无NaN/OOM
[ ] quick实验达到标准
[ ] 3 seed稳定
[ ] 完整数据复验完成
```
