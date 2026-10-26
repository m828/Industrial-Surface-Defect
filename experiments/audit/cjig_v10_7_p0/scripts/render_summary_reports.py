#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(value):
    return f"{100 * value:.3f}%"


def write_new(path, text):
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", default="/workspace/Industrial Surface Defect/Industrial-Surface-Defect/experiments/audit/cjig_v10_7_p0")
    args = parser.parse_args()
    root = Path(args.audit_root)
    final = root / "cap_resolution"
    key = load(final / "key_findings.json")
    near = load(final / "near_duplicate_summary.json")
    families = load(final / "source_family_summary.json")
    verification = load(final / "independent_verification_report.json")
    retrieval_verification = load(final / "retrieval_screen_verification.json")
    assert verification["status"] == "PASS"
    assert retrieval_verification["status"] == "PASS"
    assert near["screen_complete_after_declared_cap_resolution"] is True

    levels = ["digital_identity", "transform_equivalence", "high_confidence_candidate"]
    labels = {"digital_identity": "数字图像身份", "transform_equivalence": "精确图像 D4", "high_confidence_candidate": "高置信同源候选"}
    family_rows = []
    for level in levels:
        f = families[level]
        family_rows.append(f"| {labels[level]} | {f['components_total_including_singletons']} | {f['nonsingleton_components']} | {f['nonsingletons_with_train']} | {f['nonsingletons_with_test']} | {f['cross_split_nonsingletons']} | {f['train_records_in_nonsingletons']} | {f['test_records_in_nonsingletons']} |")
    priority_rows = []
    for c in families["priority_components"]:
        priority_rows.append(f"| `{c['group_id']}` | {c['size']} | {c['train_records']} | {c['test_records']} | {'是' if c['contains_all_eight_priority_records'] else '否'} |")
    exact_cross = key["cross_split_exact_image_pairs"]
    gt_groups = key["gt_d4_group_evidence"]
    priority = key["priority_six_rotation_pairs"]
    source_report = f"""# P0-4 分级样本来源图

## 证据等级

节点一律为 `(split, sample_id)`。数字身份、精确图像 D4 和高置信候选分别建图；GT-only 关系不会单独合并图像家族。低阈值或未复核候选不参与传递合并。

| 层级 | 总组件（含单例） | 非单例组件 | 含 train | 含 test | 跨 split | train 记录 | test 记录 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(family_rows)}

高置信候选层只表示回顾性阈值下的 `HIGH_CONFIDENCE_SHARED_SOURCE_CANDIDATE`，不能改写成同等数量的已证实泄漏或采集家族。候选覆盖比例见 `cap_resolution/source_family_summary.json`，不得当作泄漏比例。

## 采集证明层

- 显式原始采集 ID：{families['acquisition_proven']['explicit_acquisition_ids_found']}。
- 有直接采集证明的多记录家族：{families['acquisition_proven']['nonsingleton_components_with_direct_acquisition_proof']}。
- 真实采集家族总数：**无法确定（null）**，不是零。所有采集级来源仍为 `SOURCE_UNRESOLVED`。

## 八个重点记录

| 高置信候选组件 | 记录数 | train | test | 包含全部八条？ |
|---|---:|---:|---:|---|
{chr(10).join(priority_rows)}

四个原始同图对在数字身份层各自形成二人组，并不组成一个八人数字身份家族。六个 train 记录与两个 test 记录在高置信候选层也没有合成一个八人家族；不存在作者确认的采集证明家族。

最初六组跨 split 旋转关系均为 GT 精确对齐，RGB Pearson 范围 {priority['rgb_pearson_range'][0]:.9f}–{priority['rgb_pearson_range'][1]:.9f}，MAE 范围 {priority['rgb_mae_range'][0]:.6f}–{priority['rgb_mae_range'][1]:.6f}，SSIM 范围 {priority['gray_ssim_range'][0]:.9f}–{priority['gray_ssim_range'][1]:.9f}。它们全部属于 `{'; '.join(priority['categories'])}`；由于 SSIM 低于 0.95，没有一对进入冻结的高置信图像候选层。

## 新增精确跨 split 图像关系

`test:000717` --r180--> `train:004097`：图像逐像素完全相同；同方向 GT 差异 {exact_cross[0]['gt_diff_pixels']} 像素，前景 IoU={float(exact_cross[0]['gt_foreground_iou']):.12f}。这是精确的图像变换等价，不是相同 GT 版本。

## GT 规范组与图像证据

- GT D4 非单元素规范组：{gt_groups['nonsingleton_groups']}。
- 其中有图像 D4 精确边的组：{gt_groups['groups_with_image_d4_pair']}；有高置信图像边的组：{gt_groups['groups_with_high_confidence_image_pair']}。
- 没有图像 D4 或高置信图像证据的 GT 组内记录对：{gt_groups['pairs_without_image_d4_or_high_confidence_evidence']}。

`gt_d4_canonical_groups.csv` 保存每组 train/test 数量、成员和图像证据计数。该表只用于审查，不能将 GT 规范组直接命名为采集家族。

## 独立复核

`independent_verification_report.json = PASS`：源文件/像素哈希、D4 规范哈希与方向、全部候选精核指标、来源边与组件、公开内容扫描均已复核。训练、推理和新统计评价均为 NOT RUN。
"""
    write_new(root / "source_family_graph_summary.md", source_report)

    exact = key["exact_identity"]
    d4 = key["d4"]["relations"]
    near_key = key["near_duplicate_screen"]
    exhaustive = near_key["exhaustive_screen"]
    cap = near_key["cap_resolution"]
    split_report = f"""# 训练/测试划分独立性审计

## 最终判定

**CROSS-SPLIT SOURCE INDEPENDENCE = COMPROMISED。** 至少 `test:000717` 与 `train:004097` 已满足精确图像 D4 等价，不能再把当前划分称为独立图像划分。该判定不依赖任何低阈值近重复候选，也不等于已经恢复全部采集家族。

## 定量证据

- 原始图像字节/像素重复：共 {sum(exact['image_file_duplicate_pairs'].values())} 对（train/train {exact['image_file_duplicate_pairs']['train/train']}、test/test {exact['image_file_duplicate_pairs']['test/test']}、跨 split {exact['image_file_duplicate_pairs']['train/test']}）。GT 未变换重复为 0 对。
- 图像 D4：共 {sum(d4['IMAGE_D4_EXACT']['pairs_including_identity_by_scope'].values())} 对，其中非恒等变换 {sum(d4['IMAGE_D4_EXACT']['pairs_with_nonidentity_transform_by_scope'].values())} 对；跨 split 非恒等关系为 {d4['IMAGE_D4_EXACT']['pairs_with_nonidentity_transform_by_scope']['train/test']} 对。
- GT D4：共 {sum(d4['GT_D4_EXACT']['pairs_including_identity_by_scope'].values())} 对（train/train {d4['GT_D4_EXACT']['pairs_including_identity_by_scope']['train/train']}、test/test {d4['GT_D4_EXACT']['pairs_including_identity_by_scope']['test/test']}、跨 split {d4['GT_D4_EXACT']['pairs_including_identity_by_scope']['train/test']}），{key['gt_d4_group_evidence']['nonsingleton_groups']} 个规范组。GT 关系不能单独证明原图来源。
- 同名 ID：{key['same_sample_id_across_splits']['ids']} 个；图像原字节相同 {key['same_sample_id_across_splits']['image_file_equal']}，解码像素相同 {key['same_sample_id_across_splits']['image_pixel_equal']}。同名不是身份键。
- 全对预筛：{exhaustive['all_unordered_nonself_pairs']} 个非自配对，image/GT 哈希预筛并集 {exhaustive['union_hamming_prefilter_pairs']}；正式候选 {near_key['candidate_pairs']}，两遍全部精核 {near_key['fully_reviewed_pairs']}，剩余 CAP= {near_key['remaining_cap_not_reviewed']}。
- 高置信跨 split 候选：{near_key['high_confidence_cross_split_pairs']} 对，涉及 {near_key['high_confidence_cross_split_test_record_count']} 个 test 记录；超出最初 000301/000305 的其他高置信 test 记录为 {near_key['other_high_confidence_test_records_beyond_000301_000305_count']} 个。它们只是候选敏感性范围，不是已证实泄漏数量。

## 独立性解释

1. **数字记录独立**：否，test 内已有同图不同 GT 版本。
2. **train/test 图像变换独立**：否，至少一对精确 r180 等价。
3. **全部原始采集独立**：无法完整枚举，因为没有采集 ID；候选图不能替代采集证明。
4. **840 条测试记录 = 840 独立采集图像**：不成立。
5. **GT 规范组 = 采集家族**：不成立，必须联合图像证据。

## 对评价和后续工作的约束

- 冻结 mIoU 仍可作为固定清单内部结果，但不能作为独立采集泛化证明。
- 原 Bootstrap 只是固定权重下 840 条记录的重抽样不确定性；来源簇 Bootstrap 尚未执行。
- 删除若干 test 记录不能证明消除训练曝光风险；只读敏感性分析不等于独立评价，本轮 NOT RUN。
- 新训练若用于独立泛化，需要先取得采集家族级划分或独立外部数据；当前候选图不能直接保证这种划分。

## 复核与限制

第一遍精核 {cap['first_pass_fully_reviewed']} 对，第二遍既定 backlog {cap['second_pass_fully_reviewed']} 对；无新增 pair ID、无剩余未复核候选、无阈值变更。`independent_verification_report.json=PASS`。近重复检索仍是既定描述子和阈值下的回顾性审计，不能证明不存在其他未知变换或采集关系。
"""
    write_new(root / "split_independence_audit.md", split_report)

    method_report = f"""# 近重复检测实现与覆盖率审计

## 冻结规则与范围

本审计在发现异常后回顾性选择阈值，不是盲法预注册。RGB 高相似要求 Pearson≥0.990、MAE≤6、PSNR≥24dB、SSIM≥0.95；图像近似层为 Pearson≥0.950、MAE≤12、SSIM≥0.80。GT 近似要求前景 IoU≥0.95 且像素一致率≥0.995。SSIM 参数固定为 7×7、data_range=255、非高斯窗口、样本协方差、K1=0.01/K2=0.03。

先对 {exhaustive['all_unordered_nonself_pairs']} 个非自配对计算八方向 pHash/dHash 最小距离，再执行缩略图闸门。预筛并集 {exhaustive['union_hamming_prefilter_pairs']} 对，{exhaustive['prefilter_pairs_rejected_by_thumbnail_gates']} 对被缩略图规则拒绝，{exhaustive['unique_candidate_pairs_after_exact_seeds']} 对进入正式候选。预筛拒绝不是候选，也不代表来源无关。

## 实现缺陷与纠正

- 原 LSH 版本产生 503436 行但只有 500000 个逻辑 pair；3436 个反向重复来自未排序的 D4 种子索引。原汇总及下游计数标记为非权威诊断稿，精确文件/像素/D4 结果不受影响。
- 两次单纯扩大 LSH 容量不能解决桶碰撞，相关尝试未形成结论。随后改为全对 Hamming 预筛并修正索引顺序。
- 第一遍容量精核 {cap['first_pass_fully_reviewed']} 对；第二遍只复核既定 CAP backlog {cap['second_pass_fully_reviewed']} 对。最终 {near_key['fully_reviewed_pairs']} 对全部精核，阈值未变，剩余未复核为 {near_key['remaining_cap_not_reviewed']}。

## 已复核分类

| scope | 类别 | 对数 |
|---|---|---:|
""" + "\n".join(f"| {r['scope']} | `{r['category']}` | {r['pairs']} |" for r in near_key['category_counts']) + f"""

只有 FULLY_REVIEWED 且达到全部高相似条件的记录对才进入高置信候选图。`IMAGE_NEAR_ONLY` 和 GT-only 相似均保留为 `SOURCE_UNRESOLVED`。原始字节、解码像素、D4 图像精确、GT 精确+高图像相似、仅图像近似、仅 GT 近似分别记录，不以单一 Pearson 判定泄漏。

## 验证

四个已知同图对、六组旋转 GT 正例和合成 JPEG+旋转正例均通过。独立脚本重新计算全部源文件/像素哈希、D4 规范哈希、全部 {near_key['fully_reviewed_pairs']} 个精核指标及来源组件，`independent_verification_report.json = PASS`。另一脚本独立重算全对八方向 Hamming、缩略图闸门、候选集合、描述子和方向，`retrieval_screen_verification.json = PASS`。新增复核器的首次最小值比较逻辑失败已保留并纠正，未修改原数据或实验结果。没有训练、模型推理、新 Bootstrap 或新模型指标评价。
"""
    write_new(root / "near_duplicate_implementation_audit.md", method_report)
    print(json.dumps({"status": "COMPLETED", "reports": ["source_family_graph_summary.md", "split_independence_audit.md", "near_duplicate_implementation_audit.md"]}))


if __name__ == "__main__":
    main()
