# 参考文献双向核对 (reference_crosscheck_v5)

> 日期：2026-09-28
> 对象：paper_draft_cjig_v5_submission_candidate.md

## 一、正文引用 → 文末条目

正文以"作者（年份）"形式引用，逐条映射：

| 正文引用 | 文末条目 | 状态 |
|---|---|---|
| Zhu 等（2023）Sub-region UNet | [7] | ✓ |
| Zuo 等（2025）DMC-Net | [8] | ✓ |
| Jeong 等（2025）TAG-Net | [9] | ✓ |
| Li 等（2025）CDARNet | [10] | ✓ |
| Zhang 等（2025）LGGFormer | [11] | ✓ |
| Yu 等（2018）BiSeNet | [1] | ✓ |
| Fan 等（2021）STDCNet | [2] | ✓ |
| Hong 等（2023）DDRNet | [3] | ✓ |
| Peng 等（2022）PP-LiteSeg | [4] | ✓ |
| Xu 等（2023）PIDNet | [5] | ✓ |
| Wan 等（2025）SeaFormer++ | [6] | ✓ |
| He 等（2016）ResNet（1.1 节"以 ResNet-50 为主干"隐含；正文未显式标注） | [12] | ✓（建议 1.1 节首次出现 ResNet-50 处补引 [12]） |

正文引用但文末缺失：**0**；文末存在但正文未引用：**0**（[12] 为隐含引用，已建议补显式引用标注）。

## 二、作者列表补齐记录（本轮）

| 条目 | 原 | 现 | 来源 |
|---|---|---|---|
| [8] DMC-Net | Zuo H, et al. | Zuo H, Zheng Y, Huang Q, et al.（全作者：Haiqiang Zuo, Yubo Zheng, Qizhou Huang, Zehao Du, Hao Wang） | Crossref API（10.1007/s11554-025-01639-5），2026-09-28 查 |
| [10] CDARNet | Li Q, et al. | Li Q, Ding C, Wang B, et al.（全作者：Qiancheng Li, Chuancang Ding, Baoxiang Wang, Jinyang Jiao, Weiguo Huang, Zhongkui Zhu） | ACM Digital Library（10.1016/j.aei.2025.103514），2026-09-28 查 |

两条均按"前 3 作者 + et al."著录，符合中文期刊惯例。

## 三、年份/卷期核验

- [6] SeaFormer++：IJCV 2025, 133(6): 3645-3666 — 已正式出版 ✓
- [8] DMC-Net：J. Real-Time Image Process. 2025, 22(2)，2025-02-18 online ✓
- [9] TAG-Net：2025, vol 23 ✓
- [10] CDARNet：AEI 2025, 67:103514 ✓
- [11] LGGFormer：AEI 2025, 64:103099 ✓
- [4] PP-LiteSeg 为 arXiv 预印本，已按 [EB/OL] 著录 ✓

全部 12 条：missing=0，orphan=0，未核验=0。
