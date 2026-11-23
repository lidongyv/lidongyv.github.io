# 2025–2026 五会议 3D Gaussian Splatting 论文系统调研总报告

> 生成时间：2026-11-09 22:52 ｜ 范围：CVPR 2025/2026、ICCV 2025、ECCV 2026、NeurIPS 2025/2026、AAAI 2025/2026
> 逐篇清单见三个分报告：[CVPR](gs_survey_cvpr.md) ｜ [ICCV/ECCV](gs_survey_iccv_eccv.md) ｜ [NeurIPS/AAAI](gs_survey_neurips_aaai.md)

## TL;DR

- 共枚举 **870 篇** 3DGS 相关论文（NeurIPS 2026 未公开不计）。
- **per-scene 优化仍是最大盘（208 篇，24%）**——前馈范式没有取代逐场景优化，两者并行共存。
- **动态/4D 是增长最快的主题（138 篇，16%）**，CVPR 2026 单届 34 篇。
- **前馈重建逐年扩容**（CVPR25 10 → ECCV26 18），Splat3R 系与 LRM 系（DrivingRecon、DGS-LRM）两条子线。
- **压缩与存储增速最快**（CVPR25 8 → ECCV26 10 + NeurIPS25 4 + AAAI26 5），移动端落地（Seele）与 LoD（LODGE、FilterGS）成热点。
- 新交叉点：MLLM × 3DGS（MLLMSplat、REALM）、世界模型 × 驾驶（SplatAD、GaussianDWM、RemedyGS）。

## 一、各会议篇数

| 会议 | 全会论文数 | 判定 3DGS | 占比 |
|---|---|---|---|
| CVPR 2025 | 2871 | 169 | 5.9% |
| CVPR 2026 | 4042 | 174 | 4.3% |
| ICCV 2025 | 2701 | 163 | 6.0% |
| ECCV 2026 | 2834 | 150 | 5.3% |
| NeurIPS 2025 | —（OpenReview 主会枚举） | 72（主会 70 + D&B 2） | — |
| NeurIPS 2026 | 未公开（预计 2026-12 会议期间公开） | 0 | — |
| AAAI 2025 | —（OJS 检索枚举） | 40 | — |
| AAAI 2026 | —（OJS 检索枚举） | 102 | — |
| **合计** | | **870** | |

## 二、分类汇总（10 类 × 7 届）

| 类别 | CVPR25 | CVPR26 | ICCV25 | ECCV26 | NeurIPS25 | AAAI25 | AAAI26 | 合计 | 占比 |
|---|---|---|---|---|---|---|---|---|---|
| per-scene 优化 | 33 | 35 | 48 | 41 | 16 | 12 | 23 | **208** | 23.9% |
| 动态/4D | 27 | 34 | 20 | 20 | 14 | 5 | 18 | **138** | 15.9% |
| 其他 | 12 | 24 | 21 | 16 | 8 | 6 | 18 | **105** | 12.1% |
| 语义/分割/理解 | 17 | 17 | 21 | 9 | 3 | 5 | 9 | **81** | 9.3% |
| 前馈重建 | 10 | 14 | 11 | 18 | 6 | 3 | 9 | **71** | 8.2% |
| 编辑与重光照 | 17 | 9 | 13 | 11 | 5 | 1 | 6 | **62** | 7.1% |
| 渲染改进 | 21 | 12 | 5 | 6 | 8 | 1 | 6 | **59** | 6.8% |
| 单图/少图生成 | 15 | 7 | 12 | 8 | 3 | 6 | 5 | **56** | 6.4% |
| SLAM/定位建图 | 9 | 14 | 9 | 11 | 3 | 0 | 3 | **49** | 5.6% |
| 压缩与存储 | 8 | 8 | 3 | 10 | 4 | 1 | 5 | **39** | 4.5% |
| 合计 | 169 | 174 | 163 | 150 | 72 | 40 | 102 | **870** | |

## 三、趋势解读（依据篇数变化）

1. **前馈重建没有取代 per-scene 优化**：per-scene 优化两届 CVPR 都是最大类（33→35），ICCV/ECCV 更高（48/41）；前馈重建仅占 8%。前馈的价值在秒级初始化和可泛化性，质量上限仍靠逐场景微调。
2. **动态/4D 是 2026 年的主旋律**：CVPR26 34 篇、ECCV26 20 篇、NeurIPS25 14 篇；代表 Minute4D、Instant4D、PhysGM（前馈 4D 物理）、GaussianDWM（驾驶世界模型）。
3. **前馈重建内部两条子线**：Splat3R 系（无位姿 3D 高斯重建，ECCV26 扩容）与 LRM 系（DrivingRecon、DGS-LRM、ZPressor）；去位姿化继续（SelfSplat CVPR25）。
4. **压缩与移动端落地加速**：LODGE（NeurIPS25 spotlight，LoD 大规模渲染）、FilterGS、KISS-GS、OMG（最小高斯压缩）、PCGS（渐进压缩）、Seele（移动端加速）。
5. **SLAM 稳定上升**：CVPR 9→14，ECCV 11；代表 WildGS-SLAM、Triangle Splatting SLAM、4D Gaussian Splatting SLAM。
6. **单图/少图生成热度回落**：CVPR25 15 → CVPR26 7 → ECCV26 8，生成侧资源转向 4D 与世界模型。
7. **MLLM × 3DGS 成为新交叉点**：MLLMSplat、REALM（MLLM 智能体编辑）、LangSplatV2（450+FPS 语言场）、Segment Any 3D Gaussians。
8. **自动驾驶/安全向**：SplatAD（AD 仿真）、DeSiRe-GS（动态街景）、RemedyGS（安全）。

## 四、代表论文索引（按类别精选）

- 前馈重建：MonoSplat（CVPR25）、SelfSplat（CVPR25）、DrivingRecon（NeurIPS25）、DGS-LRM（NeurIPS25）、LongSplat（AAAI26）、GaussianGPT（ECCV26，自回归 3D 场景生成）
- per-scene 优化：DashGaussian（CVPR25，200 秒）、FastGS（CVPR26，100 秒）、3DGS-LM（ICCV25）、LODGE（NeurIPS25 spotlight）
- 动态/4D：DeSiRe-GS（CVPR25）、PhysGM（CVPR26）、GaussianDWM（CVPR26）、Minute4D（ECCV26）、Instant4D（NeurIPS25）
- 压缩与存储：4DGC（CVPR25）、FilterGS（CVPR26）、KISS-GS（ECCV26）、OMG（NeurIPS25）、PCGS（AAAI26）、Seele（CVPR26）
- SLAM：WildGS-SLAM（CVPR25）、Triangle Splatting SLAM（ECCV26）
- 语义/理解：MLLMSplat（CVPR26）、LangSplatV2（NeurIPS25）、Segment Any 3D Gaussians（AAAI25）
- 编辑/生成：EditSplat（CVPR25）、Scene Splatter（CVPR25）、GaussianZoom（CVPR26）、PixGS（ECCV26）
- 渲染改进：FlashGS（CVPR25）、TreeSplat（NeurIPS25）、InstGS（ECCV26）
- 世界模型/安全：SplatAD（CVPR25）、REALM（CVPR26）、RemedyGS（CVPR26）

## 五、覆盖方法与缺口（审计记录）

- 数据源：CVF OpenAccess（CVPR/ICCV/ECCV，day=all 全量实抓，CVPR 逐天页交叉一致）；eccv.ecva.net（ECCV 2026，CVF 未上线）；OpenReview 官方 API（NeurIPS 2025）；ojs.aaai.org（AAAI）。
- 筛选口径：标题含 gaussian/splat（+3dgs 等缩写补录），再人工判定 3DGS 主题；排除高斯过程/噪声/统计等误报共 84 篇（名单在各分报告附录）。
- 链接：CVF/官方源链接 100% 验证；arXiv 版仅验证 13 篇代表论文；TranSplat(AAAI-25) 因同名干扰标"链接待补"。
- 已知缺口：① NeurIPS 2026 未公开；② ECCV 2026 官方列表标注 preliminary（出版社终审可能微调）；③ ICCV 2025 逐天页接口 500（以 day=all 为准）；④ NeurIPS 2025 未含 Workshops；⑤ NeurIPS/AAAI 以标题/摘要关键词枚举，纯缩写且无关键词的 3DGS 论文存在漏检风险；⑥ dblp/papers.nips.cc/Semantic Scholar 本环境不可达，已用官方替代源。

更新时间: 2026-11-09 22:52
