# Lidong Yu 论文元数据与链接整理（逐条核验版）

> 核验时间：2026-11-23（Asia/Shanghai）。
> 核验方式：实际访问 CVF Open Access 论文页、arXiv abs 页、GitHub API / raw README、OpenAlex 与 Crossref 检索；本地 PDF 用文本提取核对标题与匿名状态。
> 判定口径：能查到官方论文页 / arXiv abs / DOI 的记为「已发表」；只在其个人主页挂 PDF、查不到任何录用记录的记为「未能确证（投稿稿 / 未发表）」。
> 一句话结论：第 1–8 篇全部可核验（3 篇 ICCV、1 篇 NeurIPS、1 篇 CVPR、1 篇 AAAI、2 篇 arXiv）；第 9–12 项是旧主页上的**投稿稿与未发表手稿**，不是已发表论文，**不要写进「发表论文」列表**。

---

## 1. 3D Visual Illusion Depth Estimation

- **英文标题**：3D Visual Illusion Depth Estimation
- **中文标题**：3D 视错觉深度估计
- **作者**：Chengtang Yao, Zhidan Liu, Jiaxi Zeng, **Lidong Yu（第 4 作者 / 共 6 人）**, Yuwei Wu, Yunde Jia
- **会议/期刊 + 年份**：NeurIPS 2025（arXiv abs 页 Comments 字段原文 "NeurIPS 2025"）
- **论文链接**：[arXiv:2505.13061](https://arxiv.org/abs/2505.13061)
- **代码链接**：[YaoChengTang/3D-Visual-Illusion-Depth-Estimation](https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation)（仓库描述即 "[NeurIPS] 3D Visual Illusion Depth Estimation"）
- **研究问题**：3D 视错觉（把平面伪装成三维的绘画、屏幕、镜面等）会欺骗人类视觉，那么它是否同样欺骗机器深度估计？单目、双目、多视图方法在幻视场景下会错到什么程度，能否用视觉语言模型的常识来纠偏？
- **方法核心**：构建约 3k 场景、20 万张图像的 3D-Visual-Illusion 数据集（生成数据 + Intel RealSense L515 实采双目数据，SAM + 人工标注幻视区域掩码）；提出「VLM 常识引导的单目-双目融合」框架，用视觉语言模型判断何处该信单目、何处该信视差，自适应融合两路深度。
- **主要结果**：SOTA 单目模型在真实幻视区域大幅失真（DepthPro(+align) bad3 = 34.98%、AbsRel = 0.09；Marigold bad3 = 59.67%），双目模型同样被镜面/透明幻视欺骗（RAFT-Stereo bad3 = 13.20%、EPE = 1.62）；该融合方法在幻视区域 EPE 1.77 / bad5 3.60，与最强双目基线相当，同时在常规场景 Middlebury zero-shot 上取得最低 EPE 1.50（RAFT-Stereo 1.92、Selective-RAFT 2.34、StereoAnything 2.89）。
- **为什么值得关注**：一个人同时交付了「新数据集 + 新评测 + 新方法 + 开源代码与权重」，并且明确指出基础模型（单目深度/VGGT/Dust3R）在安全相关场景的系统性失效——这正是自动驾驶感知负责人最看重的能力。

---

## 2. Diving into the Fusion of Monocular Priors for Generalized Stereo Matching

- **英文标题**：Diving into the Fusion of Monocular Priors for Generalized Stereo Matching
- **中文标题**：深入单目先验的融合：面向泛化立体匹配
- **作者**：Chengtang Yao, **Lidong Yu（第 2 作者 / 共 6 人）**, Zhidan Liu, Jiaxi Zeng, Yuwei Wu, Yunde Jia
- **会议/期刊 + 年份**：ICCV 2025 **Oral**（CVF 页码 pp. 14887–14897；arXiv journal-ref 明确写 "ICCV 2025 Oral"）
- **论文链接**：[arXiv:2505.14414](https://arxiv.org/abs/2505.14414) ｜ [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_Diving_into_the_Fusion_of_Monocular_Priors_for_Generalized_Stereo_ICCV_2025_paper.html)
- **代码链接**：[YaoChengTang/Diving-into-the-Fusion-of-Monocular-Priors-for-Generalized-Stereo-Matching](https://github.com/YaoChengTang/Diving-into-the-Fusion-of-Monocular-Priors-for-Generalized-Stereo-Matching)
- **研究问题**：把视觉基础模型（VFM，如 DepthAnything V2）的无偏单目先验融进立体匹配能提升遮挡、非朗伯面等病态区域的泛化，但直接融合存在三个硬问题：仿射不变的相对深度与视差的绝对深度不对齐；迭代更新中对视差过自信导致局部最优；前几轮迭代的噪声视差会误导融合。
- **方法核心**：用**二值局部排序图（binary local ordering map）**把深度图转成二值相对表示，统一相对/绝对深度；同一排序图用来重加权初始视差更新，抑制局部最优与早期噪声；把单目深度到视差的最终直接融合建模为**配准问题**，用逐像素线性回归模块做全局自适应对齐。
- **主要结果**：从 SceneFlow 泛化到 Middlebury / ETH3D 误差降低一半以上、Booster 上 Bad-x.0 提升约 10 个点，且不使用额外双目数据或数据增广；效率几乎不降。DrivingStereo 雨天 EPE 1.29（RAFT-Stereo 1.80、IGEV 2.32、Selective-IGEV 2.22）。
- **为什么值得关注**：ICCV 2025 Oral，且是「基础模型先验如何真正落地到几何任务」的范本——给出了可复现的训练/评测/可视化全链路（含对 DepthAnything V2 的改造），对做端到端感知的团队直接可迁移。

---

## 3. Sparse Point Guided 3D Lane Detection

- **英文标题**：Sparse Point Guided 3D Lane Detection
- **中文标题**：稀疏点引导的 3D 车道检测
- **作者**：Chengtang Yao, **Lidong Yu（第 2 作者 / 共 4 人）**, Yuwei Wu, Yunde Jia
- **会议/期刊 + 年份**：ICCV 2023（CVF 页码 pp. 8363–8372）
- **论文链接**：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Sparse_Point_Guided_3D_Lane_Detection_ICCV_2023_paper.html)（**无 arXiv 版本**：arXiv 标题检索 totalResults = 0）
- **代码链接**：[YaoChengTang/Sparse-Point-Guided-3D-Lane-Detection](https://github.com/YaoChengTang/Sparse-Point-Guided-3D-Lane-Detection)
- **研究问题**：3D 车道检测通常建立前视图与 BEV 的稠密对应，但车道只占极小比例，绝大部分对应是冗余背景——稠密高分辨率对应因此成为算力瓶颈。
- **方法核心**：粗到细两阶段。粗阶段在极低分辨率下建立「稠密但便宜」的前视图–BEV 对应，得到粗车道；细阶段沿粗车道采样**稀疏点**，从高分辨率前视图特征图上取局部特征，在 BEV 空间由低到高分层次精化 3D 车道。
- **主要结果**：OpenLane 验证集 F1 = 52.3%（PersFormer 50.5%）；分场景统计整体 F1 53.7、较当时 SOTA 高 3.2 分，困难场景最高 +6.2 分；同时显存几乎减半、推理速度约 2 倍（作者摘要口径：整体 F1 +3 分、困难场景 +6 分）。
- **为什么值得关注**：这是「用稀疏性换算力」的典型工业级设计——直接对应量产车上车道/道路结构感知的算力预算约束，且代码开源、指标口径清晰。

---

## 4. Parameterized Cost Volume for Stereo Matching

- **英文标题**：Parameterized Cost Volume for Stereo Matching
- **中文标题**：参数化代价体立体匹配
- **作者**：Jiaxi Zeng, Chengtang Yao, **Lidong Yu（第 3 作者 / 共 5 人）**, Yuwei Wu, Yunde Jia
- **会议/期刊 + 年份**：ICCV 2023（CVF 页码 pp. 18347–18357）
- **论文链接**：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Zeng_Parameterized_Cost_Volume_for_Stereo_Matching_ICCV_2023_paper.html)（**无 arXiv 版本**：标题检索 totalResults = 0）
- **代码链接**：[JiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching](https://github.com/JiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching)（README 引用信息与论文完全一致）
- **研究问题**：大视差范围让立体匹配计算量爆炸；动态代价体只关注局部视差空间、缺乏全局视野，需要很多次迭代才能逼近真值。
- **方法核心**：发现动态代价体近似于「固定小方差的单高斯」，于是提出用**多高斯分布**参数化整个视差空间：每个像素的视差分布由权重、均值、方差表示，均值与方差用来采样视差候选并计算代价，权重与均值用来算视差输出；参数通过 JS 散度优化，实现为前馈微分模块中的梯度下降更新。
- **主要结果**：SceneFlow EPE 0.62、单帧 62 ms（RAFT-Stereo 0.65 / 254 ms，约 4 倍加速）；KITTI 2015 上较 RAFT-Stereo 快 7 倍、较 CREStereo 仅 D1-all 差 0.24% 但快 7–8 倍，Middlebury 上比 RAFT-Stereo 快近 15 倍；Booster Bad2 27.41% vs RAFT-Stereo 38.66%，time/MP 0.14 vs 1.01。
- **为什么值得关注**：把「精度–速度帕累托前沿」整体往前推了一截，是实时双目深度可以直接替换的模块化方案（代码基于 RAFT-Stereo，迁移成本低）。

---

## 5. FoggyStereo: Stereo Matching With Fog Volume Representation

- **英文标题**：FoggyStereo: Stereo Matching With Fog Volume Representation
- **中文标题**：FoggyStereo：基于雾体积表示的立体匹配
- **作者**：Chengtang Yao, **Lidong Yu（第 2 作者 / 共 2 人）**
- **会议/期刊 + 年份**：CVPR 2022（CVF 页码 pp. 13043–13052）
- **论文链接**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Yao_FoggyStereo_Stereo_Matching_With_Fog_Volume_Representation_CVPR_2022_paper.html)（**无 arXiv 版本**：标题检索 totalResults = 0）
- **代码链接**：[项目页](https://yaochengtang.github.io/FoggyStereo-Stereo-Matching-with-Fog-Volume-Representation/) ｜ [代码仓库](https://github.com/YaoChengTang/FoggyStereo-Stereo-Matching-with-Fog-Volume-Representation)（仓库内 `Code/` 目录）
- **研究问题**：雾天散射会模糊图像、让匹配产生歧义；已有方法把雾当噪声先去掉，但雾本身其实携带深度线索。
- **方法核心**：从渲染视角「反向」利用大气散射模型：对每个视差候选对应的深度，把该深度范围内的雾去掉重新渲染；渲染图越清晰说明该深度越接近真值。把不同深度渲染出的图沿通道堆叠成**雾体积（fog volume）**，与代价体融合，纠正雾导致的匹配歧义。
- **主要结果**：雾天场景较当时 SOTA 提升 10%~30%，清晰场景性能基本持平；KITTI 2012 合成雾上较 DeepPruner 提升 30%、较 PSMNet 提升约 10%；SceneFlow 雾天上相比 DeepPruner 精度提升近 3 倍。
- **为什么值得关注**：只用一对双目图像、不需要额外雾天监督，就把「恶劣天气」从减分项变成信息源——这是量产感知里典型的低成本鲁棒性思路。

---

## 6. Deep Stereo Matching With Explicit Cost Aggregation Sub-Architecture

- **英文标题**：Deep Stereo Matching With Explicit Cost Aggregation Sub-Architecture
- **中文标题**：显式代价聚合子架构的深度立体匹配
- **作者**：**Lidong Yu（第 1 作者 / 共 4 人）**, Yucheng Wang, Yuwei Wu, Yunde Jia
- **会议/期刊 + 年份**：AAAI 2018（AAAI-18，**spotlight poster**，arXiv comments 原文 "Accepted by AAAI18 as a spotlight poster"）
- **论文链接**：[arXiv:1801.04065](https://arxiv.org/abs/1801.04065) ｜ [AAAI 论文页](https://aaai.org/papers/12267-deep-stereo-matching-with-explicit-cost-aggregation-sub-architecture/)
- **代码链接**：[lidongyv/RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture](https://github.com/lidongyv/RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture)（README 明确说明是本文 AAAI18 实现，TensorFlow 0.12）
- **研究问题**：深度立体匹配的研究多集中在特征提取与相似度度量，而同样关键的**代价聚合**长期被忽视（多靠固定窗口或隐式网络）。
- **方法核心**：把代价聚合重构成「聚合提案的生成 + 选择」的学习问题，用双流子架构实现：一条流生成候选聚合结果，另一条流做选择；选择依据来自轻量卷积网络提取的低层结构信息。这样给代价聚合引入全局视野，纠正匹配代价计算阶段局部视野造成的错误。
- **主要结果**：SceneFlow 上 error>1px 8.93%（GC-Net 11.3%）、error>3px 5.62%（7.2%）、MAE 1.75 px（2.21 px）；KITTI 2015 D1-all 2.79%（GC-Net 2.87%、PSMNet 2.38%）。
- **为什么值得关注**：Lidong Yu 的**一作**论文，AAAI spotlight，是把 cost aggregation 显式学习化的早期代表工作之一——能看出他从读研阶段就在啃立体匹配里最「不性感」但最影响精度的环节。

---

## 7. Content-Aware Inter-Scale Cost Aggregation for Stereo Matching

- **英文标题**：Content-Aware Inter-Scale Cost Aggregation for Stereo Matching
- **中文标题**：内容感知的跨尺度代价聚合立体匹配
- **作者**：Chengtang Yao, Yunde Jia, Huijun Di, Yuwei Wu, **Lidong Yu（第 5 作者，末位 / 共 5 人）**
- **会议/期刊 + 年份**：arXiv 2020（arXiv:2006.03209，19 页预印本）。**注意**：个人主页存档的 PDF 是 ECCV 2020 的匿名投稿版（正文页眉 "ECCV-20 submission ID 3306"），**未见任何录用记录**，因此只能按 arXiv 预印本引用。
- **论文链接**：[arXiv:2006.03209](https://arxiv.org/abs/2006.03209)
- **代码链接**：无公开代码（未找到对应仓库）
- **研究问题**：多尺度代价聚合为了拿上下文把代价体下采样，上采样回细尺度时必然丢细节；如何在上采样过程中既恢复细节又聚合跨尺度信息？
- **方法核心**：学习**内容感知的动态滤波权重**，对细尺度代价体上每个位置自适应地从粗代价体聚合信息；并提出分解策略——先由两尺度特征图学 2D 相似度，再据此构造 3D 滤波权重，把 3D 空间-视差联合聚合拆成「1D 视差 + 2D 空间」两步，大幅降低 3D 卷积算力。
- **主要结果**：SceneFlow EPE 0.57，为当时最优（作者称约为 PSMNet 基线的两倍好）；KITTI 2015 上接近 SOTA。消融显示把 PSMNet 的双线性/反卷积上采样换成该方法，EPE 分别改善 0.48 / 0.33。
- **为什么值得关注**：把 3D 代价聚合拆成「1D 视差 + 2D 空间」的做法，对后续高效代价聚合设计有直接参考价值，也是他带学生做「小改进、可解释」课题的一个样本。

---

## 8. DRL-Based Trajectory Tracking for Motion-Related Modules in Autonomous Driving

- **英文标题**：DRL-Based Trajectory Tracking for Motion-Related Modules in Autonomous Driving
- **中文标题**：基于深度强化学习的轨迹跟踪：面向自动驾驶中的运动相关模块
- **作者**：Yinda Xu, **Lidong Yu（第 2 作者 / 共 2 人）**
- **会议/期刊 + 年份**：arXiv 2023（arXiv:2308.15991，Technical report，v3 更新于 2024-03）
- **论文链接**：[arXiv:2308.15991](https://arxiv.org/abs/2308.15991)
- **代码链接**：[MARMOTatZJU/drl-based-trajectory-tracking](https://github.com/MARMOTatZJU/drl-based-trajectory-tracking)（arXiv 注记给出，含 readthedocs 文档）
- **研究问题**：轨迹跟踪是规划、控制等运动相关模块的基础例程；传统方法对场景、动力学模型做强假设，真实场景一变就不鲁棒。
- **方法核心**：用深度强化学习直接学跟踪策略，model-free、data-driven，靠深度网络的表示能力 + RL 的探索获得鲁棒性与精度；策略可与多种动力学模型组合（自行车模型等），并可与 iLQR 后优化串联。
- **主要结果**：初始速度 5 m/s、Bicycle 模型、优化迭代 0 步时，平均跟踪误差 0.0981，对比纯跟踪 + iLQR 的 0.2555（相对降低 61.6%）；初始速度越高（20 / 30 m/s）增益越明显；对噪声鲁棒、可适配多种动力学模型。
- **为什么值得关注**：少数从感知跨到规控、并且把「RL 策略 + 经典优化器」工程化组合讲清楚的公开工作——对做端到端自动驾驶的团队是很好的参考样本。

---

> 以下 4 项来自旧主页（project2 / project3 / project4）。经核验，它们都是**投稿稿或未发表手稿**：稿件本身匿名、无 DOI、OpenAlex / Crossref 按标题检索无记录。可以确认它们属于 Lidong Yu 的研究线（挂在其个人主页、代码在其 GitHub 账号下，其中第 12 项手稿有他本人署名），但**无法确认其中任何一篇被录用**。

## 9. Explicit Context Mapping for Stereo Matching —— 未能确证（未发表）

- **英文标题**：Explicit Context Mapping for Stereo Matching（两版稿件标题一致）
- **中文标题**：显式上下文映射立体匹配
- **作者**：**未能确证**。两版 PDF 均为匿名评审稿（首页写 "Anonymous CVPR submission / Paper ID 0000"、"Anonymous ICCV submission / Paper ID 616"），稿件本身不含作者名；归属依据只有：该工作挂在其个人主页 project2、代码仓库在其 GitHub 账号下。
- **会议/期刊 + 年份**：**未能确证**。个人主页标注为 CVPR 2019（`cvpr19.pdf`）与 ICCV 2019（`iccv2019.pdf`）两个投稿版本，均为匿名评审稿；OpenAlex 标题检索 0 命中、Crossref 无记录、无 DOI → **没有任何录用证据**。
- **论文链接**：无公开论文页。站点存档：[CVPR'19 投稿版](../assets/pdf/cvpr19.pdf)、[ICCV'19 投稿版](../assets/pdf/iccv2019.pdf)
- **代码链接**：[lidongyv/Explicit-Context-Mapping-for-Stereo-Matching](https://github.com/lidongyv/Explicit-Context-Mapping-for-Stereo-Matching)
- **研究问题**：深度立体匹配为省算力在低分辨率空间匹配，再把视差图映射回高分辨率时丢细节（边缘模糊、小目标消失）；如何在上映射过程中直接保住细节？
- **方法核心**：把 LR→HR 的映射写成贝叶斯推断（LR 视差为先验、HR 视差为后验），用 HR 与 LR 特征向量的相似度构造**显式上下文映射矩阵**，代替双线性插值/反卷积做上采样；模块可直接嵌入现有立体匹配网络而不改原结构。
- **主要结果**（来自主页/稿件，未经同行评审）：SceneFlow 上相对 PSMNet 基线，映射视差图带来约 0.22 EPE 改善，映射代价体进一步提升（作者称比基线低 0.4 以上）；KITTI 因稀疏监督提升有限（约 0.3 EPE）。
- **为什么值得关注**：不建议作为「发表论文」展示；可作为「早期探索/技术储备」提及，说明其很早就关注上采样细节保持问题。

---

## 10. Interpretable Stereo Matching With Region Support —— 未能确证（未发表）

- **英文标题**：三个版本标题各不相同，需特别注意：
  - CVPR 2018 投稿版：**A Learnable Region Support for Stereo Matching**
  - ECCV 2019 投稿版：**Learning-based Region Support for Stereo Matching**
  - 主页条目名（非稿件标题）：Interpretable stereo matching with region support
- **中文标题**：可解释的立体匹配：区域支撑（region support）
- **作者**：**未能确证**。三份 PDF 全部是匿名评审稿（"Anonymous CVPR submission" / "Anonymous ECCV submission"），稿件内无作者名；归属依据为个人主页 project3 + 其 GitHub 代码仓库。
- **会议/期刊 + 年份**：**未能确证**。仅存在 CVPR 2018、ECCV 2019 两个**投稿版**与一份 TPAMI 格式的扩展手稿，均无录用/出版记录（OpenAlex、Crossref 标题检索均无对应条目）。
- **论文链接**：无公开论文页。站点存档：[CVPR'18 投稿版](../assets/pdf/interpreatble%20cvpr18.pdf)、[ECCV'19 投稿版](../assets/pdf/interpretable%20eccv19.pdf)、[TPAMI 扩展稿](../assets/pdf/interpretable%20tpami.pdf)
- **代码链接**：[lidongyv/Interpretable-stereo-matching-with-region-support](https://github.com/lidongyv/Interpretable-stereo-matching-with-region-support)
- **研究问题**：深度立体匹配大多用网络「隐式」处理遮挡、弱纹理等难点，难以针对性解决；能否用显式的「区域支撑」把传统可变支撑窗的思想学出来？
- **方法核心**：用单目深度推断网络生成**区域支撑（region support）**——由深度相近像素构成的区域分割，作为代价聚合与视差精化的引导；并配套改进代价计算、聚合与 refinement，使遮挡/弱纹理区域的错误可被显式解释和处理。
- **主要结果**：稿件声称实验验证了区域支撑对遮挡、弱纹理区域的有效性（未发表、数字未核，**不建议引用具体数值**）。
- **为什么值得关注**：不建议作为发表成果展示；价值在于「用可解释结构约束深度网络」的研究品味，可作为面试中的方法论谈资。

---

## 11. Region-Support Depth Inference From a Single Image —— 未能确证（未发表）

> 注意：这一项常被误记为「Monocular Depth Estimation with Region Support（CVPR 2019）」；稿件真实标题是 **Region-Support Depth Inference From a Single Image**。

- **英文标题**：**Region-Support Depth Inference From a Single Image**
- **中文标题**：基于区域支撑的单图像深度推断
- **作者**：**未能确证**。NeurIPS 投稿版与 CVPR 2019 投稿版均为匿名稿（"Anonymous Author(s)" / "Anonymous CVPR submission"），稿件内无作者名；归属依据为个人主页 project4 + 其 GitHub 仓库（仓库名含 `-cvpr`）。
- **会议/期刊 + 年份**：**未能确证**。仅见 NeurIPS 与 CVPR 2019 两个**投稿版**，无录用记录、无 DOI。
- **论文链接**：无公开论文页。站点存档：[NeurIPS 投稿版](../assets/pdf/nips.pdf)、[CVPR'19 投稿版](../assets/pdf/nips-cvpr19.pdf)
- **代码链接**：[lidongyv/Monocular-depth-esitimation-with-region-support-cvpr](https://github.com/lidongyv/Monocular-depth-esitimation-with-region-support-cvpr)（仓库名 "esitimation" 为原作者拼写错误）
- **研究问题**：单目深度推断本身病态（尺度、物体外观都有歧义）；引入语义分割能消解部分歧义，却在「语义标签 ↔ 深度值」之间引入新的歧义。
- **方法核心**：用**区域支撑**作为推断引导，设计区域支撑网络：生成模块用金字塔单元从 RGB 图确定区域支撑，精化模块把区域支撑与 RGB 拼接成推理引导，以简化金字塔单元做迭代式由粗到细的深度精化。
- **主要结果**：稿件声称优于当时的单目深度推断方法（未发表、数字未核，**不建议引用具体数值**）。
- **为什么值得关注**：不建议作为发表成果展示；它与第 10 项构成同一条「region support」研究线，说明其在读研期间持续追一个想法。

---

## 12. TPAMI 扩展版 —— 未能确证（未发表手稿）

- **英文标题**：**Deep Stereo Matching With Regularization of Pixel-Set Knowledge**（这是主页标为 region-support 工作 "Further Extension" 的 TPAMI 手稿的真实标题）
- **中文标题**：像素集知识正则化的深度立体匹配
- **作者**：**Lidong Yu（唯一作者）**——这是四份旧稿中唯一带真实署名的一份，可作为归属的间接证据。
- **会议/期刊 + 年份**：**未能确证**。TPAMI 双栏排版手稿、仅 6 页、无 DOI、无卷期页码；OpenAlex / Crossref 检索 "pixel-set knowledge" 均 0 命中 → 没有证据表明被 TPAMI 接收。
- **论文链接**：无公开论文页。站点存档：[TPAMI 手稿](../assets/pdf/interpretable%20tpami.pdf)
- **代码链接**：无公开代码
- **研究问题**：深度立体匹配有两个顽疾——感受野过大削弱相似对应之间的区分度导致细节丢失；以及在无判别力像素上做冗余计算导致算力浪费。
- **方法核心**：提出**像素集知识（pixel-set knowledge）**作为正则化：把像素组织成具有相同代价/视差的判别性像素集与代表性像素集等四类先验，只对代表性像素做匹配、其余像素由像素集内共享的代价与视差约束，从而同时保细节、降冗余计算。
- **主要结果**：手稿声称在保持精度的同时显著降低冗余代价计算（未发表，**不建议引用具体数值**）。
- **为什么值得关注**：不建议作为发表成果；若要在简历中体现「研究连续性」，可写成 "unpublished manuscript / TPAMI submission"，不要写成 TPAMI 论文。

---

## 核验记录与限制（透明说明）

**逐条实际打开并核对过的链接**

- arXiv abs：2505.13061、2505.14414、2006.03209、2308.15991、1801.04065（标题、作者、comments/journal-ref 全部一致）。
- CVF Open Access：ICCV 2025（Diving，pp. 14887–14897）、ICCV 2023（Sparse Point，pp. 8363–8372）、ICCV 2023（Parameterized，pp. 18347–18357）、CVPR 2022（FoggyStereo，pp. 13043–13052）——作者顺序与页码均取自页面。
- arXiv 存在性核查：`Sparse Point Guided 3D Lane Detection`、`Parameterized Cost Volume`、`FoggyStereo` 标题检索 totalResults 均为 0 → 确认这三篇**没有 arXiv 版本**，故用 CVF 官方页作论文链接。
- GitHub：4 个仓库通过 API 读取确认存在（3D-Visual-Illusion…、Diving…、Sparse-Point…、FoggyStereo…），另外 5 个（RAN-AAAI18、JiaxiZeng/Parameterized、Explicit-Context-Mapping、Interpretable-region-support、Monocular-depth…）通过 `topics` 端点返回 200 确认存在；AAAI18 与 Parameterized 的 README 已读，内容与论文对应。
- AAAI 官方页（aaai.org/papers/12267-…）HTTP 200，页面标题与论文一致。

**未能核验 / 刻意未用的来源**

- Google Scholar 个人页（`citations?user=cYLbx3AAAAAJ`）在当前网络环境抓取失败（fetch failed / 被拒），**没有**用它做交叉核对；本文件里的作者顺序全部来自 arXiv / CVF 官方页。
- DBLP 被反爬页（Anubis）拦截；Semantic Scholar API 返回 429。改用 OpenAlex + Crossref 做「是否已发表」的判定。
- 第 9–12 项的结论基于三点：(a) 站点 PDF 本身就是匿名评审稿（首页印有 "Anonymous … submission / Paper ID"）；(b) OpenAlex / Crossref 按标题检索无任何对应记录；(c) 无 DOI。**因此它们不是已发表论文**，"ECCV 2019 / CVPR 2019" 只是投稿年份，不是录用年份。
- 第 9–12 项的具体实验数字取自其个人主页叙述，未与正式出版版本比对（不存在正式版），引用需谨慎。

---

## 汇总表

| # | 标题 | 会议/年份 | 论文链接 | 代码链接 |
|---|------|-----------|----------|----------|
| 1 | 3D Visual Illusion Depth Estimation | NeurIPS 2025 | [arXiv:2505.13061](https://arxiv.org/abs/2505.13061) | [GitHub](https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation) |
| 2 | Diving into the Fusion of Monocular Priors for Generalized Stereo Matching | ICCV 2025 (Oral) | [arXiv:2505.14414](https://arxiv.org/abs/2505.14414) ｜ [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_Diving_into_the_Fusion_of_Monocular_Priors_for_Generalized_Stereo_ICCV_2025_paper.html) | [GitHub](https://github.com/YaoChengTang/Diving-into-the-Fusion-of-Monocular-Priors-for-Generalized-Stereo-Matching) |
| 3 | Sparse Point Guided 3D Lane Detection | ICCV 2023 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Sparse_Point_Guided_3D_Lane_Detection_ICCV_2023_paper.html) | [GitHub](https://github.com/YaoChengTang/Sparse-Point-Guided-3D-Lane-Detection) |
| 4 | Parameterized Cost Volume for Stereo Matching | ICCV 2023 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zeng_Parameterized_Cost_Volume_for_Stereo_Matching_ICCV_2023_paper.html) | [GitHub](https://github.com/JiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching) |
| 5 | FoggyStereo: Stereo Matching With Fog Volume Representation | CVPR 2022 | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yao_FoggyStereo_Stereo_Matching_With_Fog_Volume_Representation_CVPR_2022_paper.html) | [项目页](https://yaochengtang.github.io/FoggyStereo-Stereo-Matching-with-Fog-Volume-Representation/) ｜ [GitHub](https://github.com/YaoChengTang/FoggyStereo-Stereo-Matching-with-Fog-Volume-Representation) |
| 6 | Deep Stereo Matching With Explicit Cost Aggregation Sub-Architecture | AAAI 2018 (spotlight) | [arXiv:1801.04065](https://arxiv.org/abs/1801.04065) ｜ [AAAI](https://aaai.org/papers/12267-deep-stereo-matching-with-explicit-cost-aggregation-sub-architecture/) | [GitHub](https://github.com/lidongyv/RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture) |
| 7 | Content-Aware Inter-Scale Cost Aggregation for Stereo Matching | arXiv 2020（ECCV'20 投稿版，无录用记录） | [arXiv:2006.03209](https://arxiv.org/abs/2006.03209) | 无公开代码 |
| 8 | DRL-Based Trajectory Tracking for Motion-Related Modules in Autonomous Driving | arXiv 2023 | [arXiv:2308.15991](https://arxiv.org/abs/2308.15991) | [GitHub](https://github.com/MARMOTatZJU/drl-based-trajectory-tracking) |
| 9 | Explicit Context Mapping for Stereo Matching | **未能确证**（CVPR'19 / ICCV'19 投稿稿） | 仅站点 PDF，无公开论文页 | [GitHub](https://github.com/lidongyv/Explicit-Context-Mapping-for-Stereo-Matching) |
| 10 | A Learnable / Learning-based Region Support for Stereo Matching | **未能确证**（CVPR'18 / ECCV'19 投稿稿） | 仅站点 PDF，无公开论文页 | [GitHub](https://github.com/lidongyv/Interpretable-stereo-matching-with-region-support) |
| 11 | Region-Support Depth Inference From a Single Image | **未能确证**（NeurIPS / CVPR'19 投稿稿） | 仅站点 PDF，无公开论文页 | [GitHub](https://github.com/lidongyv/Monocular-depth-esitimation-with-region-support-cvpr) |
| 12 | Deep Stereo Matching With Regularization of Pixel-Set Knowledge | **未能确证**（TPAMI 手稿，未发表） | 仅站点 PDF，无公开论文页 | 无公开代码 |
