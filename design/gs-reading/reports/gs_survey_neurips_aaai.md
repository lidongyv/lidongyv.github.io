# 3D Gaussian Splatting 论文调研：NeurIPS 2025/2026 · AAAI 2025/2026

> 生成时间：2026-11-09 · 范围：NeurIPS 2025、NeurIPS 2026、AAAI 2025、AAAI 2026 中与 3DGS 相关的论文
> 分类体系：前馈重建(多视图→高斯) / per-scene优化 / 单图·少图生成 / 压缩与存储 / 动态·4D / 编辑与重光照 / SLAM·定位建图 / 语义·分割·理解 / 渲染改进 / 其他

## 一、检索方法与覆盖自检

**数据源（实际使用）**
1. **dblp 全站被 Anubis 反爬拦截**（HTTP 202 质询页，web_fetch 无法执行 JS），dblp 检索与会议页全部不可用，**未使用**。
2. **papers.nips.cc / proceedings.neurips.cc 在本环境网络不可达**（fetch failed ×5，重试无效），NeurIPS 官方 proceedings 页面无法直接核对，**改用 OpenReview 官方 API v2**（NeurIPS 2025 投稿/录用系统）枚举：`/notes/search?term=<词根>&content=title&group=NeurIPS.cc/2025/Conference`（term=gaussian、splat、gauss、3dgs 四个词根 + D&B 轨道 gauss/splat），按 offset 分页翻完（搜索 API 单页固定返回约 25 条）。
3. **AAAI 用官方 OJS 站内检索**（ojs.aaai.org，AAAI 官方论文集托管）：合并词元查询 `gaussian gaussians splatting splatter splats 3dgs`（OJS 多词为 OR 匹配），按 dateFromYear/dateToYear 过滤 2025 / 2026，逐页翻完；另以 3dgs / splatter 单词元查询交叉验证（3dgs 结果均被合并查询覆盖；splatter 两届均无命中）。
4. Semantic Scholar API 全程 HTTP 429 限流，未使用。arXiv 导出 API（export.arxiv.org）可用，用于代表性论文链接批量验证。

**覆盖自检（截至 2026-11-09）**

| 会议 | title 命中 gaussian/splat/3dgs 的条目总数 | 判定 3DGS 相关 | 排除(高斯过程/噪声/统计等无关) | 链接核实 |
|---|---|---|---|---|
| NeurIPS 2025 主会 | 107（+D&B 轨道 2） | 70（+D&B 2=72） | 37 | 全部条目链接来自 OpenReview 官方 API 返回（源头已核实）；6 篇另经 arXiv API 验证 |
| NeurIPS 2026 | 0（论文未公开，详见下文结论） | 0 | 0 | — |
| AAAI 2025 | 103 | 40 | 63 | 全部条目链接来自 OJS 官方检索页（源头已核实）；2 篇另经 arXiv API 验证 |
| AAAI 2026 | 196 | 102 | 94 | 同上；5 篇另经 arXiv API 验证 |

**链接说明**：每一条目的链接均为官方来源（NeurIPS→OpenReview forum 页；AAAI→ojs.aaai.org article 页），即检索页面实际返回的链接，视为已核实。另经 `export.arxiv.org/api` 标题检索验证 13 篇 arXiv 链接并给出 abs 地址；TranSplat(AAAI-25) 的 arXiv 检索被同名异文干扰未确认（链接待补，线索：`TranSplat Generalizable 3D Gaussian Splatting Sparse Multi-View Transformers site:arxiv.org`）。未对所有条目的 arXiv 版本逐一补录。

**抓取失败/缺口记录**：① dblp（含 NeurIPS 2026 页、AAAI 两届页）被反爬拦截，未取得 dblp 交叉验证；② papers.nips.cc 2025 hash 分段页本环境网络不可达，NeurIPS 2025 以 OpenReview 官方数据为准（覆盖主会全部录用论文标题）；③ Semantic Scholar 限流未用；④ NeurIPS 2025 的 Workshops 未纳入（仅主会 + D&B 轨道）。

## 二、NeurIPS 2025（3DGS 判定 72 篇：主会 70 + D&B 2；另有 37 篇含 gaussian/splat 标题的论文被判为无关已排除）

### 前馈重建(多视图→高斯)
- [Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting](https://openreview.net/forum?id=KsmgnL6PUb) | 前馈重建(多视图→高斯) | 体素融合精修前馈3DGS支持流式重建
- [ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](https://openreview.net/forum?id=zbucdbZ0fU) | 前馈重建(多视图→高斯) | 信息瓶颈压缩多视图输入扩展前馈3DGS
- [DGS-LRM: Real-Time Deformable 3D Gaussian Reconstruction From Monocular Videos](https://openreview.net/forum?id=X2u8esISdb) | 前馈重建(多视图→高斯) | 实时可变形3D高斯重建模型
- [DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving](https://openreview.net/forum?id=HF5A73jmxq) | 前馈重建(多视图→高斯) | 自动驾驶大4D高斯重建模型
- [PLANA3R: Zero-shot Metric Planar 3D Reconstruction via Feed-forward Planar Splatting](https://openreview.net/forum?id=YTwRZP8mNO) | 前馈重建(多视图→高斯) | 前馈平面泼溅零样本度量3D重建
- [SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction](https://openreview.net/forum?id=0sUihPtncP) | 前馈重建(多视图→高斯) | 可泛化高斯surfel稀疏视图表面重建

### per-scene优化
- [Deep Gaussian from Motion: Exploring 3D Geometric Foundation Models for Gaussian Splatting](https://openreview.net/forum?id=sl4KqWBkDq) | per-scene优化 | 3D几何基础模型免COLMAP位姿训练3DGS
- [Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting](https://openreview.net/forum?id=p3z75I8lVq) | per-scene优化 | 多尺度双边网格统一外观编码驾驶重建
- [Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](https://openreview.net/forum?id=GrPo8NTtzK) | per-scene优化 | 量化并缓解稀疏视图高斯共适应伪影
- [DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting](https://openreview.net/forum?id=PQcSYOBZii) | per-scene优化 | 方向一致性驱动自适应密度控制
- [Reframing Gaussian Splatting Densification with Complexity-Density Consistency of Primitives](https://openreview.net/forum?id=VKJTyhAtoA) | per-scene优化 | 视觉复杂度-密度一致性引导致密化
- [Holistic Large-Scale Scene Reconstruction via Mixed Gaussian Splatting](https://openreview.net/forum?id=yT8v2QFv5w) | per-scene优化 | 混合高斯表示整体大规模场景重建
- [EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](https://openreview.net/forum?id=shFhW4zqd6) | per-scene优化 | 事件相机辅助自由轨迹3DGS重建
- [NopeRoomGS: Indoor 3D Gaussian Splatting Optimization without Camera Pose Input](https://openreview.net/forum?id=SoPSI570Ap) | per-scene优化 | 免相机位姿输入的室内3DGS优化
- [3D Gaussian Flats: Hybrid 2D/3D Photometric Scene Reconstruction](https://openreview.net/forum?id=uVxQEIgXfL) | per-scene优化 | 2D/3D混合高斯光度场景重建
- [Eve3D: Elevating Vision Models for Enhanced 3D Surface Reconstruction via Gaussian Splatting](https://openreview.net/forum?id=rRPyI2h0yN) | per-scene优化 | 视觉基础模型提升GS表面重建
- [Metropolis-Hastings Sampling for 3D Gaussian Reconstruction](https://openreview.net/forum?id=BAscADMoQc) | per-scene优化 | MH采样优化3D高斯重建
- [MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild](https://openreview.net/forum?id=efDNv5XvVo) | per-scene优化 | 野外稀疏视图多外观3DGS重建
- [Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](https://openreview.net/forum?id=jPaM3AiFLq) | per-scene优化 | 非对称双3DGS抑制野外重建伪影
- [PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](https://openreview.net/forum?id=38GF07Tmtr) | per-scene优化 | 视觉语言平面先验引导室内3DGS
- [VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment](https://openreview.net/forum?id=ZnsR3waLUo) | per-scene优化 | 视图对齐增强3DGS几何表示
- [REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints](https://openreview.net/forum?id=6LGyChG6Ep) | per-scene优化 | 几何运动约束关节物体重建与生成

### 单图/少图生成
- [VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](https://openreview.net/forum?id=iLbXyPNTeb) | 单图/少图生成 | 单图音频驱动逼真高斯头部化身
- [TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](https://openreview.net/forum?id=iE0oCRx81h) | 单图/少图生成 | 时空裁剪加速3D高斯扩散推理
- [CGS-GAN: 3D Consistent Gaussian Splatting GANs for High Resolution Human Head Synthesis](https://openreview.net/forum?id=JoBCAKbCpC) | 单图/少图生成 | 3D一致高斯泼溅GAN高分辨率头部合成

### 压缩与存储
- [Optimized Minimal 3D Gaussian Splatting](https://arxiv.org/abs/2503.16924) | 压缩与存储 | 最小高斯数+子向量量化压缩50%
- [Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](https://openreview.net/forum?id=u8aoIxG1A4) | 压缩与存储 | 小波时域平滑率失真优化4DGS压缩
- [Gaussian Herding across Pens: An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS](https://openreview.net/forum?id=j1QkrVjNVF) | 压缩与存储 | 最优传输全局削减90%高斯
- [4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](https://openreview.net/forum?id=MbqPKsVe1d) | 压缩与存储 | 层级4D高斯渐进式体视频流压缩

### 动态/4D
- [Instant4D: 4D Gaussian Splatting in Minutes](https://arxiv.org/abs/2510.01119) | 动态/4D | 单目视频分钟级4D高斯重建
- [TreeSplat: Mergeable Tree for Deformable Gaussian Splatting](https://openreview.net/forum?id=5BfO9i4kj1) | 动态/4D | 树结构可合并变形高斯200+FPS
- [HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis](https://openreview.net/forum?id=V56unBiTHP) | 动态/4D | 可逆变形高斯具身视角合成
- [Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](https://openreview.net/forum?id=QeJOsm3qkS) | 动态/4D | 散焦+运动模糊视频动态GS重建
- [1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering](https://openreview.net/forum?id=YbKdduMtyN) | 动态/4D | 4DGS动态场景渲染超1000FPS
- [4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](https://openreview.net/forum?id=qMRFNxioPC) | 动态/4D | 单目视频学习4D高斯Transformer
- [DGH: Dynamic Gaussian Hair](https://openreview.net/forum?id=LUvu0H3uIG) | 动态/4D | 数据驱动动态头发高斯建模
- [H3D-DGS: Exploring Heterogeneous 3D Motion Representation for Deformable 3D Gaussian Splatting](https://openreview.net/forum?id=6VoDizmIoY) | 动态/4D | 异构3D运动表示变形高斯
- [HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](https://openreview.net/forum?id=ztVk8XNffY) | 动态/4D | 锚点驱动层次诱导流动态重建
- [MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](https://openreview.net/forum?id=Qr0YbuilFb) | 动态/4D | 物理模拟+3DGS精确动态化身
- [ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction of Dynamic Scenes](https://openreview.net/forum?id=bWZSy9ARPD) | 动态/4D | 连续体保持高斯流式动态重建
- [Spike4DGS: Towards High-Speed Dynamic Scene Rendering with 4D Gaussian Splatting via a Spike Camera Array](https://openreview.net/forum?id=V5efEA8nIr) | 动态/4D | 脉冲相机阵列高速4DGS动态渲染
- [Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](https://openreview.net/forum?id=tDT6HF7jK5) | 动态/4D | 关键点驱动运动紧凑高斯流式传输
- [Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](https://openreview.net/forum?id=gxgPjFjGmd) | 动态/4D | 方向锚定超高斯日常视频4D重建

### 编辑与重光照
- [Diffusion Feature Field for Text-based 3D Editing with Gaussian Splatting](https://openreview.net/forum?id=Kf9eNbp4wy) | 编辑与重光照 | 3D一致扩散特征解决Janus编辑问题
- [CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting](https://openreview.net/forum?id=kjWB8iaO3l) | 编辑与重光照 | 跨模态通用高斯风格迁移插件
- [3DOT: Texture Transfer for 3DGS Objects from a Single Reference Image](https://openreview.net/forum?id=BCpDJWshVV) | 编辑与重光照 | 单参考图渐进纹理迁移到3DGS
- [BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](https://openreview.net/forum?id=eSZcf8LCxz) | 编辑与重光照 | 混合神经着色可重光照高斯化身
- [MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting](https://openreview.net/forum?id=UXc87Orcri) | 编辑与重光照 | 元学习+Phong先验OOD场景重光照

### SLAM/定位建图
- [ProDyG: Progressive Dynamic Scene Reconstruction via Gaussian Splatting from Monocular Videos](https://openreview.net/forum?id=3iPmM7cLzx) | SLAM/定位建图 | 单目动态场景渐进SLAM重建
- [BevSplat: Resolving Height Ambiguity via Feature-Based Gaussian Primitives for Weakly-Supervised Cross-View Localization](https://openreview.net/forum?id=Ig5mtZ8etr) | SLAM/定位建图 | 特征高斯基元跨视角定位解高度歧义
- [3D Gaussian Splatting based Scene-independent Relocalization with Unidirectional and Bidirectional Feature Fusion](https://openreview.net/forum?id=ewgZItWaHh) | SLAM/定位建图 | 3DGS双阶段场景无关重定位

### 语义/分割/理解
- [FHGS: Feature-Homogenized Gaussian Splatting](https://openreview.net/forum?id=BEYzDKMEhX) | 语义/分割/理解 | 特征同质化蒸馏CLIP/SAM进3DGS
- [Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](https://openreview.net/forum?id=ycPVp0577R) | 语义/分割/理解 | 先分割再泼溅开放词表3D分割
- [LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS](https://arxiv.org/abs/2507.07136) | 语义/分割/理解 | 稀疏系数场加速语言高斯至450FPS

### 渲染改进
- [IBGS: Image-Based Gaussian Splatting](https://arxiv.org/abs/2511.14357) | 渲染改进 | 源图像残差补偿视角依赖高光细节
- [Anti-Aliased 2D Gaussian Splatting](https://arxiv.org/abs/2506.11252) | 渲染改进 | 世界空间滤波+Mip滤波抗锯齿2DGS
- [LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](https://arxiv.org/abs/2505.23158) | 渲染改进 | 层级LOD大规模场景移动端渲染
- [Fix False Transparency by Noise Guided Splatting](https://openreview.net/forum?id=uG8kRtNGEI) | 渲染改进 | 噪声引导泼溅修复伪透明伪影
- [SAP: Exact Sorting in Splatting via Screen-Aligned Primitives](https://openreview.net/forum?id=mGjmPnzGQo) | 渲染改进 | 屏幕对齐基元实现泼溅精确排序
- [Bézier Splatting for Fast and Differentiable Vector Graphics Rendering](https://openreview.net/forum?id=bTclOYRfYJ) | 渲染改进 | 贝塞尔泼溅快速可微矢量渲染(2D)
- [EGGS: Exchangeable 2D/3D Gaussian Splatting for Geometry-Appearance Balanced Novel View Synthesis](https://openreview.net/forum?id=25C8oC1pb2) | 渲染改进 | 2D/3D高斯可交换平衡几何外观
- [MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference](https://openreview.net/forum?id=I64ZLbUP6u) | 渲染改进 | 多视图一致材料推断反射高斯

### 其他
- [Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance](https://openreview.net/forum?id=KuXnKedjAj) | 其他 | 动物姿态外观3DGS量化建模
- [RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning](https://openreview.net/forum?id=9V3crVSPH7) | 其他 | 3DGS环境闭环RL训练驾驶策略
- [GeGS-PCR: Fast and Robust Color 3D Point Cloud Registration with Two-Stage Geometric-3DGS Fusion](https://openreview.net/forum?id=UkBwyp3aXG) | 其他 | 3DGS融合彩色点云配准
- [Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders](https://openreview.net/forum?id=hvykiwFiF8) | 其他 | 高斯增强物理仿真与系统辨识
- [GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis](https://openreview.net/forum?id=E3oNDQ8e9r) | 其他 | 复值3DGS高效射频数据合成
- [VR-Drive: Viewpoint-Robust End-to-End Driving with Feed-Forward 3D Gaussian Splatting](https://openreview.net/forum?id=NXPaQRErIT) | 其他 | 前馈3DGS视角增强端到端驾驶
- [SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](https://openreview.net/forum?id=plpAecfkf4) | 其他(存疑) | 查询式泼溅增强稀疏感知(非高斯,存疑)
- [GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](https://openreview.net/forum?id=asS4W7Yw5e) | 其他(存疑) | 高斯-图像协同扩散策略多智能体(存疑)

### NeurIPS 2025 Datasets & Benchmarks Track（2 篇）
- [SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting](https://openreview.net/forum?id=02ymnxlypN) | 其他 | 49K场景语言高斯数据集与基准
- [GS2E: Gaussian Splatting is an Effective Data Generator for Event Stream Generation](https://openreview.net/forum?id=XhNOSbLH0R) | 其他 | 3DGS合成事件流数据集生成器

### NeurIPS 2025 排除项（37 篇，title 含 gaussian/splat 但非 3DGS）
高斯过程族（GP回归/bandits/联邦学习/神经算子等约 20 篇：Gaussian Processes for Shuffled Regression、GP-UCB regret×2、RCaGP、Recurrent Memory GP、Sparse GP Power-EP、Turbocharging GP、Infinite Neural Operators、Robust Satisficing GP Bandits、No-Regret TS GP 等）、高斯混合/统计类（GMMotion、Continual GMM CISS、Contrastive GMM、Go With the Flow GMM、Self-Distillation GMM、Score-Based EB GMM、Generative Leap Multi-Index）、差分隐私/机制（Gaussian Mixing Mechanism、Sketched Gaussian Mechanism）、高斯近似理论（Gaussian Approximation SGD、Sharp Gaussian FL、Quantitative NN→GP）、非高斯因果（FANTOM、non-Gaussian Cyclic、GUI 等）、Gauss-Newton（RGNMR）、以及 3 篇"高斯"但非 3DGS 的视觉/表示论文：**GaussianFusion**（2D高斯多传感器融合自动驾驶，非泼溅渲染）、**GauSAM**（2D高斯场医学分割，非3DGS）、**GPSToken**（2D图像高斯参数化tokenization）。另剔除检索噪声：被拒稿 2 篇（Geometrically Consistent Generalizable Splatting、O-MMGP、Efficient Sampling DGP 等 Rejected_Submission）与官方评论帖（GauSAM 评论等）不计入。

## 三、NeurIPS 2026：截至 2026-11-09 论文未公开

- 核实途径：① OpenReview 官方 API：`NeurIPS.cc/2026/Conference` 组已存在（会议配置齐全：2026-12-06~12 悉尼主会场，相机就绪截止 2026-11-06），但 `notes/search`（term=deep/splat）返回 **0 条**，即无任何公开论文条目；② 官网 neurips.cc/Conferences/2026 无 accepted papers / proceedings 链接（最新公告为注册与 MLRC 2026 workshop 录用名单）；③ dblp 会议页因反爬无法核对、papers.nips.cc/paper_files/paper/2026 本环境网络不可达。
- **结论：NeurIPS 2026 论文尚未公开，不列入清单；预计 2026 年 12 月会议前后于 papers.nips.cc 与 OpenReview 公开。**

## 四、AAAI 2025（3DGS 判定 40 篇；另有 63 篇 title/摘要含关键词但非 3DGS 已排除）

### 前馈重建(多视图→高斯)
- [TranSplat: Generalizable 3D Gaussian Splatting from Sparse Multi-View Images with Transformers](https://ojs.aaai.org/index.php/AAAI/article/view/33070) | 前馈重建(多视图→高斯) | Transformer稀疏多视图泛化3DGS(arXiv待补)
- [DrivingForward: Feed-forward 3D Gaussian Splatting for Driving Scene Reconstruction from Flexible Surround-view Input](https://ojs.aaai.org/index.php/AAAI/article/view/32793) | 前馈重建(多视图→高斯) | 环视灵活输入驾驶场景前馈重建
- [GGS: Generalizable Gaussian Splatting for Lane Switching in Autonomous Driving](https://ojs.aaai.org/index.php/AAAI/article/view/32344) | 前馈重建(多视图→高斯) | 可泛化GS驾驶车道切换视角合成

### per-scene优化
- [Topology-Aware 3D Gaussian Splatting: Leveraging Persistent Homology for Optimized Structural Integrity](https://ojs.aaai.org/index.php/AAAI/article/view/32732) | per-scene优化 | 持续同调约束优化高斯结构完整性
- [FatesGS: Fast and Accurate Sparse-View Surface Reconstruction Using Gaussian Splatting with Depth-Feature Consistency](https://ojs.aaai.org/index.php/AAAI/article/view/32379) | per-scene优化 | 深度特征一致性稀疏视图表面重建
- [Micro-macro Wavelet-based Gaussian Splatting for 3D Reconstruction from Unconstrained Images](https://ojs.aaai.org/index.php/AAAI/article/view/32536) | per-scene优化 | 微宏小波提升无约束图像3D重建
- [Cross-Spectral Gaussian Splatting with Spatial Occupancy Consistency](https://ojs.aaai.org/index.php/AAAI/article/view/32333) | per-scene优化 | 跨光谱空间占据一致性重建
- [KeyGS: A Keyframe-Centric Gaussian Splatting Method for Monocular Image Sequences](https://ojs.aaai.org/index.php/AAAI/article/view/32195) | per-scene优化 | 关键帧中心单目图像序列GS
- [SpikeGS: Reconstruct 3D Scene Captured by a Fast-Moving Bio-Inspired Camera](https://ojs.aaai.org/index.php/AAAI/article/view/32340) | per-scene优化 | 脉冲相机高速运动场景重建
- [DeRainGS: Gaussian Splatting for Enhanced Scene Reconstruction in Rainy Environments](https://ojs.aaai.org/index.php/AAAI/article/view/32592) | per-scene优化 | 雨天环境增强场景重建
- [GigaGS: 3D Gaussian Based Planar Representation for Large-Scene Surface Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/32206) | per-scene优化 | 平面表示大场景表面重建
- [Enhancing Close-up Novel View Synthesis via Pseudo-labeling](https://ojs.aaai.org/index.php/AAAI/article/view/32925) | per-scene优化 | 伪标签增强特写新视角合成
- [Toy-GS: Assembling Local Gaussians for Precisely Rendering Large-Scale Free Camera Trajectories](https://ojs.aaai.org/index.php/AAAI/article/view/33108) | per-scene优化 | 局部高斯组装大范围自由轨迹渲染
- [Frequency-Aware Density Control via Reparameterization for High-Quality Rendering of 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/33066) | per-scene优化 | 重参数化频率感知密度控制
- [Novel View Synthesis Under Large-Deviation Viewpoint for Autonomous Driving](https://ojs.aaai.org/index.php/AAAI/article/view/32641) | per-scene优化(存疑) | 大视角偏差驾驶新视角合成(存疑)

### 单图/少图生成
- [GaussianPainter: Painting Point Cloud into 3D Gaussians with Normal Guidance](https://arxiv.org/abs/2412.17715) | 单图/少图生成 | 参考图法向引导点云前馈生成3D高斯
- [BloomScene: Lightweight Structured 3D Gaussian Splatting for Crossmodal Scene Generation](https://ojs.aaai.org/index.php/AAAI/article/view/32367) | 单图/少图生成 | 轻量结构化3DGS跨模态场景生成
- [Single-view Image to Novel-view Generation for Hand-Object Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/33128) | 单图/少图生成 | 手物交互单图新视角生成
- [GraphAvatar: Compact Head Avatars with GNN-Generated 3D Gaussians](https://ojs.aaai.org/index.php/AAAI/article/view/32895) | 单图/少图生成 | GNN生成3D高斯紧凑头部化身
- [PointTalk: Audio-Driven Dynamic Lip Point Cloud for 3D Gaussian-based Talking Head Synthesis](https://ojs.aaai.org/index.php/AAAI/article/view/32946) | 单图/少图生成 | 音频驱动动态唇部高斯说话头
- [3D²-Actor: Learning Pose-Conditioned 3D-Aware Denoiser for Realistic Gaussian Avatar Modeling](https://ojs.aaai.org/index.php/AAAI/article/view/32789) | 单图/少图生成 | 位姿条件3D感知去噪器高斯化身

### 压缩与存储
- [Compressing Streamable Free-Viewpoint Videos to 0.1 MB per Frame](https://ojs.aaai.org/index.php/AAAI/article/view/32780) | 压缩与存储 | 流式自由视角视频压缩0.1MB/帧

### 动态/4D
- [Efficient Gaussian Splatting for Monocular Dynamic Scene Rendering via Sparse Time-Variant Attribute Modeling](https://ojs.aaai.org/index.php/AAAI/article/view/32460) | 动态/4D | 稀疏时变属性建模单目动态渲染
- [EvHDR-GS: Event-guided HDR Video Reconstruction with 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/32237) | 动态/4D | 事件引导HDR视频3DGS重建
- [Hierarchically Controlled Deformable 3D Gaussians for Talking Head Synthesis](https://ojs.aaai.org/index.php/AAAI/article/view/32921) | 动态/4D | 层级控制变形高斯说话头合成
- [Motion Decoupled 3D Gaussian Splatting for Dynamic Object Representation](https://ojs.aaai.org/index.php/AAAI/article/view/32373) | 动态/4D | 运动解耦动态物体表示
- [GFlow: Recovering 4D World from Monocular Video](https://ojs.aaai.org/index.php/AAAI/article/view/32847) | 动态/4D | 单目视频高斯流恢复4D世界

### 编辑与重光照
- [Multi-StyleGS: Stylized Gaussian Splatting with Multiple Styles](https://ojs.aaai.org/index.php/AAAI/article/view/32562) | 编辑与重光照 | 多风格嵌入高斯泼溅风格化

### 语义/分割/理解
- [SOVGaussian: Sparse-View 3D Gaussian Splatting for Open-Vocabulary Scene Understanding](https://ojs.aaai.org/index.php/AAAI/article/view/32568) | 语义/分割/理解 | 稀疏视图3DGS开放词表理解
- [Segment Any 3D Gaussians](https://ojs.aaai.org/index.php/AAAI/article/view/32193) | 语义/分割/理解 | SAM引导任意3D高斯交互分割
- [FastLGS: Speeding Up Language Embedded Gaussians with Feature Grid Mapping](https://ojs.aaai.org/index.php/AAAI/article/view/32410) | 语义/分割/理解 | 特征网格映射加速语言高斯
- [Bootstraping Clustering of Gaussians for View-consistent 3D Scene Understanding](https://ojs.aaai.org/index.php/AAAI/article/view/33103) | 语义/分割/理解 | 高斯聚类自举视角一致3D理解
- [Rethinking Open-Vocabulary Segmentation of Radiance Fields in 3D Space](https://ojs.aaai.org/index.php/AAAI/article/view/32473) | 语义/分割/理解 | 3D空间开放词表分割框架反思

### 渲染改进
- [Decoupling Appearance Variations with 3D Consistent Features in Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/32556) | 渲染改进 | 3D一致特征解耦外观变化渲染

### 其他
- [GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution](https://arxiv.org/abs/2407.18046) | 其他 | 2D高斯泼溅任意尺度图像超分(2D)
- [Large Images Are Gaussians: High-Quality Large Image Representation with Levels of 2D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/33193) | 其他 | 层级2D高斯表示大图像(2D)
- [EGSRAL: An Enhanced 3D Gaussian Splatting Based Renderer with Automated Labeling for Large-Scale Driving Scene](https://ojs.aaai.org/index.php/AAAI/article/view/32403) | 其他 | 3DGS渲染器+驾驶场景自动标注
- [HOGSA: Bimanual Hand-Object Interaction Understanding with 3D Gaussian Splatting Based Data Augmentation](https://ojs.aaai.org/index.php/AAAI/article/view/32712) | 其他 | 3DGS数据增强双手物交互理解
- [Self-Correcting Robot Manipulation via Gaussian-Splatted Foresight](https://ojs.aaai.org/index.php/AAAI/article/view/34866) | 其他 | 高斯泼溅前视自纠正机械臂操作
- [ProtoCar: Learning 3D Vehicle Prototypes from Single-View and Unconstrained Driving Scene Images](https://ojs.aaai.org/index.php/AAAI/article/view/32581) | 其他(存疑) | 单视图驾驶图车辆3D原型(存疑)

### AAAI 2025 排除项（63 篇）
高斯过程回归/优化/时间序列（Summarized Data GP、KernelMatmul、Error Bounds GP、Asynchronous Distributed GP、GP相关BO若干）、非高斯因果与统计建模（Non-Gaussian Graphical Models、Gaussian Graphical Modelling、Boosting Causal Structure、DR-Encoder 等）、高斯噪声/滤波类（Truncated Gaussian Policy、GHOST、A Gaussian Filter-Based EM Registration、BGDB、ProPose）、以及摘要命中"gaussian"但与 3DGS 无关的其他论文（扩散/图/检索等约 40 篇，如 DreaMark 为 NeRF 水印）。

## 五、AAAI 2026（3DGS 判定 102 篇；另有 94 篇已排除）

### 前馈重建(多视图→高斯)
- [LongSplat: Online Generalizable 3D Gaussian Splatting from Long Sequence Images](https://arxiv.org/abs/2507.16144) | 前馈重建(多视图→高斯) | 在线可泛化长序列图像3DGS
- [SkySplat: Generalizable 3D Gaussian Splatting from Multi-Temporal Sparse Satellite Images](https://arxiv.org/abs/2508.09479) | 前馈重建(多视图→高斯) | 多时相稀疏卫星图像泛化3DGS
- [MuSASplat: Efficient Sparse-View 3D Gaussian Splats via Lightweight Multi-Scale Adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/38115) | 前馈重建(多视图→高斯) | 轻量多尺度适配稀疏视图高斯
- [Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/38362) | 前馈重建(多视图→高斯) | 尺度感知点图前馈人体场景重建
- [One-Shot Refiner: Boosting Feed-forward Novel View Synthesis via One-Step Diffusion](https://ojs.aaai.org/index.php/AAAI/article/view/37365) | 前馈重建(多视图→高斯) | 一步扩散增强前馈新视角合成
- [MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](https://arxiv.org/abs/2508.17811) | 前馈重建(多视图→高斯) | 2DGS桥接可泛化稀疏视图表面重建
- [VGGS: VGGT-guided Gaussian Splatting for Efficient and Faithful Sparse-View Surface Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/38074) | 前馈重建(多视图→高斯) | VGGT引导高效稀疏视图表面重建
- [SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images](https://ojs.aaai.org/index.php/AAAI/article/view/42499) | 前馈重建(多视图→高斯) | 稀疏多视图前馈超分高斯泼溅
- [PHOTONS: Pose-Free Human-Centric Photo-Realistic Real-Time Novel View Synthesis from Sparse Views](https://ojs.aaai.org/index.php/AAAI/article/view/42340) | 前馈重建(多视图→高斯)(存疑) | 免位姿人体实时稀疏视图合成(存疑)

### per-scene优化
- [TrackGS: Optimizing COLMAP-Free 3D Gaussian Splatting with Global Track Constraints](https://ojs.aaai.org/index.php/AAAI/article/view/37851) | per-scene优化 | 全局轨迹约束免COLMAP 3DGS
- [TileGS: Adaptive Gaussian Densification Through Tile-Guided Perceptual Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/37997) | per-scene优化 | 分块感知分析自适应高斯致密化
- [Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](https://ojs.aaai.org/index.php/AAAI/article/view/37438) | per-scene优化 | 自适应探索+曲率感知开发优化
- [COSMOS: Coherent SuperGaussian Modeling with Spatial Priors for Sparse-View 3D Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37446) | per-scene优化 | 空间先验超高斯稀疏视图泼溅
- [DentalGS: Pose-Free 3D Gaussian Splatting from Five Intraoral Images for Novel View Synthesis](https://ojs.aaai.org/index.php/AAAI/article/view/37343) | per-scene优化 | 5张口内图像免位姿牙科3DGS
- [CAG-GS: Consistent Anchor Guided Gaussian Splatting for Large-scale Scene Rendering](https://ojs.aaai.org/index.php/AAAI/article/view/38120) | per-scene优化 | 一致锚点引导大规模场景渲染
- [Reconstruction Using the Invisible: Intuition from NIR and Metadata for Enhanced 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37259) | per-scene优化 | NIR与元数据先验增强3DGS
- [OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/37498) | per-scene优化 | 目标感知水下场景三目一致性重建
- [DehazeGS: Seeing Through Fog with 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/38205) | per-scene优化 | 雾天场景3DGS去雾重建
- [Scene Experts: Specializing in 3D Gaussian Splatting with Adaptive Decomposition](https://ojs.aaai.org/index.php/AAAI/article/view/37407) | per-scene优化 | 自适应分解专家化3DGS
- [SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/42428) | per-scene优化 | 稀疏视图3DGS表面重建
- [AquaSplatting: A Hybrid 3D Representation for Robust Underwater Scene Reconstruction via Dual-Branch Rendering](https://ojs.aaai.org/index.php/AAAI/article/view/42487) | per-scene优化 | 双分支混合表示水下鲁棒重建
- [TSPE-GS: Probabilistic Depth Extraction for Semi-Transparent Surface Reconstruction via 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/38130) | per-scene优化 | 概率深度提取半透明表面重建
- [Through the Water: Refractive Gaussian Splatting for Water Surface Scenes](https://ojs.aaai.org/index.php/AAAI/article/view/38197) | per-scene优化 | 折射建模水面场景高斯重建
- [3D Gaussian Splatting for Reconstructing Large Sparse Environments (Student Abstract)](https://ojs.aaai.org/index.php/AAAI/article/view/42189) | per-scene优化 | 大型稀疏环境3DGS重建(学生摘要)
- [Cross-temporal 3D Gaussian Splatting for Sparse-view Guided Scene Update](https://ojs.aaai.org/index.php/AAAI/article/view/37217) | per-scene优化 | 跨时相稀疏视图引导场景更新
- [Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37470) | per-scene优化 | 3D注意力去偏扩散先验
- [Pano-GS: Perception-Aware Gaussian Optimization with Gradient Consistency and Multi-Criteria Densification for High-Quality Rendering](https://ojs.aaai.org/index.php/AAAI/article/view/37354) | per-scene优化 | 感知感知优化+多准则致密化
- [LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](https://ojs.aaai.org/index.php/AAAI/article/view/37289) | per-scene优化 | 扩散先验改进LiDAR高斯重建
- [Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning](https://ojs.aaai.org/index.php/AAAI/article/view/38150) | per-scene优化 | 透视变换亮度调优雨天3D重建
- [IGFuse: Interactive 3D Gaussian Scene Reconstruction via Multi-Scans Fusion](https://ojs.aaai.org/index.php/AAAI/article/view/42497) | per-scene优化 | 多扫描融合交互式3D高斯重建
- [PMGS: Reconstruction of Projectile Motion Across Large Spatiotemporal Spans via 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/38128) | per-scene优化 | 大时空跨度弹道运动3DGS重建
- [Active3D: Active High-Fidelity 3D Reconstruction via Multi-Level Uncertainty Quantification](https://ojs.aaai.org/index.php/AAAI/article/view/37585) | per-scene优化(存疑) | 多级不确定性主动3D重建(存疑)

### 单图/少图生成
- [4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation](https://ojs.aaai.org/index.php/AAAI/article/view/37659) | 单图/少图生成 | 时空校正生成式4D高斯
- [DipGuava: Disentangling Personalized Gaussian Features for 3D Head Avatars from Monocular Video](https://ojs.aaai.org/index.php/AAAI/article/view/37510) | 单图/少图生成 | 单目视频解耦个性化头部化身
- [EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion](https://ojs.aaai.org/index.php/AAAI/article/view/37663) | 单图/少图生成(存疑) | 双稀疏潜扩散可扩展3D地球生成(存疑)
- [Orthogonal Spatial-temporal Distributional Transfer for 4D Generation](https://ojs.aaai.org/index.php/AAAI/article/view/37666) | 单图/少图生成(存疑) | 正交时空分布迁移4D生成(存疑)
- [High-Quality Full-Head 3D Avatar Generation from Any Single Portrait Image](https://ojs.aaai.org/index.php/AAAI/article/view/42417) | 单图/少图生成(存疑) | 单张肖像全头3D化身生成(存疑)

### 压缩与存储
- [NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting for Compact 3D Representations](https://ojs.aaai.org/index.php/AAAI/article/view/37910) | 压缩与存储 | 神经场桥接3DGS紧凑表示
- [Plug-and-Play Optimization for 3D Gaussian Splatting Compression: Distribution Regularization, Probabilistic Pruning and Detail Compensation](https://ojs.aaai.org/index.php/AAAI/article/view/37222) | 压缩与存储 | 分布正则+概率剪枝+细节补偿压缩
- [PCGS: Progressive Compression of 3D Gaussian Splatting](https://arxiv.org/abs/2503.08511) | 压缩与存储 | 渐进式掩码量化3DGS压缩
- [SymGS: Leveraging Reflective Symmetries for 3DGS Compression](https://ojs.aaai.org/index.php/AAAI/article/view/42449) | 压缩与存储 | 反射对称性利用3DGS压缩
- [D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos](https://ojs.aaai.org/index.php/AAAI/article/view/38674) | 压缩与存储 | 前馈动态GS自由视角视频压缩

### 动态/4D
- [CPOStream: Collaborating Prediction and Observation for Flicker-Free Streamable Free-Viewpoint Video with 3DGS](https://ojs.aaai.org/index.php/AAAI/article/view/37228) | 动态/4D | 预测观测协同无闪烁流式FVV
- [4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor Growing for Efficient and High-Fidelity Dynamic Scene Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/37332) | 动态/4D | 动态感知锚点生长脚手架4DGS
- [HDGS: Hierarchical Dynamic Gaussian Splatting for Urban Driving Scenes](https://ojs.aaai.org/index.php/AAAI/article/view/42419) | 动态/4D | 层级动态高斯城市驾驶场景
- [Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](https://ojs.aaai.org/index.php/AAAI/article/view/42474) | 动态/4D | 物理信息变形高斯本构材料场
- [Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video](https://ojs.aaai.org/index.php/AAAI/article/view/38047) | 动态/4D | 模糊单目视频去模糊4DGS
- [StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](https://ojs.aaai.org/index.php/AAAI/article/view/37482) | 动态/4D | 时空高斯网格流式实时FVV
- [Multi-Step Deformable Gaussian Splatting for Dynamic Scene Rendering](https://ojs.aaai.org/index.php/AAAI/article/view/42486) | 动态/4D | 多步可变形高斯动态场景渲染
- [MGD: Mesh-guided Gaussians with Diffusion Priors for Dynamic Objects Reconstruction from Monocular RGB-D Video](https://ojs.aaai.org/index.php/AAAI/article/view/38088) | 动态/4D | 网格引导+扩散先验单目动态重建
- [Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/37848) | 动态/4D | 稀疏帧动态场景4DGS重建
- [DcSplat: Dual-Constraint Human Gaussian Splatting with Latent Multi-View Consistency](https://ojs.aaai.org/index.php/AAAI/article/view/38077) | 动态/4D | 双约束潜空间多视图一致人体GS
- [MCGS: Markov Chain Gaussian Splatting for Dynamic Scenes Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/38004) | 动态/4D | 马尔可夫链高斯动态场景重建
- [UAV4D: Dynamic Neural Rendering of Human-Centric UAV Imagery Using Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37333) | 动态/4D | 无人机人体影像动态高斯渲染
- [MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](https://ojs.aaai.org/index.php/AAAI/article/view/37234) | 动态/4D | 运动去模糊动态3DGS
- [Edge Consistency for 4D Gaussian Splatting in Dynamic Scene Rendering](https://ojs.aaai.org/index.php/AAAI/article/view/37847) | 动态/4D | 边缘一致性约束4DGS动态渲染
- [DenoiseGS: Delta-Based 3D Gaussian Splatting with B-spline Trajectory Optimization for Dynamic Driving Scene Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/37640) | 动态/4D | Delta高斯+B样条轨迹驾驶重建
- [Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos](https://ojs.aaai.org/index.php/AAAI/article/view/38129) | 动态/4D | 非同步视频动态高斯重建
- [FLAG-4D: Flow-Guided Local-Global Dual-Deformation Model for 4D Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/37889) | 动态/4D | 光流引导局部全局双变形4D重建
- [BulletTime4D: Towards High Spatio-Temporal Resolution Dynamic Scene Rendering via Spike-Guided Stereo Vision](https://ojs.aaai.org/index.php/AAAI/article/view/37263) | 动态/4D(存疑) | 脉冲引导立体高时空分辨率渲染(存疑)

### 编辑与重光照
- [UV-RGS: Relightable 3D Gaussian Splatting from Unposed Views Under Varied Illuminations](https://ojs.aaai.org/index.php/AAAI/article/view/37397) | 编辑与重光照 | 免位姿变光照可重光照3DGS
- [FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/38164) | 编辑与重光照 | 可控风格化蒸馏3DGS
- [Personalize Your Gaussian: Consistent 3D Scene Personalization from a Single Image](https://ojs.aaai.org/index.php/AAAI/article/view/38003) | 编辑与重光照 | 单图一致性3D场景个性化
- [GPGS: Consistent 3D Object Removal via Geometry-Aware 3D Inpainting and Projected Image Refinement in 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37515) | 编辑与重光照 | 几何感知修复一致3D物体移除
- [GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37667) | 编辑与重光照 | 几何感知纹理迁移高斯泼溅
- [WeatherEdit: Controllable Weather Editing with 4D Gaussian Field](https://ojs.aaai.org/index.php/AAAI/article/view/37802) | 编辑与重光照 | 4D高斯场可控天气编辑

### SLAM/定位建图
- [HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction (Abstract Reprint)](https://ojs.aaai.org/index.php/AAAI/article/view/41423) | SLAM/定位建图 | 几何感知高斯SLAM快速单目重建
- [CoMA-SLAM: Collaborative Multi-Agent Gaussian SLAM with Geometric Consistency](https://ojs.aaai.org/index.php/AAAI/article/view/37283) | SLAM/定位建图 | 多智能体协同高斯SLAM
- [PointSLAM++: Robust Dense Neural Gaussian Point Cloud-based SLAM](https://ojs.aaai.org/index.php/AAAI/article/view/37991) | SLAM/定位建图 | 鲁棒稠密神经高斯点云SLAM

### 语义/分割/理解
- [Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/37269) | 语义/分割/理解 | 视觉高斯协同语义占用预测
- [SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion](https://ojs.aaai.org/index.php/AAAI/article/view/37803) | 语义/分割/理解 | 深度引导解耦高斯语义场景补全
- [IQGS: Instance Query-based Gaussian Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/42415) | 语义/分割/理解 | 实例查询高斯分割
- [Binary-Gaussian: Compact and Progressive Representation for 3D Gaussian Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/38142) | 语义/分割/理解 | 紧凑渐进二值3D高斯分割
- [GUIDE: Gaussian Unified Instance Detection for Enhanced Obstacle Perception in Autonomous Driving](https://ojs.aaai.org/index.php/AAAI/article/view/42484) | 语义/分割/理解 | 高斯统一实例检测驾驶障碍感知
- [GAGS: Granularity-Aware Feature Distillation for Language Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37787) | 语义/分割/理解 | 粒度感知特征蒸馏语言高斯
- [Evolving Semantic Propagation for Aerial Semantic 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/42418) | 语义/分割/理解 | 进化语义传播航空3DGS
- [ID-Splat: Propagating Object Identities for Segmenting 3D Aerial-view Scenes](https://ojs.aaai.org/index.php/AAAI/article/view/37995) | 语义/分割/理解 | 目标身份传播分割航空3D场景
- [UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](https://ojs.aaai.org/index.php/AAAI/article/view/37357) | 语义/分割/理解(存疑) | 对比学习统一3D实例分割(存疑)

### 渲染改进
- [UniMGS: Unifying Mesh and 3D Gaussian Splatting with Single-Pass Rasterization and Proxy-Based Deformation](https://ojs.aaai.org/index.php/AAAI/article/view/38080) | 渲染改进 | 网格与3DGS单趟光栅化统一
- [Pushing Rendering Boundaries: Hard Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/38116) | 渲染改进 | 难度样本加权突破渲染边界
- [Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](https://arxiv.org/abs/2511.15102) | 渲染改进 | 空间分布高斯混合替代alpha混合
- [IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution](https://ojs.aaai.org/index.php/AAAI/article/view/37398) | 渲染改进 | 内外知识融合3DGS超分
- [PATexGS: Perceptual-Adaptive Texture Scheduling for Visual Coherence in Textured Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37999) | 渲染改进 | 感知自适应纹理调度保视觉一致
- [Arbitrary-Scale 3D Gaussian Super-Resolution](https://ojs.aaai.org/index.php/AAAI/article/view/38222) | 渲染改进 | 任意尺度3D高斯超分辨率

### 其他
- [Splats in Splats: Robust and Effective 3D Steganography Towards Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/42447) | 其他 | 3DGS隐写鲁棒信息隐藏(安全)
- [Perceptual Quality Assessment of 3D Gaussian Splatting: A Subjective Dataset and Prediction Metric](https://ojs.aaai.org/index.php/AAAI/article/view/37928) | 其他 | 3DGS主观质量数据集与预测指标
- [Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?](https://ojs.aaai.org/index.php/AAAI/article/view/37429) | 其他 | 3DGS版权保护水印可行性(安全)
- [GS-Checker: Tampering Localization for 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/42455) | 其他 | 3DGS篡改定位检测(安全)
- [Fading the Digital Ink: A Universal Black-Box Attack Framework for 3DGS Watermarking Systems](https://ojs.aaai.org/index.php/AAAI/article/view/41146) | 其他 | 3DGS水印系统黑盒攻击框架(安全)
- [PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](https://ojs.aaai.org/index.php/AAAI/article/view/37477) | 其他 | 物理信息高斯泼溅4D流MRI超分(医学)
- [FreeGaussian: Annotation-free Control of Articulated Objects via 3D Gaussian Splats with Flow Derivatives](https://ojs.aaai.org/index.php/AAAI/article/view/37291) | 其他 | 流导数高斯免标注关节物体控制
- [SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images](https://ojs.aaai.org/index.php/AAAI/article/view/37558) | 其他 | 特征智能高斯超高分辨率图像压缩(2D)
- [GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/37572) | 其他 | 2D高斯泼溅增强图像压缩(2D)
- [Zero-Shot Robotic Manipulation via 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation](https://ojs.aaai.org/index.php/AAAI/article/view/38936) | 其他 | 3DGS增强多模态RAG零样本机械臂
- [TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/38361) | 其他 | 几何感知辐射高斯场体层重建
- [2D Gaussians Spatial Transport for Point-supervised Density Regression](https://ojs.aaai.org/index.php/AAAI/article/view/37836) | 其他 | 2D高斯空间传输点监督密度回归(2D)
- [Style4D-Bench: A Benchmark Suite for 4D Stylization](https://ojs.aaai.org/index.php/AAAI/article/view/37266) | 其他 | 4D风格化基准套件(基准)
- [3DDM: Physically-based Anisotropic 3D Diffusion Model with 3D Gaussian for Point Cloud Completion](https://ojs.aaai.org/index.php/AAAI/article/view/38070) | 其他(存疑) | 各向异性3D高斯扩散点云补全(存疑)
- [DiffNR: Diffusion-Enhanced Neural Representation Optimization for Sparse-View 3D Tomographic Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/37871) | 其他(存疑) | 扩散增强神经表示体层重建(存疑)
- [Slender3D: Curve-Guided Multi-View Reconstruction of Slender Structures](https://ojs.aaai.org/index.php/AAAI/article/view/37979) | 其他(存疑) | 曲线引导细长结构多视图重建(存疑)
- [InterCoser: Interactive 3D Character Creation with Disentangled Fine-Grained Features](https://ojs.aaai.org/index.php/AAAI/article/view/37993) | 其他(存疑) | 解耦细粒度特征交互3D角色创建(存疑)
- [MoReMouse: Monocular Reconstruction of Laboratory Mouse](https://ojs.aaai.org/index.php/AAAI/article/view/38360) | 其他(存疑) | 实验鼠单目重建(存疑)

### AAAI 2026 排除项（94 篇）
高斯过程/统计类（Streaming GP Experts、Flow-Induced Diagonal GP、GP Chaining、FANOVA GP、Reward Redistribution GP、Gaussian Approximation Two-Timescale 等约 15 篇）、高斯噪声/模型类（Multivariate Gaussian Medical Action、Gaussian Uncertainty Multi-Model Fitting、Contrastive Gaussian Fusion Clustering、GUI-G²、GEM、Revisiting Downsampling Gaussian/Gabor、Discovering Linear non-Gaussian 等约 12 篇）、以及摘要命中关键词但主题无关的大量论文（扩散、图学习、RL、医学、遥感等约 65 篇）；Student Abstract 中仅 1 篇 3DGS 保留，其余排除。

