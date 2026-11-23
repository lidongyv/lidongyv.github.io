# ICCV 2025 与 ECCV 2026 3D Gaussian Splatting 论文系统清单

> 生成时间：2026-11-09（Asia/Shanghai）。本清单仅收录在两届会议官方页面实际出现并逐条核实的论文；未凭模型记忆补录任何条目。

## 一、检索方法与覆盖自检

### 1. 数据源与抓取情况

- **ICCV 2025**：CVF Open Access `https://openaccess.thecvf.com/ICCV2025?day=all`，HTTP 200，本地全文 7.69 MB，共 **2701** 篇主会论文（bibtex 页码范围 284–29163）。`?day=1..5` 分页接口在抓取当日均返回 HTTP 500（服务器端错误），无法按天交叉核对；`day=all` 包含全部条目，以之为准。
- **ECCV 2026**：先核实可获取性——`https://openaccess.thecvf.com/ECCV2026?day=all` 返回 **HTTP 404**（CVF 尚未发布 ECCV 2026 开放获取页）；dblp 页面 `dblp.org/db/conf/eccv/eccv2026.html` 被 Anubis 反爬挑战拦截（HTTP 202，无法取到内容）。改用 **ECCV 官方站点**（`eccv.ecva.net`）：会议已确认于 2026-09-08~12 在瑞典 Malmö 召开，Springer LNCS 出版 78 卷；官方 "List of Accepted Papers" 页（`/Conferences/2026/AcceptedPapers`，注明"preliminary pending publisher checks"）共 2851 行、去重后 **2834** 篇，每篇附公开 poster 页（含摘要与 PDF 链接）。**结论：ECCV 2026 论文已公开可获取（官方虚拟站点），仅 CVF openaccess 尚未上线。**
- **筛选规则**：标题含 `gaussian` 或 `splat`（不区分大小写）→ 候选；再人工判定是否属 3DGS 主题。由于部分论文标题只写 `GS/3DGS` 缩写（如 MagnetGS-Mesh、S3R-GS），在规则之外补充扫描标题内嵌 `GS` 的条目并逐条人工核对，作为补充收录。
- **链接验证**：对全部收录论文的官方链接逐条做 HTTP GET 验证（ICCV 使用 CVF PDF href，ECCV 使用官方 poster 页 href）。ICCV 服务器对分段请求返回 206 Partial Content（资源存在即通过），ECCV 返回 200。全部 313 条链接验证通过。

### 2. 覆盖自检统计

| 指标 | ICCV 2025 | ECCV 2026 | 合计 |
| --- | --- | --- | --- |
| 会议论文总数（抓取到） | 2701 | 2834（去重） | 5535 |
| 标题含 gaussian/splat 候选数 | 163 | 148（去重 147） | 311 |
| 判定为 3DGS 主题 | **163**（含 5 篇 GS 缩写补录） | **150**（含 13 篇 GS 缩写补录） | **313** |
| 排除（高斯过程/噪声/无关） | 9 | 12 | 21 |
| 链接验证通过 | 163/163 | 150/150 | 313/313 |
| 链接验证失败 | 0 | 0 | 0 |

### 3. 分类分布

- ICCV 2025（163 篇）：前馈重建 11、per-scene优化 48、单图/少图生成 12、压缩与存储 3、动态/4D 20、编辑与重光照 13、SLAM/定位建图 9、语义/分割/理解 21、渲染改进 5、其他 21
- ECCV 2026（150 篇）：前馈重建 18、per-scene优化 41、单图/少图生成 8、压缩与存储 10、动态/4D 20、编辑与重光照 11、SLAM/定位建图 11、语义/分割/理解 9、渲染改进 6、其他 16

### 4. 覆盖缺口说明（如实记录）

- ICCV 2025 的 `?day=1..5` 分页接口返回 500，未逐天核对；`day=all` 页面完整（2701 条）且与公开页码范围一致，视为全覆盖。
- ICCV 2025 的 day=all 页面曾被 web_fetch 截断；已改用 Python 直连下载全文，无截断。
- ECCV 2026 官方列表注明 "preliminary pending publisher checks"，可能随出版社终审微调；以抓取当日（2026-11-09）快照为准。
- 标题仅含 "GS" 缩写但无 gaussian/splat 的论文通过补充扫描人工补录（ICCV +5、ECCV +13）；不排除极少数标题完全不含任何 GS 相关词（如个别以 "Splatting" 变体拼写）被遗漏的可能性，此概率极低。
- 存疑归"其他"并标注的条目：ICCV 3 篇（Gaussian-based World Model、SketchSplat、Mixture of Experts Guided by Gaussian Splatters）、ECCV 3 篇（Complex-Valued 2D Gaussian/CGH、Heat Kernel Textures、MorphGS）。

---

## 二、ICCV 2025 — 3DGS 论文清单（163 篇，链接全部验证通过）

### 前馈重建(多视图→高斯)（11 篇）

- [RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_RayletDF_Raylet_Distance_Fields_for_Generalizable_3D_Surface_Reconstruction_from_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 点云/高斯泛化表面重建距离场
- [StreamGS: Online Generalizable Gaussian Splatting Reconstruction for Unposed Image Streams](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_StreamGS_Online_Generalizable_Gaussian_Splatting_Reconstruction_for_Unposed_Image_Streams_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 无位姿图像流在线泛化重建
- [MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Lou_MuGS_Multi-Baseline_Generalizable_Gaussian_Splatting_Reconstruction_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 多基线泛化高斯重建
- [PanoSplatt3R: Leveraging Perspective Pretraining for Generalized Unposed Wide-Baseline Panorama Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Ren_PanoSplatt3R_Leveraging_Perspective_Pretraining_for_Generalized_Unposed_Wide-Baseline_Panorama_Reconstruction_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 无位姿宽基线全景泛化重建
- [GSRecon: Efficient Generalizable Gaussian Splatting for Surface Reconstruction from Sparse Views](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_GSRecon_Efficient_Generalizable_Gaussian_Splatting_for_Surface_Reconstruction_from_Sparse_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 稀疏视图泛化表面重建
- [GUAVA: Generalizable Upper Body 3D Gaussian Avatar](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_GUAVA_Generalizable_Upper_Body_3D_Gaussian_Avatar_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 可泛化上半身高斯头像
- [Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](https://openaccess.thecvf.com/content/ICCV2025/papers/Kirschstein_Avat3r_Large_Animatable_Gaussian_Reconstruction_Model_for_High-fidelity_3D_Head_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 大规模可动高斯头部重建模型
- [Learning Efficient and Generalizable Human Representation with Human Gaussian Model](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Learning_Efficient_and_Generalizable_Human_Representation_with_Human_Gaussian_Model_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 高效可泛化人体高斯表示
- [Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats](https://openaccess.thecvf.com/content/ICCV2025/papers/Ziwen_Long-LRM_Long-sequence_Large_Reconstruction_Model_for_Wide-coverage_Gaussian_Splats_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 长序列大重建模型宽覆盖高斯
- [UniGS: Modeling Unitary 3D Gaussians for Novel View Synthesis from Sparse-view Images](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_UniGS_Modeling_Unitary_3D_Gaussians_for_Novel_View_Synthesis_from_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 统一高斯稀疏视图新视角合成
- [CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](https://openaccess.thecvf.com/content/ICCV2025/papers/Roh_CATSplat_Context-Aware_Transformer_with_Spatial_Guidance_for_Generalizable_3D_Gaussian_ICCV_2025_paper.pdf) | 前馈重建(多视图→高斯) | 单图上下文Transformer泛化3DGS

### per-scene优化（48 篇）

- [OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_OccluGaussian_Occlusion-Aware_Gaussian_Splatting_for_Large_Scene_Reconstruction_and_Rendering_ICCV_2025_paper.pdf) | per-scene优化 | 遮挡感知大场景重建，自动剔除遮挡物
- [ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_ResGS_Residual_Densification_of_3D_Gaussian_for_Efficient_Detail_Recovery_ICCV_2025_paper.pdf) | per-scene优化 | 残差稠密化高效恢复高频细节
- [A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](https://openaccess.thecvf.com/content/ICCV2025/papers/Peng_A_Constrained_Optimization_Approach_for_Gaussian_Splatting_from_Coarsely-posed_Images_ICCV_2025_paper.pdf) | per-scene优化 | 粗位姿+噪声LiDAR的约束优化重建
- [3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](https://openaccess.thecvf.com/content/ICCV2025/papers/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.pdf) | per-scene优化 | Levenberg-Marquardt加速高斯优化收敛
- [RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](https://openaccess.thecvf.com/content/ICCV2025/papers/Cheng_RegGS_Unposed_Sparse_Views_Gaussian_Splatting_with_3DGS_Registration_ICCV_2025_paper.pdf) | per-scene优化 | 无位姿稀疏视图重建+3DGS配准
- [No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_No_Pose_at_All_Self-Supervised_Pose-Free_3D_Gaussian_Splatting_from_ICCV_2025_paper.pdf) | per-scene优化 | 自监督免位姿稀疏视图3DGS
- [LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](https://openaccess.thecvf.com/content/ICCV2025/papers/Lin_LongSplat_Robust_Unposed_3D_Gaussian_Splatting_for_Casual_Long_Videos_ICCV_2025_paper.pdf) | per-scene优化 | 免位姿长视频稳健高斯重建
- [GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments](https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_GaussianUpdate_Continual_3D_Gaussian_Splatting_Update_for_Changing_Environments_ICCV_2025_paper.pdf) | per-scene优化 | 环境变化下的持续3DGS增量更新
- [EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_EvaGaussians_Event_Stream_Assisted_Gaussian_Splatting_from_Blurry_Images_ICCV_2025_paper.pdf) | per-scene优化 | 事件流辅助模糊图像重建
- [Liberated-GS: 3D Gaussian Splatting Independent from SfM Point Clouds](https://openaccess.thecvf.com/content/ICCV2025/papers/Pan_Liberated-GS_3D_Gaussian_Splatting_Independent_from_SfM_Point_Clouds_ICCV_2025_paper.pdf) | per-scene优化 | 摆脱SfM点云初始化的依赖
- [Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Self-Ensembling_Gaussian_Splatting_for_Few-Shot_Novel_View_Synthesis_ICCV_2025_paper.pdf) | per-scene优化 | 自集成缓解少视图过拟合
- [Splat-based 3D Scene Reconstruction with Extreme Motion-blur](https://openaccess.thecvf.com/content/ICCV2025/papers/Jang_Splat-based_3D_Scene_Reconstruction_with_Extreme_Motion-blur_ICCV_2025_paper.pdf) | per-scene优化 | 极端运动模糊下的场景重建
- [Tile-wise vs. Image-wise: Random-Tile Loss and Training Paradigm for Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Tile-wise_vs._Image-wise_Random-Tile_Loss_and_Training_Paradigm_for_Gaussian_ICCV_2025_paper.pdf) | per-scene优化 | 随机tile损失改进训练范式
- [Faster and Better 3D Splatting via Group Training](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Faster_and_Better_3D_Splatting_via_Group_Training_ICCV_2025_paper.pdf) | per-scene优化 | 分组训练实现更快更优splatting
- [RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS](https://openaccess.thecvf.com/content/ICCV2025/papers/Fu_RobustSplat_Decoupling_Densification_and_Dynamics_for_Transient-Free_3DGS_ICCV_2025_paper.pdf) | per-scene优化 | 稠密化/动态解耦剔除瞬态干扰
- [OCSplats: Observation Completeness Quantification and Label Noise Separation in 3DGS](https://openaccess.thecvf.com/content/ICCV2025/papers/Ling_OCSplats_Observation_Completeness_Quantification_and_Label_Noise_Separation_in_3DGS_ICCV_2025_paper.pdf) | per-scene优化 | 观测完备性量化与标签噪声分离
- [PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_PCR-GS_COLMAP-Free_3D_Gaussian_Splatting_via_Pose_Co-Regularizations_ICCV_2025_paper.pdf) | per-scene优化 | 免COLMAP位姿协同正则重建
- [FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_FreeSplatter_Pose-free_Gaussian_Splatting_for_Sparse-view_3D_Reconstruction_ICCV_2025_paper.pdf) | per-scene优化 | 免位姿稀疏视图高斯重建
- [RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](https://openaccess.thecvf.com/content/ICCV2025/papers/Paliwal_RI3D_Few-Shot_Gaussian_Splatting_With_Repair_and_Inpainting_Diffusion_Priors_ICCV_2025_paper.pdf) | per-scene优化 | 修复+修补扩散先验少视图重建
- [RogSplat: Robust Gaussian Splatting via Generative Priors](https://openaccess.thecvf.com/content/ICCV2025/papers/Kong_RogSplat_Robust_Gaussian_Splatting_via_Generative_Priors_ICCV_2025_paper.pdf) | per-scene优化 | 生成先验增强的鲁棒重建
- [NeRF Is a Valuable Assistant for 3D Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Fang_NeRF_Is_a_Valuable_Assistant_for_3D_Gaussian_Splatting_ICCV_2025_paper.pdf) | per-scene优化 | NeRF辅助3DGS提升渲染质量
- [Self-Calibrating Gaussian Splatting for Large Field-of-View Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Deng_Self-Calibrating_Gaussian_Splatting_for_Large_Field-of-View_Reconstruction_ICCV_2025_paper.pdf) | per-scene优化 | 大视场自标定高斯重建
- [Street Gaussians without 3D Object Tracker](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Street_Gaussians_without_3D_Object_Tracker_ICCV_2025_paper.pdf) | per-scene优化 | 免3D跟踪器的街景高斯重建
- [Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Yuan_Robust_and_Efficient_3D_Gaussian_Splatting_for_Urban_Scene_Reconstruction_ICCV_2025_paper.pdf) | per-scene优化 | 城市级稳健高效高斯重建
- [Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Momentum-GS_Momentum_Gaussian_Self-Distillation_for_High-Quality_Large_Scene_Reconstruction_ICCV_2025_paper.pdf) | per-scene优化 | 动量自蒸馏提升大场景质量
- [HUG: Hierarchical Urban Gaussian Splatting with Block-Based Reconstruction for Large-Scale Aerial Scenes](https://openaccess.thecvf.com/content/ICCV2025/papers/Su_HUG_Hierarchical_Urban_Gaussian_Splatting_with_Block-Based_Reconstruction_for_Large-Scale_ICCV_2025_paper.pdf) | per-scene优化 | 分块层次化大尺度航拍场景重建
- [Seam360GS: Seamless 360deg Gaussian Splatting from Real-World Omnidirectional Images](https://openaccess.thecvf.com/content/ICCV2025/papers/Shin_Seam360GS_Seamless_360deg_Gaussian_Splatting_from_Real-World_Omnidirectional_Images_ICCV_2025_paper.pdf) | per-scene优化 | 全景图无缝360°高斯重建
- [AccidentalGS: 3D Gaussian Splatting from Accidental Camera Motion](https://openaccess.thecvf.com/content/ICCV2025/papers/Mao_AccidentalGS_3D_Gaussian_Splatting_from_Accidental_Camera_Motion_ICCV_2025_paper.pdf) | per-scene优化 | 意外相机运动下的场景重建
- [CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_CoMoGaussian_Continuous_Motion-Aware_Gaussian_Splatting_from_Motion-Blurred_Images_ICCV_2025_paper.pdf) | per-scene优化 | 运动模糊图像的连续运动建模
- [Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Quadratic_Gaussian_Splatting_High_Quality_Surface_Reconstruction_with_Second-order_Geometric_ICCV_2025_paper.pdf) | per-scene优化 | 二阶几何原语高质量表面重建
- [SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_SurfaceSplat_Connecting_Surface_Reconstruction_and_Gaussian_Splatting_ICCV_2025_paper.pdf) | per-scene优化 | 统一表面重建与高斯泼溅
- [QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_QuickSplat_Fast_3D_Surface_Reconstruction_via_Learned_Gaussian_Initialization_ICCV_2025_paper.pdf) | per-scene优化 | 学习高斯初始化加速表面重建
- [PolGS: Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Han_PolGS_Polarimetric_Gaussian_Splatting_for_Fast_Reflective_Surface_Reconstruction_ICCV_2025_paper.pdf) | per-scene优化 | 偏振高斯快速反光表面重建
- [MGSR: 2D/3D Mutual-boosted Gaussian Splatting for High-fidelity Surface Reconstruction under Various Light Conditions](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_MGSR_2D3D_Mutual-boosted_Gaussian_Splatting_for_High-fidelity_Surface_Reconstruction_under_ICCV_2025_paper.pdf) | per-scene优化 | 2D/3D互增强多光照表面重建
- [2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_2D_Gaussian_Splatting-based_Sparse-view_Transparent_Object_Depth_Reconstruction_via_Physics_ICCV_2025_paper.pdf) | per-scene优化 | 透明物体深度重建与场景更新
- [SplArt: Articulation Estimation and Part-Level Reconstruction with 3D Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Lin_SplArt_Articulation_Estimation_and_Part-Level_Reconstruction_with_3D_Gaussian_Splatting_ICCV_2025_paper.pdf) | per-scene优化 | 关节估计与部件级高斯重建
- [CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](https://openaccess.thecvf.com/content/ICCV2025/papers/Ackermann_CL-Splats_Continual_Learning_of_Gaussian_Splatting_with_Local_Optimization_ICCV_2025_paper.pdf) | per-scene优化 | 局部优化的3DGS持续学习
- [SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_SVG-Head_Hybrid_Surface-Volumetric_Gaussians_for_High-Fidelity_Head_Reconstruction_and_Real-Time_ICCV_2025_paper.pdf) | per-scene优化 | 表面-体积混合高斯头部重建
- [Fine-Grained 3D Gaussian Head Avatars Modeling from Static Captures via Joint Reconstruction and Registration](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Fine-Grained_3D_Gaussian_Head_Avatars_Modeling_from_Static_Captures_via_ICCV_2025_paper.pdf) | per-scene优化 | 静态采集联合重建配准头像
- [GeoAvatar: Adaptive Geometrical Gaussian Splatting for 3D Head Avatar](https://openaccess.thecvf.com/content/ICCV2025/papers/Moon_GeoAvatar_Adaptive_Geometrical_Gaussian_Splatting_for_3D_Head_Avatar_ICCV_2025_paper.pdf) | per-scene优化 | 自适应几何高斯头部头像
- [SEHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SEHDR_Single-Exposure_HDR_Novel_View_Synthesis_via_3D_Gaussian_Bracketing_ICCV_2025_paper.pdf) | per-scene优化 | 单曝光LDR合成HDR新视角
- [RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes](https://openaccess.thecvf.com/content/ICCV2025/papers/Kung_RadarSplat_Radar_Gaussian_Splatting_for_High-Fidelity_Data_Synthesis_and_3D_ICCV_2025_paper.pdf) | per-scene优化 | 雷达高斯驾驶场景重建合成
- [InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_InsideOut_Integrated_RGB-Radiative_Gaussian_Splatting_for_Comprehensive_3D_Object_Representation_ICCV_2025_paper.pdf) | per-scene优化 | RGB+辐射高斯完整物体表示
- [AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_AD-GS_Object-Aware_B-Spline_Gaussian_Splatting_for_Self-Supervised_Autonomous_Driving_ICCV_2025_paper.pdf) | per-scene优化 | B样条物体感知自动驾驶重建
- [S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_S3R-GS_Streamlining_the_Pipeline_for_Large-Scale_Street_Scene_Reconstruction_ICCV_2025_paper.pdf) | per-scene优化 | 简化大规模街景重建流水线
- [RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors](https://openaccess.thecvf.com/content/ICCV2025/papers/Du_RGE-GS_Reward-Guided_Expansive_Driving_Scene_Reconstruction_via_Diffusion_Priors_ICCV_2025_paper.pdf) | per-scene优化 | 奖励引导扩散先验驾驶重建
- [CityGS-X: A Scalable Architecture for Efficient and Geometrically Accurate Large-Scale Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_CityGS-X_A_Scalable_Architecture_for_Efficient_and_Geometrically_Accurate_Large-Scale_ICCV_2025_paper.pdf) | per-scene优化 | 并行分层架构城市场景重建
- [ExploreGS: Explorable 3D Scene Reconstruction with Virtual Camera Samplings and Diffusion Priors](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_ExploreGS_Explorable_3D_Scene_Reconstruction_with_Virtual_Camera_Samplings_and_ICCV_2025_paper.pdf) | per-scene优化 | 虚拟相机采样补全可探索重建

### 单图/少图生成（12 篇）

- [MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Dong_MoGA_3D_Generative_Avatar_Prior_for_Monocular_Gaussian_Avatar_Reconstruction_ICCV_2025_paper.pdf) | 单图/少图生成 | 单目高斯头像生成式先验重建
- [SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](https://openaccess.thecvf.com/content/ICCV2025/papers/Schoneveld_SHeaP_Self-Supervised_Head_Geometry_Predictor_Learned_via_2D_Gaussians_ICCV_2025_paper.pdf) | 单图/少图生成 | 单目自监督头部几何预测
- [HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_HairCUP_Hair_Compositional_Universal_Prior_for_3D_Gaussian_Avatars_ICCV_2025_paper.pdf) | 单图/少图生成 | 头发组合通用先验高斯头像
- [SIGMAN: Scaling 3D Human Gaussian Generation with Millions of Assets](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_SIGMAN_Scaling_3D_Human_Gaussian_Generation_with_Millions_of_Assets_ICCV_2025_paper.pdf) | 单图/少图生成 | 百万资产规模人体高斯生成
- [Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Cai_Baking_Gaussian_Splatting_into_Diffusion_Denoiser_for_Fast_and_Scalable_ICCV_2025_paper.pdf) | 单图/少图生成 | 扩散去噪器内置GS单阶段图生3D
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiang_Repurposing_2D_Diffusion_Models_with_Gaussian_Atlas_for_3D_Generation_ICCV_2025_paper.pdf) | 单图/少图生成 | 高斯图集复用2D扩散做3D生成
- [Generative Gaussian Splatting: Generating 3D Scenes with Video Diffusion Priors](https://openaccess.thecvf.com/content/ICCV2025/papers/Schwarz_Generative_Gaussian_Splatting_Generating_3D_Scenes_with_Video_Diffusion_Priors_ICCV_2025_paper.pdf) | 单图/少图生成 | 视频扩散先验生成3D场景
- [MiDSummer: Multi-Guidance Diffusion for Controllable Zero-Shot Immersive Gaussian Splatting Scene Generation](https://openaccess.thecvf.com/content/ICCV2025/papers/Hu_MiDSummer_Multi-Guidance_Diffusion_for_Controllable_Zero-Shot_Immersive_Gaussian_Splatting_Scene_ICCV_2025_paper.pdf) | 单图/少图生成 | 多引导扩散可控沉浸式场景生成
- [VideoRFSplat: Direct Scene-Level Text-to-3D Gaussian Splatting Generation with Flexible Pose and Multi-View Joint Modeling](https://openaccess.thecvf.com/content/ICCV2025/papers/Go_VideoRFSplat_Direct_Scene-Level_Text-to-3D_Gaussian_Splatting_Generation_with_Flexible_Pose_ICCV_2025_paper.pdf) | 单图/少图生成 | 文本直出场景级3DGS生成
- [A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](https://openaccess.thecvf.com/content/ICCV2025/papers/Peng_A_Lesson_in_Splats_Teacher-Guided_Diffusion_for_3D_Gaussian_Splats_ICCV_2025_paper.pdf) | 单图/少图生成 | 教师引导扩散2D监督生成高斯
- [GSV3D: Gaussian Splatting-based Geometric Distillation with Stable Video Diffusion for Single-Image 3D Object Generation](https://openaccess.thecvf.com/content/ICCV2025/papers/Tao_GSV3D_Gaussian_Splatting-based_Geometric_Distillation_with_Stable_Video_Diffusion_for_ICCV_2025_paper.pdf) | 单图/少图生成 | 视频扩散蒸馏单图3D物体生成
- [Hi-Gaussian: Hierarchical Gaussians under Normalized Spherical Projection for Single-View 3D Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Xie_Hi-Gaussian_Hierarchical_Gaussians_under_Normalized_Spherical_Projection_for_Single-View_3D_ICCV_2025_paper.pdf) | 单图/少图生成 | 球面投影层次高斯单视图重建

### 压缩与存储（3 篇）

- [Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_Can3Tok_Canonical_3D_Tokenization_and_Latent_Modeling_of_Scene-Level_3D_ICCV_2025_paper.pdf) | 压缩与存储 | 场景级3D高斯的规范token化建模
- [ContraGS: Codebook-Condensed and Trainable Gaussian Splatting for Fast, Memory-Efficient Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Durvasula_ContraGS_Codebook-Condensed_and_Trainable_Gaussian_Splatting_for_Fast_Memory-Efficient_Reconstruction_ICCV_2025_paper.pdf) | 压缩与存储 | 码本压缩高斯快速省内存重建
- [Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Compression_of_3D_Gaussian_Splatting_with_Optimized_Feature_Planes_and_ICCV_2025_paper.pdf) | 压缩与存储 | 特征平面+视频编解码压缩3DGS

### 动态/4D（20 篇）

- [7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_7DGS_Unified_Spatial-Temporal-Angular_Gaussian_Splatting_ICCV_2025_paper.pdf) | 动态/4D | 统一空间-时间-视角7D高斯
- [MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_MEGA_Memory-Efficient_4D_Gaussian_Splatting_for_Dynamic_Scenes_ICCV_2025_paper.pdf) | 动态/4D | 省内存4D高斯动态场景表示
- [GaussianVideo: Efficient Video Representation via Hierarchical Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Bond_GaussianVideo_Efficient_Video_Representation_via_Hierarchical_Gaussian_Splatting_ICCV_2025_paper.pdf) | 动态/4D | 层次高斯高效视频表示
- [EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_EMD_Explicit_Motion_Modeling_for_High-Quality_Street_Gaussian_Splatting_ICCV_2025_paper.pdf) | 动态/4D | 显式运动建模高质量街景高斯
- [BezierGS: Dynamic Urban Scene Reconstruction with Bezier Curve Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_BezierGS_Dynamic_Urban_Scene_Reconstruction_with_Bezier_Curve_Gaussian_Splatting_ICCV_2025_paper.pdf) | 动态/4D | 贝塞尔曲线高斯动态城市场景
- [CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving](https://openaccess.thecvf.com/content/ICCV2025/papers/Song_CoDa-4DGS_Dynamic_Gaussian_Splatting_with_Context_and_Deformation_Awareness_for_ICCV_2025_paper.pdf) | 动态/4D | 上下文变形感知驾驶4DGS
- [TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_TimeFormer_Capturing_Temporal_Relationships_of_Deformable_3D_Gaussians_for_Robust_ICCV_2025_paper.pdf) | 动态/4D | 时序关系建模稳健形变高斯重建
- [Hierarchy UGP: Hierarchy Unified Gaussian Primitive for Large-Scale Dynamic Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Hierarchy_UGP_Hierarchy_Unified_Gaussian_Primitive_for_Large-Scale_Dynamic_Scene_ICCV_2025_paper.pdf) | 动态/4D | 层次统一原语大尺度动态重建
- [DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_DeGauss_Dynamic-Static_Decomposition_with_Gaussian_Splatting_for_Distractor-free_3D_Reconstruction_ICCV_2025_paper.pdf) | 动态/4D | 动静分解剔除干扰物的重建
- [MaGS: Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_MaGS_Reconstructing_and_Simulating_Dynamic_3D_Objects_with_Mesh-adsorbed_Gaussian_ICCV_2025_paper.pdf) | 动态/4D | 网格吸附高斯动态物体重建仿真
- [TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_TRACE_Learning_3D_Gaussian_Physical_Dynamics_from_Multi-view_Videos_ICCV_2025_paper.pdf) | 动态/4D | 多视图视频学习高斯物理动力学
- [Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Event-boosted_Deformable_3D_Gaussians_for_Dynamic_Scene_Reconstruction_ICCV_2025_paper.pdf) | 动态/4D | 事件增强形变高斯动态重建
- [STD-GS: Exploring Frame-Event Interaction for SpatioTemporal-Disentangled Gaussian Splatting to Reconstruct High-Dynamic Scene](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_STD-GS_Exploring_Frame-Event_Interaction_for_SpatioTemporal-Disentangled_Gaussian_Splatting_to_Reconstruct_ICCV_2025_paper.pdf) | 动态/4D | 帧-事件交互时空解耦高动态重建
- [InteractAvatar: Modeling Hand-Face Interaction in Photorealistic Avatars with Deformable Gaussians](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_InteractAvatar_Modeling_Hand-Face_Interaction_in_Photorealistic_Avatars_with_Deformable_Gaussians_ICCV_2025_paper.pdf) | 动态/4D | 手脸交互可变形高斯头像
- [Sequential Gaussian Avatars with Hierarchical Motion Context](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Sequential_Gaussian_Avatars_with_Hierarchical_Motion_Context_ICCV_2025_paper.pdf) | 动态/4D | 层次运动上下文序列高斯头像
- [DGTalker: Disentangled Generative Latent Space Learning for Audio-Driven Gaussian Talking Heads](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_DGTalker_Disentangled_Generative_Latent_Space_Learning_for_Audio-Driven_Gaussian_Talking_ICCV_2025_paper.pdf) | 动态/4D | 音频驱动高斯说话人头像
- [GaussianSpeech: Audio-Driven Personalized 3D Gaussian Avatars](https://openaccess.thecvf.com/content/ICCV2025/papers/Aneja_GaussianSpeech_Audio-Driven_Personalized_3D_Gaussian_Avatars_ICCV_2025_paper.pdf) | 动态/4D | 音频驱动个性化高斯头像
- [GGTalker: Talking Head Systhesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](https://openaccess.thecvf.com/content/ICCV2025/papers/Hu_GGTalker_Talking_Head_Systhesis_with_Generalizable_Gaussian_Priors_and_Identity-Specific_ICCV_2025_paper.pdf) | 动态/4D | 泛化先验+身份适配说话人头像
- [Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Gaussian_Variation_Field_Diffusion_for_High-fidelity_Video-to-4D_Synthesis_ICCV_2025_paper.pdf) | 动态/4D | 高斯变分场扩散视频转4D
- [LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_LocalDyGS_Multi-view_Global_Dynamic_Scene_Modeling_via_Adaptive_Local_Implicit_ICCV_2025_paper.pdf) | 动态/4D | 局部隐式解耦多视图动态建模

### 编辑与重光照（13 篇）

- [SU-RGS: Relightable 3D Gaussian Splatting from Sparse Views under Unconstrained Illuminations](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_SU-RGS_Relightable_3D_Gaussian_Splatting_from_Sparse_Views_under_Unconstrained_ICCV_2025_paper.pdf) | 编辑与重光照 | 无约束光照稀疏视图可重光照3DGS
- [GaRe: Relightable 3D Gaussian Splatting for Outdoor Scenes from Unconstrained Photo Collections](https://openaccess.thecvf.com/content/ICCV2025/papers/Bai_GaRe_Relightable_3D_Gaussian_Splatting_for_Outdoor_Scenes_from_Unconstrained_ICCV_2025_paper.pdf) | 编辑与重光照 | 户外无约束照片集可重光照
- [GS-ID: Illumination Decomposition on Gaussian Splatting via Adaptive Light Aggregation and Diffusion-Guided Material Priors](https://openaccess.thecvf.com/content/ICCV2025/papers/Du_GS-ID_Illumination_Decomposition_on_Gaussian_Splatting_via_Adaptive_Light_Aggregation_ICCV_2025_paper.pdf) | 编辑与重光照 | 自适应光聚合光照分解+材质先验
- [GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](https://openaccess.thecvf.com/content/ICCV2025/papers/Ye_GeoSplatting_Towards_Geometry_Guided_Gaussian_Splatting_for_Physically-based_Inverse_Rendering_ICCV_2025_paper.pdf) | 编辑与重光照 | 几何引导物理逆渲染3DGS
- [GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR](https://openaccess.thecvf.com/content/ICCV2025/papers/Bolduc_GaSLight_Gaussian_Splats_for_Spatially-Varying_Lighting_in_HDR_ICCV_2025_paper.pdf) | 编辑与重光照 | HDR空间变化光照高斯表示
- [Tune-Your-Style: Intensity-tunable 3D Style Transfer with Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Tune-Your-Style_Intensity-tunable_3D_Style_Transfer_with_Gaussian_Splatting_ICCV_2025_paper.pdf) | 编辑与重光照 | 强度可调3D风格迁移
- [A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Fang_A3GS_Arbitrary_Artistic_Style_into_Arbitrary_3D_Gaussian_Splatting_ICCV_2025_paper.pdf) | 编辑与重光照 | 任意艺术风格注入任意3DGS
- [InterGSEdit: Interactive 3D Gaussian Splatting Editing with 3D Geometry-Consistent Attention Prior](https://openaccess.thecvf.com/content/ICCV2025/papers/Wen_InterGSEdit_Interactive_3D_Gaussian_Splatting_Editing_with_3D_Geometry-Consistent_Attention_ICCV_2025_paper.pdf) | 编辑与重光照 | 几何一致注意力先验交互编辑
- [Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Robust_3D-Masked_Part-level_Editing_in_3D_Gaussian_Splatting_with_Regularized_ICCV_2025_paper.pdf) | 编辑与重光照 | 正则SDS稳健部件级3D编辑
- [Perspective-aware 3D Gaussian Inpainting with Multi-view Consistency](https://openaccess.thecvf.com/content/ICCV2025/papers/Cheng_Perspective-aware_3D_Gaussian_Inpainting_with_Multi-view_Consistency_ICCV_2025_paper.pdf) | 编辑与重光照 | 多视图一致透视感知3D修补
- [GauUpdate: New Object Insertion in 3D Gaussian Fields with Consistent Global Illumination](https://openaccess.thecvf.com/content/ICCV2025/papers/Ren_GauUpdate_New_Object_Insertion_in_3D_Gaussian_Fields_with_Consistent_ICCV_2025_paper.pdf) | 编辑与重光照 | 全局光照一致的新物体插入
- [Gaussian Splatting with Discretized SDF for Relightable Assets](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_Gaussian_Splatting_with_Discretized_SDF_for_Relightable_Assets_ICCV_2025_paper.pdf) | 编辑与重光照 | 离散SDF可重光照资产表示
- [GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_GazeGaussian_High-Fidelity_Gaze_Redirection_with_3D_Gaussian_Splatting_ICCV_2025_paper.pdf) | 编辑与重光照 | 双流3DGS高保真视线重定向

### SLAM/定位建图（9 篇）

- [Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](https://openaccess.thecvf.com/content/ICCV2025/papers/Giacomini_Splat-LOAM_Gaussian_Splatting_LiDAR_Odometry_and_Mapping_ICCV_2025_paper.pdf) | SLAM/定位建图 | 高斯LiDAR里程计与建图
- [DyGS-SLAM: Real-Time Accurate Localization and Gaussian Reconstruction for Dynamic Scenes](https://openaccess.thecvf.com/content/ICCV2025/papers/Hu_DyGS-SLAM_Real-Time_Accurate_Localization_and_Gaussian_Reconstruction_for_Dynamic_Scenes_ICCV_2025_paper.pdf) | SLAM/定位建图 | 动态场景实时定位与高斯重建
- [4D Gaussian Splatting SLAM](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_4D_Gaussian_Splatting_SLAM_ICCV_2025_paper.pdf) | SLAM/定位建图 | 4D高斯实时SLAM
- [GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Xie_GS-LIVM_Real-Time_Photo-Realistic_LiDAR-Inertial-Visual_Mapping_with_Gaussian_Splatting_ICCV_2025_paper.pdf) | SLAM/定位建图 | LiDAR-惯性-视觉实时建图
- [SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding](https://openaccess.thecvf.com/content/ICCV2025/papers/Wen_SEGS-SLAM_Structure-enhanced_3D_Gaussian_Splatting_SLAM_with_Appearance_Embedding_ICCV_2025_paper.pdf) | SLAM/定位建图 | 结构增强高斯SLAM+外观嵌入
- [Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps](https://openaccess.thecvf.com/content/ICCV2025/papers/Cheng_Outdoor_Monocular_SLAM_with_Global_Scale-Consistent_3D_Gaussian_Pointmaps_ICCV_2025_paper.pdf) | SLAM/定位建图 | 全局尺度一致户外单目SLAM
- [ToF-Splatting: Dense SLAM using Sparse Time-of-Flight Depth and Multi-Frame Integration](https://openaccess.thecvf.com/content/ICCV2025/papers/Conti_ToF-Splatting_Dense_SLAM_using_Sparse_Time-of-Flight_Depth_and_Multi-Frame_Integration_ICCV_2025_paper.pdf) | SLAM/定位建图 | 稀疏ToF多帧融合稠密SLAM
- [6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Jin_6DOPE-GS_Online_6D_Object_Pose_Estimation_using_Gaussian_Splatting_ICCV_2025_paper.pdf) | SLAM/定位建图 | 在线6D物体位姿估计
- [IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_IGL-Nav_Incremental_3D_Gaussian_Localization_for_Image-goal_Navigation_ICCV_2025_paper.pdf) | SLAM/定位建图 | 增量高斯定位图像目标导航

### 语义/分割/理解（21 篇）

- [SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](https://openaccess.thecvf.com/content/ICCV2025/papers/Sheng_SpatialSplat_Efficient_Semantic_3D_from_Sparse_Unposed_Images_ICCV_2025_paper.pdf) | 语义/分割/理解 | 稀疏无位姿图像高效语义3D
- [CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiao_CLIP-GS_Unifying_Vision-Language_Representation_with_3D_Gaussian_Splatting_ICCV_2025_paper.pdf) | 语义/分割/理解 | 统一视觉语言表示与3DGS
- [CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Tian_CCL-LGS_Contrastive_Codebook_Learning_for_3D_Language_Gaussian_Splatting_ICCV_2025_paper.pdf) | 语义/分割/理解 | 对比码本学习语言高斯泼溅
- [Identity-aware Language Gaussian Splatting for Open-vocabulary 3D Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2025/papers/Jang_Identity-aware_Language_Gaussian_Splatting_for_Open-vocabulary_3D_Semantic_Segmentation_ICCV_2025_paper.pdf) | 语义/分割/理解 | 身份感知开放词汇3D语义分割
- [SplatTalk: 3D VQA with Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Thai_SplatTalk_3D_VQA_with_Gaussian_Splatting_ICCV_2025_paper.pdf) | 语义/分割/理解 | 3DGS场景视觉问答
- [SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SceneSplat_Gaussian_Splatting-based_Scene_Understanding_with_Vision-Language_Pretraining_ICCV_2025_paper.pdf) | 语义/分割/理解 | 视觉语言预训练场景理解
- [VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VoteSplat_Hough_Voting_Gaussian_Splatting_for_3D_Scene_Understanding_ICCV_2025_paper.pdf) | 语义/分割/理解 | 霍夫投票高斯场景理解
- [AG2aussian: Anchor-Graph Structured Gaussian Splatting for Instance-Level 3D Scene Understanding and Editing](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_AG2aussian_Anchor-Graph_Structured_Gaussian_Splatting_for_Instance-Level_3D_Scene_Understanding_ICCV_2025_paper.pdf) | 语义/分割/理解 | 锚图结构实例级理解与编辑
- [LUDVIG: Learning-Free Uplifting of 2D Visual Features to Gaussian Splatting Scenes](https://openaccess.thecvf.com/content/ICCV2025/papers/Marrie_LUDVIG_Learning-Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_ICCV_2025_paper.pdf) | 语义/分割/理解 | 免学习2D特征提升到高斯场景
- [Online Language Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Katragadda_Online_Language_Splatting_ICCV_2025_paper.pdf) | 语义/分割/理解 | 在线语言场增量映射
- [AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AutoOcc_Automatic_Open-Ended_Semantic_Occupancy_Annotation_via_Vision-Language_Guided_Gaussian_ICCV_2025_paper.pdf) | 语义/分割/理解 | 视觉语言引导开放语义占用标注
- [3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_3D_Gaussian_Map_with_Open-Set_Semantic_Grouping_for_Vision-Language_Navigation_ICCV_2025_paper.pdf) | 语义/分割/理解 | 开放集语义分组视觉语言导航
- [ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_ObjectGS_Object-aware_Scene_Reconstruction_and_Scene_Understanding_via_Gaussian_Splatting_ICCV_2025_paper.pdf) | 语义/分割/理解 | 物体感知场景重建与理解
- [Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing](https://openaccess.thecvf.com/content/ICCV2025/papers/Shen_Trace3D_Consistent_Segmentation_Lifting_via_Gaussian_Instance_Tracing_ICCV_2025_paper.pdf) | 语义/分割/理解 | 高斯实例追踪一致分割提升
- [Communication-Efficient Multi-Vehicle Collaborative Semantic Segmentation via Sparse 3D Gaussian Sharing](https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_Communication-Efficient_Multi-Vehicle_Collaborative_Semantic_Segmentation_via_Sparse_3D_Gaussian_Sharing_ICCV_2025_paper.pdf) | 语义/分割/理解 | 稀疏高斯共享多车协作分割
- [Joint Semantic and Rendering Enhancements in 3D Gaussian Modeling with Anisotropic Local Encoding](https://openaccess.thecvf.com/content/ICCV2025/papers/He_Joint_Semantic_and_Rendering_Enhancements_in_3D_Gaussian_Modeling_with_ICCV_2025_paper.pdf) | 语义/分割/理解 | 各向异性局部编码联合语义渲染
- [Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_Self-supervised_Learning_of_Hybrid_Part-aware_3D_Representations_of_2D_Gaussians_ICCV_2025_paper.pdf) | 语义/分割/理解 | 2D高斯+超二次曲面部件表示
- [GaussianOcc: Fully Self-supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Gan_GaussianOcc_Fully_Self-supervised_and_Efficient_3D_Occupancy_Estimation_with_Gaussian_ICCV_2025_paper.pdf) | 语义/分割/理解 | 全自监督高效3D占用估计
- [GaussRender: Learning 3D Occupancy with Gaussian Rendering](https://openaccess.thecvf.com/content/ICCV2025/papers/Chambon_GaussRender_Learning_3D_Occupancy_with_Gaussian_Rendering_ICCV_2025_paper.pdf) | 语义/分割/理解 | 高斯渲染约束3D占用学习
- [GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](https://openaccess.thecvf.com/content/ICCV2025/papers/Boeder_GaussianFlowOcc_Sparse_and_Weakly_Supervised_Occupancy_Estimation_using_Gaussian_Splatting_ICCV_2025_paper.pdf) | 语义/分割/理解 | 时序流+高斯弱监督占用估计
- [GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Ye_GS-Occ3D_Scaling_Vision-only_Occupancy_Reconstruction_with_Gaussian_Splatting_ICCV_2025_paper.pdf) | 语义/分割/理解 | 纯视觉占用重建规模化扩展

### 渲染改进（5 篇）

- [StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Kheradmand_StochasticSplats_Stochastic_Rasterization_for_Sorting-Free_3D_Gaussian_Splatting_ICCV_2025_paper.pdf) | 渲染改进 | 随机光栅化免排序3DGS
- [RayGaussX: Accelerating Gaussian-Based Ray Marching for Real-Time and High-Quality Novel View Synthesis](https://openaccess.thecvf.com/content/ICCV2025/papers/Blanc_RayGaussX_Accelerating_Gaussian-Based_Ray_Marching_for_Real-Time_and_High-Quality_Novel_ICCV_2025_paper.pdf) | 渲染改进 | 高斯光线步进加速实时NVS
- [AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](https://openaccess.thecvf.com/content/ICCV2025/papers/Steiner_AAA-Gaussians_Anti-Aliased_and_Artifact-Free_3D_Gaussian_Rendering_ICCV_2025_paper.pdf) | 渲染改进 | 抗锯齿无伪影3D高斯渲染
- [Neural Shell Texture Splatting: More Details and Fewer Primitives](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Neural_Shell_Texture_Splatting_More_Details_and_Fewer_Primitives_ICCV_2025_paper.pdf) | 渲染改进 | 神经壳纹理少原语多细节
- [Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](https://openaccess.thecvf.com/content/ICCV2025/papers/Niedermayr_Lightweight_Gradient-Aware_Upscaling_of_3D_Gaussian_Splatting_Images_ICCV_2025_paper.pdf) | 渲染改进 | 梯度感知轻量渲染图超分

### 其他（21 篇）

- [Gaussian-based World Model: Gaussian Priors for Voxel-Based Occupancy Prediction and Future Motion Prediction](https://openaccess.thecvf.com/content/ICCV2025/papers/Feng_Gaussian-based_World_Model_Gaussian_Priors_for_Voxel-Based_Occupancy_Prediction_and_ICCV_2025_paper.pdf) | 其他 | 高斯世界模型占用+运动预测(存疑)
- [SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Ying_SketchSplat_3D_Edge_Reconstruction_via_Differentiable_Multi-view_Sketch_Splatting_ICCV_2025_paper.pdf) | 其他 | 草图线原语splatting重建(存疑)
- [BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](https://openaccess.thecvf.com/content/ICCV2025/papers/Svitov_BillBoard_Splatting_BBSplat_Learnable_Textured_Primitives_for_Novel_View_Synthesis_ICCV_2025_paper.pdf) | 其他 | 可学习贴片原语NVS(可替高斯)
- [Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_Curve-Aware_Gaussian_Splatting_for_3D_Parametric_Curve_Reconstruction_ICCV_2025_paper.pdf) | 其他 | 高斯泼溅重建3D参数曲线
- [Discretized Gaussian Representation for Tomographic Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_Discretized_Gaussian_Representation_for_Tomographic_Reconstruction_ICCV_2025_paper.pdf) | 其他 | 离散高斯直接CT体数据重建
- [GaussianReg: Rapid 2D/3D Registration for Emergency Surgery via Explicit 3D Modeling with Gaussian Primitives](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_GaussianReg_Rapid_2D3D_Registration_for_Emergency_Surgery_via_Explicit_3D_ICCV_2025_paper.pdf) | 其他 | 高斯原语手术2D/3D快速配准
- [X2-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_X2-Gaussian_4D_Radiative_Gaussian_Splatting_for_Continuous-time_Tomographic_Reconstruction_ICCV_2025_paper.pdf) | 其他 | 4D辐射高斯连续时间CT重建
- [RhythmGuassian: Repurposing Generalizable Gaussian Model For Remote Physiological Measurement](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_RhythmGuassian_Repurposing_Generalizable_Gaussian_Model_For_Remote_Physiological_Measurement_ICCV_2025_paper.pdf) | 其他 | 可泛化高斯模型测rPPG生理信号
- [GWM: Towards Scalable Gaussian World Models for Robotic Manipulation](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_GWM_Towards_Scalable_Gaussian_World_Models_for_Robotic_Manipulation_ICCV_2025_paper.pdf) | 其他 | 可扩展高斯世界模型机器人操作
- [GausSim: Foreseeing Reality by Gaussian Simulator for Elastic Objects](https://openaccess.thecvf.com/content/ICCV2025/papers/Shao_GausSim_Foreseeing_Reality_by_Gaussian_Simulator_for_Elastic_Objects_ICCV_2025_paper.pdf) | 其他 | 高斯核弹性物体物理仿真
- [GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_GaussianProperty_Integrating_Physical_Properties_to_3D_Gaussians_with_LMMs_ICCV_2025_paper.pdf) | 其他 | LMM赋予3D高斯物理属性
- [PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_PhysSplat_Efficient_Physics_Simulation_for_3D_Scenes_via_MLLM-Guided_Gaussian_ICCV_2025_paper.pdf) | 其他 | MLLM引导高斯场景物理仿真
- [GAP: Gaussianize Any Point Clouds with Text Guidance](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_GAP_Gaussianize_Any_Point_Clouds_with_Text_Guidance_ICCV_2025_paper.pdf) | 其他 | 文本引导无色彩点云高斯化
- [StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions](https://openaccess.thecvf.com/content/ICCV2025/papers/Ke_StealthAttack_Robust_3D_Gaussian_Splatting_Poisoning_via_Density-Guided_Illusions_ICCV_2025_paper.pdf) | 其他 | 密度引导幻觉3DGS投毒攻击
- [3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](https://openaccess.thecvf.com/content/ICCV2025/papers/Lou_3D_Gaussian_Splatting_Driven_Multi-View_Robust_Physical_Adversarial_Camouflage_Generation_ICCV_2025_paper.pdf) | 其他 | 3DGS驱动多视角物理对抗伪装
- [CULTURE3D: A Large-Scale and Diverse Dataset of Cultural Landmarks and Terrains for Gaussian-Based Scene Rendering](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_CULTURE3D_A_Large-Scale_and_Diverse_Dataset_of_Cultural_Landmarks_and_ICCV_2025_paper.pdf) | 其他 | 文化地标高斯渲染大规模数据集
- [VoluMe - Authentic 3D Video Calls from Live Gaussian Splat Prediction](https://openaccess.thecvf.com/content/ICCV2025/papers/de_La_Gorce_VoluMe_-_Authentic_3D_Video_Calls_from_Live_Gaussian_Splat_ICCV_2025_paper.pdf) | 其他 | 实时高斯预测3D视频通话
- [EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](https://openaccess.thecvf.com/content/ICCV2025/papers/Chhablani_EmbodiedSplat_Personalized_Real-to-Sim-to-Real_Navigation_with_Gaussian_Splats_from_a_Mobile_ICCV_2025_paper.pdf) | 其他 | 手机高斯捕捉个性化具身导航
- [Mixture of Experts Guided by Gaussian Splatters Matters: A new Approach to Weakly-Supervised Video Anomaly Detection](https://openaccess.thecvf.com/content/ICCV2025/papers/Amicantonio_Mixture_of_Experts_Guided_by_Gaussian_Splatters_Matters_A_new_ICCV_2025_paper.pdf) | 其他 | 高斯泼溅引导MoE视频异常检测(存疑)
- [Instant GaussianImage: A Generalizable and Self-Adaptive Image Representation via 2D Gaussian Splatting](https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.pdf) | 其他 | 2D高斯可泛化图像表示
- [Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Generalized_and_Efficient_2D_Gaussian_Splatting_for_Arbitrary-scale_Super-Resolution_ICCV_2025_paper.pdf) | 其他 | 2D高斯任意尺度图像超分

---

## 三、ECCV 2026 — 3DGS 论文清单（150 篇，链接全部验证通过）

链接为 ECCV 官方 virtual site 的 poster 页（含摘要与官方 PDF 跳转）。官方页面另将多数条目标注了主题 "3D Reconstruction, Gaussian Splatting & Neural Rendering"。

### 前馈重建(多视图→高斯)（18 篇）

- [AdaptiveSplat: Texture Aware Controllable 3D Gaussian Allocation for Feed-Forward Reconstruction](https://eccv.ecva.net/virtual/2026/poster/4240) | 前馈重建(多视图→高斯) | 纹理感知可控高斯分配前馈重建
- [AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3540) | 前馈重建(多视图→高斯) | 对齐与评分稳健前馈3DGS
- [GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens](https://eccv.ecva.net/virtual/2026/poster/5256) | 前馈重建(多视图→高斯) | 全局场景token高效前馈3DGS
- [BEV-GS: Feed-forward Gaussian Splatting in Bird’s-Eye-View for Road Reconstruction](https://eccv.ecva.net/virtual/2026/poster/4436) | 前馈重建(多视图→高斯) | 鸟瞰视角前馈道路重建
- [ReSplat: Learning Recurrent Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3147) | 前馈重建(多视图→高斯) | 循环网络学习高斯泼溅
- [SubSplat: High-Resolution Pixel-aligned 3DGS via Sub-pixel Gaussian Reparameterization](https://eccv.ecva.net/virtual/2026/poster/4274) | 前馈重建(多视图→高斯) | 亚像素重参数化高分辨率3DGS
- [VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction](https://eccv.ecva.net/virtual/2026/poster/3169) | 前馈重建(多视图→高斯) | 体素对齐预测的前馈3DGS
- [ReconSplat: Generalizable 3D Scene Reconstruction Beyond Observed Views](https://eccv.ecva.net/virtual/2026/poster/4425) | 前馈重建(多视图→高斯) | 超观测视角泛化场景重建
- [ViewSplat: View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis](https://eccv.ecva.net/virtual/2026/poster/3759) | 前馈重建(多视图→高斯) | 视角自适应动态高斯前馈合成
- [WildSplat: Feedforward Gaussian Splatting from Unposed In-the-Wild Images](https://eccv.ecva.net/virtual/2026/poster/4603) | 前馈重建(多视图→高斯) | 野外无位姿图像前馈重建
- [F⁴Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5023) | 前馈重建(多视图→高斯) | 前馈预测稠密化的前馈3DGS
- [AVSplat:Dense-View Feed-Forward 3D Gaussian Splatting with Assist-View Preconditioning](https://eccv.ecva.net/virtual/2026/poster/3993) | 前馈重建(多视图→高斯) | 辅助视角预处理稠密前馈3DGS
- [MAC-Splat: Multi-Attribute Consistency for High-Fidelity Sparse-View Reconstruction](https://eccv.ecva.net/virtual/2026/poster/4429) | 前馈重建(多视图→高斯) | 多属性一致性高保真稀疏重建
- [StereoGS: Sparse-View 3D Gaussian Splatting via Stereo Priors](https://eccv.ecva.net/virtual/2026/poster/3379) | 前馈重建(多视图→高斯) | 立体先验稀疏视图3DGS
- [StructSplat: Generalizable 3D Gaussian Splatting from Uncalibrated Sparse Views](https://eccv.ecva.net/virtual/2026/poster/3486) | 前馈重建(多视图→高斯) | 未标定稀疏视图泛化3DGS
- [One-Shot Feed-Forward 360° Animatable Avatar via Inpainted UV-Space Gaussian Modeling](https://eccv.ecva.net/virtual/2026/poster/4657) | 前馈重建(多视图→高斯) | UV空间修补一次前馈360头像
- [Instant Expressive Gaussian Head Avatars at Over 100 FPS](https://eccv.ecva.net/virtual/2026/poster/4074) | 前馈重建(多视图→高斯) | 百帧率即时表情高斯头像
- [MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5867) | 前馈重建(多视图→高斯) | 层次前馈高斯医学超分重建

### per-scene优化（41 篇）

- [TopoGS: Planar Reconstruction via Topology-Aware 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5769) | per-scene优化 | 拓扑感知平面重建
- [AnchorSplat: Fast and Structure Consistent Detail Synthesis for Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/4233) | per-scene优化 | 结构一致细节合成
- [InceptionGS: Generative Bootstrapping for Large-Scale Gaussian Splatting under Unstructured View Sampling](https://eccv.ecva.net/virtual/2026/poster/5029) | per-scene优化 | 生成式引导非结构化采样重建
- [Drop-In Perceptual Optimization for 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/4997) | per-scene优化 | 即插即用感知损失优化
- [MVGS: Multi-view Regulated Gaussian Splatting for Novel View Synthesis](https://eccv.ecva.net/virtual/2026/poster/4315) | per-scene优化 | 多视图正则高斯新视角合成
- [SharpGS: Sharpness-Preserving 3D Gaussian Splatting with Differentiable Blur-Driven Density Control](https://eccv.ecva.net/virtual/2026/poster/4346) | per-scene优化 | 可微模糊驱动密度控制保锐度
- [Geometric Probing for Isotropic Optimization Manifold in Sparse-View 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5303) | per-scene优化 | 稀疏视图各向同性优化流形探测
- [Incremental Online Scene Reconstruction by 3D Gaussian Triangulation](https://eccv.ecva.net/virtual/2026/poster/3228) | per-scene优化 | 高斯三角化在线增量重建
- [InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](https://eccv.ecva.net/virtual/2026/poster/3662) | per-scene优化 | 单次前向高斯HDR重建
- [PriSplat: Propagating Reliable Multi-view Information for Distractor-Free 3DGS](https://eccv.ecva.net/virtual/2026/poster/4019) | per-scene优化 | 多视图可靠信息传播剔除干扰
- [SupIR-GS: Thermal Infrared Super-Resolution Novel View Synthesis with Imaging-Calibrated 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5680) | per-scene优化 | 成像标定热红外超分NVS
- [ActiveStructure: Plane Scene Graph-Guided Active 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5184) | per-scene优化 | 平面场景图引导主动重建
- [Gaussians on Fire: High-Frequency Reconstruction of Flames](https://eccv.ecva.net/virtual/2026/poster/4870) | per-scene优化 | 火焰高频重建
- [From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians](https://eccv.ecva.net/virtual/2026/poster/5943) | per-scene优化 | 定向高斯高保真表面重建
- [Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis](https://eccv.ecva.net/virtual/2026/poster/5011) | per-scene优化 | 光度不确定性预测NVS
- [ReInGS: Re-Initializing 3D Gaussians against Sparsity Discrepancy in Few-Shot Novel View Synthesis](https://eccv.ecva.net/virtual/2026/poster/5651) | per-scene优化 | 重初始化对抗稀疏度差异
- [StructureGS: Structure-aware Gaussian Splatting for Articulated Object Reconstruction](https://eccv.ecva.net/virtual/2026/poster/4158) | per-scene优化 | 结构感知关节物体重建
- [Denoising-GS: Gaussian Splatting with Spatial-aware Denoising](https://eccv.ecva.net/virtual/2026/poster/4248) | per-scene优化 | 空间感知去噪的高斯泼溅
- [SA-ResGS: Self-Augmented Residual 3D Gaussian Splatting for Next Best View Selection](https://eccv.ecva.net/virtual/2026/poster/3511) | per-scene优化 | 自增强残差高斯下一最佳视角
- [Geometry-Propagated Gaussian Splatting for Aerial Sparse Novel View Synthesis](https://eccv.ecva.net/virtual/2026/poster/5360) | per-scene优化 | 几何传播航拍稀疏NVS
- [StratoSplat: Taming Layered Regularities for Sparse Aerial 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5641) | per-scene优化 | 分层规则稀疏航拍3DGS
- [LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision](https://eccv.ecva.net/virtual/2026/poster/4594) | per-scene优化 | 伪LiDAR监督外推视角合成
- [PaD-GS: Leveraging Distortion Map for Panoramic Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/4125) | per-scene优化 | 畸变图全景高斯泼溅
- [DPGS: A Diffusion-Prior Guided Framework for Large-Scale 3D Gaussian Splatting Reconstruction](https://eccv.ecva.net/virtual/2026/poster/4885) | per-scene优化 | 扩散先验引导大规模重建
- [Confidence-Based Mesh Extraction from 3D Gaussians](https://eccv.ecva.net/virtual/2026/poster/3645) | per-scene优化 | 置信度网格提取自3D高斯
- [WilLaGS: Latent-Conditional 3D Appearance Fields for Robust Gaussian Splatting In-the-Wild](https://eccv.ecva.net/virtual/2026/poster/5306) | per-scene优化 | 潜在条件外观场野外鲁棒GS
- [IndoorSplat: Enhanced Indoor Scene Reconstruction with Structured 2D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5326) | per-scene优化 | 结构化2D高斯室内重建
- [GaussianLens: Localized High-Resolution Reconstruction via On-Demand Gaussian Densification](https://eccv.ecva.net/virtual/2026/poster/4560) | per-scene优化 | 按需稠密化局部高分辨率重建
- [Large-Scale High-Quality 3D Gaussian Head Reconstruction from Multi-View Captures](https://eccv.ecva.net/virtual/2026/poster/3643) | per-scene优化 | 多视角大尺度高质量头部重建
- [DualDiff3D: Dual Structure-Appearance Diffusion Priors for Reliability-Enhanced 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/4077) | per-scene优化 | 结构-外观双扩散先验增强3DGS
- [Sparse-View Surface Reconstruction using Gaussian Splatting through High-Confidence Depth Propagation with Normal Priors](https://eccv.ecva.net/virtual/2026/poster/4097) | per-scene优化 | 高置信深度传播+法线先验表面重建
- [When 3D Gaussian Splatting Recovers Real Surfaces](https://eccv.ecva.net/virtual/2026/poster/5350) | per-scene优化 | 探究3DGS恢复真实表面条件
- [Physically Grounded Dual-Opacity Gaussian Splatting for Joint RGB-TIR Reconstruction](https://eccv.ecva.net/virtual/2026/poster/5920) | per-scene优化 | 双不透明度RGB-TIR联合重建
- [GlassGS: Geometry and Concept-Aware 3D Gaussian Splatting for Reflective Enclosures](https://eccv.ecva.net/virtual/2026/poster/4381) | per-scene优化 | 反射围护结构几何概念感知重建
- [EAGS: Error-Aware Gaussian Splatting with Dual-Confidence-Guided Modeling for Uncalibrated Driving Scenes](https://eccv.ecva.net/virtual/2026/poster/4526) | per-scene优化 | 误差感知双置信未标定驾驶重建
- [MagnetGS-Mesh: High-Quality Multi-Object Mesh Reconstruction via Adaptive Surface Optimization](https://eccv.ecva.net/virtual/2026/poster/5724) | per-scene优化 | 自适应表面优化多物体网格重建
- [URHead: A Unified UV-Space Representation for Joint Mesh–3DGS Optimization in Head Avatars](https://eccv.ecva.net/virtual/2026/poster/4747) | per-scene优化 | UV空间网格+3DGS联合头部优化
- [Improving Sparse-View 3DGS Generalization via Flat Minima Optimization](https://eccv.ecva.net/virtual/2026/poster/5368) | per-scene优化 | 平坦极小值优化提升稀疏视图泛化
- [LiteGS: a high-performance framework to train 3dgs in subminutes via system and algorithm codesign](https://eccv.ecva.net/virtual/2026/poster/3854) | per-scene优化 | 系统算法协同设计亚分钟训练
- [SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training](https://eccv.ecva.net/virtual/2026/poster/5008) | per-scene优化 | 稠密化后向跳过加速3DGS训练
- [SMP-UWGS: Coupled Physics-Geometry Optimization for Scalable Multi-Partition Underwater 3D Reconstruction](https://eccv.ecva.net/virtual/2026/poster/5565) | per-scene优化 | 物理-几何耦合水下多分区重建

### 单图/少图生成（8 篇）

- [Penetration-Free Compositional 3D Generation via Gaussian Surface Offset](https://eccv.ecva.net/virtual/2026/poster/5611) | 单图/少图生成 | 高斯表面偏移无穿透组合生成
- [Free-Range Gaussians: Non-Grid-Aligned Generative 3D Gaussian Reconstruction](https://eccv.ecva.net/virtual/2026/poster/3583) | 单图/少图生成 | 非网格对齐生成式高斯重建
- [RoomPlanner: Reachability-Aware View Sampling for Text-to-Room 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5856) | 单图/少图生成 | 可达性视角采样文本生成房间
- [PixGS: Pixel-Space Diffusion for Direct 3D Gaussian Splat Generation](https://eccv.ecva.net/virtual/2026/poster/5420) | 单图/少图生成 | 像素空间扩散直接生成3D高斯
- [GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation](https://eccv.ecva.net/virtual/2026/poster/4503) | 单图/少图生成 | 自回归3D高斯场景生成
- [SplatCtrlA: Generalizable Single Image to Fully Controllable 3D Avatar](https://eccv.ecva.net/virtual/2026/poster/5172) | 单图/少图生成 | 单图生成完全可控3D头像
- [DiGS-Avatar: Single-Image Animatable 3D Human Reconstruction via UV-Space Diffusion](https://eccv.ecva.net/virtual/2026/poster/3991) | 单图/少图生成 | UV空间扩散单图可动人体重建
- [Skyfall-GS: Synthesizing Immersive 3D Urban Scenes from Satellite Imagery](https://eccv.ecva.net/virtual/2026/poster/3179) | 单图/少图生成 | 卫星影像合成沉浸式城市场景

### 压缩与存储（10 篇）

- [PointSplat: Compact Gaussian Splatting via Human-Centric Prediction](https://eccv.ecva.net/virtual/2026/poster/3962) | 压缩与存储 | 人本预测紧凑高斯泼溅
- [3D Gaussian Splatting Compression with Object Scalability](https://eccv.ecva.net/virtual/2026/poster/4180) | 压缩与存储 | 物体可扩展的3DGS压缩
- [Monte Carlo Energy Aggregation for Mobile 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3145) | 压缩与存储 | 蒙特卡洛能量聚合移动端GS
- [EGGS: Explicitly Granular 3D Gaussian Splatting via Luma-Aware and Volume-Preserving Attribute Factorization](https://eccv.ecva.net/virtual/2026/poster/5695) | 压缩与存储 | 亮度感知体积保持属性分解压缩
- [Matryoshka Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3234) | 压缩与存储 | 套娃式嵌套压缩3DGS
- [KISS-GS: 3D Gaussian Splatting Compression Kept Simple](https://eccv.ecva.net/virtual/2026/poster/4959) | 压缩与存储 | 极简3DGS压缩方案
- [RaPTGS: Render-Agnostic Post-Training Compression of 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5498) | 压缩与存储 | 渲染无关3DGS后训练压缩
- [NanoGS: Training-Free and Lightweight Gaussian Splat Simplification](https://eccv.ecva.net/virtual/2026/poster/3359) | 压缩与存储 | 免训练轻量高斯简化
- [Fast and Compact 3D Gaussian Splatting with Polarized Opacity Prior](https://eccv.ecva.net/virtual/2026/poster/3772) | 压缩与存储 | 偏振不透明度先验快速紧凑GS
- [REFINE: Super-efficient Pruning for 3D Gaussian Splatting via Rendering-Free Primitive Importance](https://eccv.ecva.net/virtual/2026/poster/3858) | 压缩与存储 | 免渲染原语重要性超高效剪枝

### 动态/4D（20 篇）

- [Temporally Aware Densification for Dynamic 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/4688) | 动态/4D | 时序感知稠密化动态3DGS
- [TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5074) | 动态/4D | 时序刚体运动可扩展4DGS
- [Relaxed Rigidity with Ray-based Grouping for Dynamic Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/4602) | 动态/4D | 松弛刚体+射线分组动态GS
- [Multi4D: High-Fidelity Dynamic Gaussian Splatting via Multi-Level Competitive Allocation](https://eccv.ecva.net/virtual/2026/poster/3218) | 动态/4D | 多级竞争分配高保真动态GS
- [Towards Alias-Free 4D Gaussian Representations with Motion-Aware Filtering](https://eccv.ecva.net/virtual/2026/poster/5460) | 动态/4D | 运动感知滤波无混叠4D表示
- [Minute4D: Training High-Fidelity 4D Gaussian Splatting in One Minute](https://eccv.ecva.net/virtual/2026/poster/5815) | 动态/4D | 一分钟训练高保真4DGS
- [FillGS: Filling Observation Gaps in 4D Gaussian Splatting via Viewpoint-Time Selection and Generative Refinement](https://eccv.ecva.net/virtual/2026/poster/3622) | 动态/4D | 视角-时间选择+生成式补全4DGS
- [MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5347) | 动态/4D | 运动方差引导时序注意力动态GS
- [KineticGS: Momentum-driven Coherent 4D Gaussian Splatting for Monocular Dynamic Scene Reconstruction](https://eccv.ecva.net/virtual/2026/poster/4910) | 动态/4D | 动量驱动单目动态场景重建
- [Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction](https://eccv.ecva.net/virtual/2026/poster/4896) | 动态/4D | 手物交互动态重建
- [Learning Implicit Constitutive Laws for Dynamic 3D Gaussian Splatting from Monocular Videos](https://eccv.ecva.net/virtual/2026/poster/5073) | 动态/4D | 单目视频隐式本构律动态3DGS
- [SMG: Semantic Motion Graph for Monocular Dynamic Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3878) | 动态/4D | 语义运动图单目动态高斯
- [SIMSplat: Language-Aligned 4D Gaussian Splatting for Driving Scenario Generation](https://eccv.ecva.net/virtual/2026/poster/4537) | 动态/4D | 语言对齐4DGS驾驶场景生成
- [Unified Panoramic–Gaussian Representation for Monocular 4D Scene Synthesis](https://eccv.ecva.net/virtual/2026/poster/4215) | 动态/4D | 全景-高斯统一单目4D合成
- [4DGS360: 360° Gaussian Reconstruction of Dynamic Objects from a Single Video](https://eccv.ecva.net/virtual/2026/poster/4866) | 动态/4D | 单视频360°动态物体重建
- [FlexiAvatar: Unified 3D Gaussian Human Avatars Under Arbitrary Body Visibility](https://eccv.ecva.net/virtual/2026/poster/3675) | 动态/4D | 任意身体可见度统一高斯人体
- [AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors](https://eccv.ecva.net/virtual/2026/poster/3219) | 动态/4D | 遮挡YouTube视频+扩散先验人体
- [HandSCS: Structural Coordinate Space for Animatable Hand Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3877) | 动态/4D | 结构坐标空间可动手部高斯
- [Director: Instance-aware Gaussian Splatting for Dynamic Scene Modeling and Understanding](https://eccv.ecva.net/virtual/2026/poster/5035) | 动态/4D | 实例感知动态场景建模理解
- [MoBa-GS: Learning a Spatially-Varying Motion Basis over a Dynamic Canonical Space for 4D Reconstruction](https://eccv.ecva.net/virtual/2026/poster/5617) | 动态/4D | 规范空间运动基4D重建

### 编辑与重光照（11 篇）

- [AGE: Agentic Gaussian Editing in 3D Scenarios](https://eccv.ecva.net/virtual/2026/poster/5221) | 编辑与重光照 | 智能体3D场景高斯编辑
- [Geometry-Aware Style Transfer in 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/4174) | 编辑与重光照 | 几何感知3DGS风格迁移
- [DR-GS: Physically-Based Deformable and Relightable 2D Gaussians](https://eccv.ecva.net/virtual/2026/poster/4152) | 编辑与重光照 | 物理可变形可重光照2D高斯
- [GAINS: Gaussian-based Inverse Rendering from Sparse Multi-View Captures](https://eccv.ecva.net/virtual/2026/poster/4584) | 编辑与重光照 | 稀疏多视角高斯逆渲染
- [Capacity-Controlled Multi-View Stylization of 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/4306) | 编辑与重光照 | 容量控制多视图风格化
- [SplatPainter: Interactive Authoring of 3D Gaussians from 2D Edits via Test-Time Training](https://eccv.ecva.net/virtual/2026/poster/4992) | 编辑与重光照 | 2D编辑测试时训练交互创作
- [CoIn: Comprehensive 2D-3D Inpainting with Gaussian Splatting Guidance](https://eccv.ecva.net/virtual/2026/poster/4201) | 编辑与重光照 | 2D-3D联合高斯引导修补
- [Flash-Refine: Frustum-Guided Local Incremental Learning for Efficient 3D Gaussian Splatting Completion](https://eccv.ecva.net/virtual/2026/poster/5936) | 编辑与重光照 | 平截头体局部增量学习补全
- [MotionEditGS: Editing Motion and Appearance of 4D Scenes from Monocular Video via Semantically Anchored Gaussians](https://eccv.ecva.net/virtual/2026/poster/4803) | 编辑与重光照 | 语义锚定高斯编辑4D运动外观
- [COSY: Compositional 3DGS Synthesis for Disentangled Human Head Editing](https://eccv.ecva.net/virtual/2026/poster/3814) | 编辑与重光照 | 组合式3DGS解耦人头编辑
- [MeGAS: Thermomechanical Dynamic Gaussian Splatting for Thermophysical Scene Editing](https://eccv.ecva.net/virtual/2026/poster/3980) | 编辑与重光照 | 热力学相变高斯场景编辑

### SLAM/定位建图（11 篇）

- [Real-Time LiDAR Gaussian Splatting SLAM via Geometry-Aware Covariance Coupling](https://eccv.ecva.net/virtual/2026/poster/3180) | SLAM/定位建图 | 几何感知协方差耦合实时SLAM
- [Robust and Efficient Monocular 3D Gaussian SLAM for Kilometer-Scale Outdoor Scenes](https://eccv.ecva.net/virtual/2026/poster/4689) | SLAM/定位建图 | 公里级户外单目高斯SLAM
- [Triangle Splatting SLAM](https://eccv.ecva.net/virtual/2026/poster/5458) | SLAM/定位建图 | 三角原语稠密RGB-D SLAM
- [Cube-Splat: High-Fidelity 360° Gaussian Splatting SLAM via Cubemap Factorization and Adjoint-Consistent Optimization](https://eccv.ecva.net/virtual/2026/poster/4224) | SLAM/定位建图 | 立方体贴图分解360°高斯SLAM
- [Manifold-Aware Spectral Compaction: A Graph Signal Processing Perspective on Online Gaussian Reduction for 3DGS SLAM](https://eccv.ecva.net/virtual/2026/poster/4067) | SLAM/定位建图 | 谱图信号在线高斯削减SLAM
- [VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM](https://eccv.ecva.net/virtual/2026/poster/3982) | SLAM/定位建图 | 视觉惯性高斯SLAM
- [Structure Gaussian Splatting SLAM](https://eccv.ecva.net/virtual/2026/poster/5019) | SLAM/定位建图 | 结构化高斯泼溅SLAM
- [SpectralSplats: Robust Differentiable Tracking via Spectral Moment Supervision](https://eccv.ecva.net/virtual/2026/poster/3727) | SLAM/定位建图 | 谱矩监督稳健可微跟踪
- [Graph-GSReg: Leveraging 3D Scene Graphs for Gaussian Splatting Registration](https://eccv.ecva.net/virtual/2026/poster/5375) | SLAM/定位建图 | 场景图配准多3DGS融合
- [Geometry-Preserving in 3D Gaussian Splatting for LiDAR-Camera Extrinsic Calibration](https://eccv.ecva.net/virtual/2026/poster/4656) | SLAM/定位建图 | 3DGS保几何LiDAR-相机外参标定
- [Robust 3DGS-based SLAM via Adaptive Kernel Smoothing](https://eccv.ecva.net/virtual/2026/poster/5891) | SLAM/定位建图 | 自适应核平滑鲁棒3DGS SLAM

### 语义/分割/理解（9 篇）

- [ZeroSplat: Generalized Referring Segmentation in 3D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3373) | 语义/分割/理解 | 3DGS泛化指代分割
- [DLGStream: Dynamic Language-embedded Guassian Splatting for Open-vocabulary Enabled Free-viewpoint Video Streaming](https://eccv.ecva.net/virtual/2026/poster/4664) | 语义/分割/理解 | 动态语言高斯自由视角流媒体
- [LEGO: Leveled Language Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5407) | 语义/分割/理解 | 分层语言高斯泼溅
- [FLEG: Feed-Forward Language Embedded Gaussian Splatting from Any Views via Compact Semantic Representation](https://eccv.ecva.net/virtual/2026/poster/5168) | 语义/分割/理解 | 任意视角紧凑语义前馈语言GS
- [SkySplat-OV: Generalizable Language Gaussian Splatting for Open-Vocabulary Scene Understanding from Sparse Satellite Views](https://eccv.ecva.net/virtual/2026/poster/3371) | 语义/分割/理解 | 稀疏卫星视图开放词汇理解
- [Online Segment 3D Gaussians via Launching Virtual Drones](https://eccv.ecva.net/virtual/2026/poster/5801) | 语义/分割/理解 | 虚拟无人机在线3D高斯分割
- [G2P: Gaussian-to-Point Attribute Alignment for Boundary-Aware 3D Segmentation](https://eccv.ecva.net/virtual/2026/poster/5646) | 语义/分割/理解 | 高斯到点属性对齐边界感知分割
- [CapFrame: Text-Instructed Viewpoint Grounding in 3D Gaussian Scenes via Geometric Pseudo Labels](https://eccv.ecva.net/virtual/2026/poster/4479) | 语义/分割/理解 | 几何伪标签文本指代视角定位
- [RIGS: Radar-Informed Gaussian Splatting for Uncertainty-Aware 3D Occupancy and Motion Prediction](https://eccv.ecva.net/virtual/2026/poster/5796) | 语义/分割/理解 | 雷达高斯占用与运动预测

### 渲染改进（6 篇）

- [InstGS: Shared-Template Gaussian Instancing for Object-Redundancy-Free Rendering](https://eccv.ecva.net/virtual/2026/poster/4199) | 渲染改进 | 共享模板实例化去冗余渲染
- [3D Gaussian Texture for Real-time Mesoscale Appearance Synthesis and Rendering](https://eccv.ecva.net/virtual/2026/poster/5185) | 渲染改进 | 实时介观外观合成与渲染
- [RefracGS: Novel View Synthesis Through Refractive Water Surfaces with 3D Gaussian Ray Tracing](https://eccv.ecva.net/virtual/2026/poster/3476) | 渲染改进 | 折射水面高斯光线追踪NVS
- [UniTriSplat: A Unified 3D Gaussian Splatting Framework with Uniform Spherical Rasterization for Universal Cameras](https://eccv.ecva.net/virtual/2026/poster/4818) | 渲染改进 | 统一球面光栅化通用相机框架
- [RAGA: Real Time Ray Traced Gaussian Shadow Casting for 3DGS Avatar-Scene Interaction](https://eccv.ecva.net/virtual/2026/poster/3220) | 渲染改进 | 实时光线追踪高斯阴影
- [3DGS3: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering](https://eccv.ecva.net/virtual/2026/poster/3945) | 渲染改进 | 超采样+插帧实时大规模渲染

### 其他（16 篇）

- [Complex-Valued 2D Gaussian Representation for Computer-Generated Holography](https://eccv.ecva.net/virtual/2026/poster/3347) | 其他 | 复值2D高斯计算全息(存疑)
- [Heat Kernel Textures -- the Geodesic Gaussians That Do Not Splat](https://eccv.ecva.net/virtual/2026/poster/3652) | 其他 | 热核测地高斯纹理(存疑)
- [Fourier Splatting: Generalized Fourier encoded primitives for scalable radiance fields](https://eccv.ecva.net/virtual/2026/poster/5499) | 其他 | 傅里叶编码原语可扩展辐射场
- [MLP Splatting: Object-Centric Neural Fields](https://eccv.ecva.net/virtual/2026/poster/5496) | 其他 | 物体中心MLP原语神经场
- [Deformable Triangle Splatting: Flexible Primitives for Real-Time Radiance Field Rendering](https://eccv.ecva.net/virtual/2026/poster/3724) | 其他 | 可变形三角原语实时辐射场
- [ThermoGS: Decoupling Physical Surface Attributes for Spatio-Temporal Thermal Field Emulation via 4D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3922) | 其他 | 4D高斯热力学热场仿真
- [Proto-Gaussian: MRI Modality Translation Based on Learnable Structural Prototypes and 2D Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/5283) | 其他 | 2D高斯MRI模态转换
- [Active View Selection with Perturbed Gaussian Ensemble for Tomographic Reconstruction](https://eccv.ecva.net/virtual/2026/poster/3847) | 其他 | 扰动高斯集成CT主动选角
- [FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3541) | 其他 | 快速可扩展CT高斯重建
- [X-SG2S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks](https://eccv.ecva.net/virtual/2026/poster/5800) | 其他 | 多维度水印版权保护3DGS
- [DefenseSplat: Enhancing the Robustness of 3D Gaussian Splatting via Frequency-Aware Filtering](https://eccv.ecva.net/virtual/2026/poster/4267) | 其他 | 频率感知滤波抵御3DGS对抗
- [DecoupleGS: Interactive 3D Gaussian Splatting for End-to-End Autonomous Driving Testing](https://eccv.ecva.net/virtual/2026/poster/5765) | 其他 | 动静解耦3DGS自动驾驶测试
- [E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes](https://eccv.ecva.net/virtual/2026/poster/4639) | 其他 | 3DGS场景主动感知基准
- [Boba: Batched Simulation for Physics-Based Gaussian Digital Twins](https://eccv.ecva.net/virtual/2026/poster/4947) | 其他 | 批量物理仿真高斯数字孪生
- [Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting](https://eccv.ecva.net/virtual/2026/poster/3787) | 其他 | 动态高斯高保真导航仿真器
- [MorphGS: Morphology-Adaptive Articulated Motion Transfer from Videos](https://eccv.ecva.net/virtual/2026/poster/4778) | 其他 | 形态自适应关节运动迁移(存疑)

---

## 四、附录：排除条目（共 21 篇，均经标题/摘要核实为无关主题）

### ICCV 2025 排除（9 篇）

- UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale（卷积核高斯分布理论）
- Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks（高斯噪声对抗）
- Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection（高斯离群合成 OOD 检测）
- Unraveling the Smoothness Properties of Diffusion Models: A Gaussian Mixture Perspective（高斯混合理论）
- Robust Test-Time Adaptation for Single Image Denoising Using Deep Gaussian Prior（深度高斯先验去噪）
- ASGS: Single-Domain Generalizable Open-Set Object Detection via Adaptive Subgraph Searching（子图搜索检测）
- GSOT3D: Towards Generic 3D Single Object Tracking in the Wild（点云 3D 跟踪基准）
- MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion（多相机 SfM）
- VGGSounder: Audio-Visual Evaluations for Foundation Models（视听评测）

### ECCV 2026 排除（12 篇）

- NURBS Splatting: A Unified Differentiable Rendering Framework for Vector Graphics（矢量图形可微渲染）
- CubicSplat: Differentiable Vector Graphics via Error-Bounded Forward Relaxation（矢量图形可微光栅化）
- Uncertainty-Driven Gaussian Sphere Propagation for 3D Semantic Segmentation（点云语义分割特征传播）
- Gaussian Volumetric Representation for Efficient Shear–Warp Visualization（医学体绘制可视化）
- Gaussian Belief Propagation Network for Depth Completion（图模型置信传播）
- Accelerating Diffusion Transformers with Gaussian Process Rectified Feature Cache（高斯过程缓存）
- Geometric Regularization for Long-Tailed Semi-Supervised Learning via Gaussian Feature Bridges（特征空间高斯桥）
- Slim-DETR: Real-Time Tiny Object Detection with Efficient Interaction and Gaussian Query（DETR 高斯查询）
- Prior-Conditioned Gaussian Discriminants for Generalizable AI-generated Image Detection（高斯判别式 AIGC 检测）
- GRE-Diff: Gaussian Room Embeddings for Structured Layout Diffusion（户型布局扩散）
- DGSeg: Dynamic Gating of Semantic-Spatial Guided Predictions for Reasoning Segmentation（推理分割门控）
- DIGS: Differentiable, Incremental, Global, Scalable Pruning for Language Models（大模型剪枝）

### 存疑保留条目说明

- 归入"其他"并标注存疑的条目：ICCV 的 Gaussian-based World Model（高斯世界模型做占用/运动预测，未明确 3DGS 原语）、SketchSplat（线原语 splatting 而非高斯）、Mixture of Experts Guided by Gaussian Splatters（视频异常检测中 3DGS 用法未在摘要前段明示）；ECCV 的 Complex-Valued 2D Gaussian（计算全息方向）、Heat Kernel Textures（受 3DGS 启发但"不 splat"）、MorphGS（运动迁移，摘要未明示高斯渲染）。
- 非高斯原语但属 splatting/3DGS 相邻研究（已收录并注明）：BBSplat、MLP Splatting、Fourier Splatting、Deformable Triangle Splatting、Triangle Splatting SLAM 等。

---

更新时间: 2026-11-09 22:50
