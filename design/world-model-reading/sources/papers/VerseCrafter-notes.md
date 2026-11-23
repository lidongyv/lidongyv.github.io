# VerseCrafter 精读笔记

> 数据源：`auto_docs/papers/txt/VerseCrafter.txt`（arXiv:2601.05138v2，1941 行，25 页）。全部事实性陈述附论文原句（英文逐字 + 行号）；未写内容标注「原文未写」；本人解读标注【推断】。

## 1) 原文信息

- 标题：`VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control`（L1）
- 作者：`Sixiao Zheng1,2 , Minghao Yin 3, Wenbo Hu4†, Xiaoyu Li4, Ying Shan4, Yanwei Fu1,2†`（L2，† 为通讯作者，L23）
- 机构：`1Fudan University 2Shanghai Innovation Institute 3HKU 4ARC Lab, Tencent PCG`（L3）
- 项目页：`Project Page:https://sixiaozheng.github.io/VerseCrafter_page/`（L4）
- ID：`arXiv:2601.05138v2  [cs.CV]  30 Mar 2026`（L37）
- CVPR 2026：**原文未写**——txt 全文未出现本文自述的会议归属（"CVPR"字样仅出现在参考文献条目中）。

## 2) Abstract 全文（L11–36，逐字；"V er-seCrafter"等为换行断字符）

> "Video world models aim to simulate dynamic, real-world
> environments, yet existing methods struggle to provide uni-
> fied and precise control over camera and multi-object mo-
> tion, as videos inherently capture dynamics in the projected
> 2D image plane. To bridge this gap, we introduceV er-
> seCrafter, a geometry-driven video world model that gen-
> erates dynamic, realistic videos from a unified 4D geometric
> world state. Our approach is centered on a novel4D Ge-
> ometric Controlrepresentation, which encodes the world
> state as a static background point cloud and per-object 3D
> Gaussian trajectories. This representation captures each
> object's motion path and probabilistic 3D occupancy over
> time, providing a flexible, category-agnostic alternative to
> rigid bounding boxes and parametric models. We render
> 4D Geometric Control into 4D control maps for a pre-
> trained video diffusion model, enabling high-fidelity, view-
> consistent video generation that faithfully follows the spec-
> ified dynamics. To enable training at scale, we develop
> an automatic data engine and constructV erseControl4D,
> a real-world dataset of 35K training samples with automat-
> ically derived prompts and rendered 4D control maps. Ex-
> tensive experiments show that VerseCrafter achieves supe-
> rior visual quality and more accurate control over camera
> and multi-object motion than prior methods."

## 3) 问题与动机（逐字引用）

论文主张的核心矛盾是「理想世界模型应模拟完整 4D 时空，但视频只能记录投影后的 2D 平面」：

> "However, the reliance on video introduces a
> fundamental challenge: while an ideal world model should
> simulate the full 4D spatiotemporal space to reflect our
> physical reality, videos inherently capture dynamics in the
> projected 2D image plane."（L44–48）

2D 投影为何限制控制精度（三段递进论据）：

1. 2D 线索缺乏 3D 感知、大视角变化下失效：`Existing approaches typically rely on 2D cues such as point trajectories [98], optical flow [58], masks [125], or bounding boxes [91], which lack 3D aware-ness and often fail under large viewpoint changes.`（L54–57）
2. 更高级的 3D 感知信号也无法形成统一可编辑的 4D 场景状态：`More advanced 3D-aware methods use depth maps [124], sparse 3D trajectories [15], 3D bounding boxes [94], or paramet-ric human models like SMPL-X [11] to align camera and object motion in 3D space. Nevertheless, these control rep-resentations remain inadequate for modeling multi-object dynamics as a unified, compact, and editable 4D geomet-ric scene state in a shared world coordinate frame.`（L58–64）
3. 具体缺陷逐一列举：`For in-stance, sparse trajectories are often noisy and incomplete, 3D bounding boxes impose rigid constraints ill-suited to natural objects, and SMPL-X representations are category-limited.`（L64–68）

补充论据（真实场景 vs 合成环境）：`Furthermore, several existing works focus on syn-thetic game environments [41, 111, 115], where precise an-notations are available for training. However, controllable modeling of complex, realistic 4D scenes with multi-object motion remains underexplored.`（L68–72）

论文对自身方案的定位句：`Unlike 2D control signals, our 4D Geometric Control is inherently 3D-aware, making it naturally more view-consistent and robust to occlusions, and thus a more effec-tive and reliable interface for video world modeling.`（L97–100）

## 4) 方法逐机制

### 4.1 4D Geometric Control 表示（§3.1）

**定义（背景点云 + 逐物体 3DGS 轨迹，共享世界坐标系）：**

> "We represent each scene as an explicit 4D geometric scene
> state, which we term4D Geometric Control. This editable
> representation consists of a static background point cloud
> P bg and per-object 3D Gaussian trajectories{G to}, all de-
> fined in a shared world coordinate frame."（L209–213）

共享坐标系的意义：`Crucially, the back-ground point cloud and per-object 3D Gaussian trajectories share a common world coordinate frame, enabling coherent and unified control over both camera and object motion.`（L87–90）

**背景点云构建：** `we start from the input image, estimate monocular depth and camera intrinsicsKusing MoGe-2 [95], and obtain object masks {Mo}with Grounded SAM2 [77], where the user selects one or more objects to control via text prompts or clicks.`（L214–219）输入视角被定为参考世界坐标系（L219–220，`the reference camera pose is given byR1 =I andt 1 =0`）；像素按式 (1) 反投影，物体掩码把点云分割为逐物体点云与静态背景点云（L224–235）。关键渲染语义：`During generation, the background at frametis rendered fromP bg under the camera pose, so viewpoint changes are realized as rigid camera motion in a fixed 3D world rather than by hallucinating a new background at every frame.`（L236–239）

**3D Gaussian 轨迹：** 单高斯编码位置（均值）/形状尺寸（特征值）/朝向（特征向量）：`A single 3D GaussianG o(x) = N(x|µ o,Σ o)in the world coordinate frame compactly en-codes an object's position (throughµ o), approximate shape and size (through the eigenvalues ofΣ o), and orientation (through the eigenvectors ofΣo).`（L240–244）；轨迹定义为高斯序列，`whose means{µ to}trace the motion path in 3D, while the covariances{Σ to}capture how the object's spatial extent and orientation evolve over time.`（L247–249 与 L292 衔接）。相比刚性盒的优势：`This probabilistic formu-lation describes the object's 3D occupancy in a soft, con-tinuous manner and yields a compact control space that is more flexible than rigid 3D bounding boxes and more category-agnostic than parametric body models.`（L292–296）

**可编辑性（用户界面）：** `The low-dimensional parameters{µ to,Σ to}naturally support flexible, user-driven editing. In practice, we convert eachG to into an ellipsoid mesh for visualization in a 3D edi-tor such as Blender, and let the user specify or refine the tra-jectory by dragging and keyframing this ellipsoid in world coordinate space. The edited poses and shapes are mapped back to{µ to,Σ to}as control signals. The ellipsoids are only a user interface; all conditioning maps used by model are rendered directly from the underlying 3D Gaussians.`（L303–311）

### 4.2 控制图渲染（§3.1 末）

每帧渲染三类解耦通道：`For each framet, we generate three types of maps: (i) background RGB/depth, RGB bgt and Depthbgt , by projecting the static background point cloudP bg un-der the camera pose(R t,t t); (ii) 3D Gaussian trajectory RGB/depth, RGB trajt and Depth trajt , by projecting the per-object Gaussians{G to}into soft elliptical footprints and tak-ing depth from the corresponding ellipsoid surfaces; and (iii) a soft merged maskM t, used as a control mask, that indicates regions where the diffusion model should synthe-size or overwrite content, obtained by inverting background visibility and merging it with the projected 3D Gaussian footprints, followed by Gaussian smoothing.`（L314–325）首帧保持：`For the first framet= 1, we replace RGB bg1 with input image and set M1 = 0, so that the first frame is preserved and only future frames are modified.`（L325–327）解耦语义：`Background and 3D Gaussian maps share the same 4D geometric scene state but are rendered throughdecoupledchannels, disentangling camera motion from object motion while preserving geometric consistency.`（L328–331）

### 4.3 GeoAdapter 架构（§3.2）

骨干冻结：`We adopt Wan2.1-14B [88] as a frozen latent video diffusion backbone with a Wan Encoder, a Wan-DiT denoiser and a Wan Decoder. VerseCrafter treats Wan2.1 as a generic video prior: we do not change its architecture or weights, and instead attach a lightweight geometric adapter that conditions the backbone on rendered 4D control maps.`（L333–338）

注入方式（类 ControlNet）：`The encoded RGB/depth maps and the processed mask are concatenated channel-wise to form a spatio-temporal geometry tensor. GeoAdapter is a lightweight DiT-style branch that operates on this geometry tensor. It shares the same hidden dimen-sionality as the Wan-DiT blocks, but uses far fewer layers. We interleave GeoAdapter blocks with the frozen Wan-DiT: everyk-th DiT block in Wan-DiT is paired with a GeoAd-apter block whose output is linearly projected and added to the corresponding DiT block as a residual modulation.`（L345–353）文本由 umT5 注入两侧（L354–356）。附录给出零初始化残差细节：`Each GeoAdapter block is initialized from the weights of its paired Wan-DiT block for stable training, whileW (m)0 is initialized to zero. As a result, VerseCrafter behaves identically to the origi-nal Wan2.1 backbone at the beginning of training.`（L853–857，注入集 `[0,5,10,15,20,25,30,35]`、Wan-DiT 40 层、隐藏维 5120，Table 4，L753–757）

**推理三模式：** `At inference time, VerseCrafter supports camera-only, object-only, and joint control within a uni-fied framework.`（L360–361）；camera-only 时轨迹图置零、object-only 时相机固定、joint 时两类图同时渲染（L361–372）。

### 4.4 数据引擎与 VerseControl4D（§4）

`VerseControl4D is built through four stages: data collec-tion, clip extraction, quality filtering, and data annotation.`（L378–379）
- 数据来源：`VerseControl4D is built from two recent world-exploration datasets, Sekai-Real-HQ [56] and SpatialVID-HQ [93], which provide long in-the-wild videos with diverse outdoor and urban scenes, camera poses, and captions, but lack object-motion labels.`（L380–384）
- 切片：`We apply PySceneDetect to detect shots in the videos. For each shot longer than 81 frames, we uniformly sample an 81-frame sub-clip and discard shorter shots, matching the default temporal length used by the Wan2.1 backbone.`（L387–391）
- 质量过滤：`we keep only clips whose con-trollable object count lies in[1,6]. We then discard clips where any object mask covers more than20%of the im-age area.`（L396–399；另有"person . human . car . animal"提示词、人体截断/宽高比 [2,4] 过滤与美学亮度过滤，L394–403）
- 标注：`We first generate a descriptive caption using Qwen2.5-VL-72B [5], which serves as text prompt.`（L442–443）；几何管线：`we adopt MegaSAM as the base pipeline and replace its monocular and metric depth modules with MoGe-2 [95] and UniDepth V2 [72], respectively, to obtain more accurate and temporally consistent depth.`（L444–447）
- 规模：`In total, VerseControl4D contains 35,000 training sam-ples and 1,000 validation samples. In the training set, about 26% of samples are sourced from Sekai-Real-HQ and 74% from SpatialVID-HQ, while 20% of the samples de-pict static scenes`（L463–468）；`The validation set includes 250 static-scene sam-ples to specifically assess camera-only control.`（L493–494）附录 Table 5 细分：Train = Sekai 动态 9,071 + Sekai 静态 7,000 + SpatialVID 18,929；Val = 468 + 250 + 282（L866–869）。

## 5) 实验与主结果

**训练/推理配置：** `we setk= 5so that ev-ery 5th DiT block in Wan-DiT is paired with a GeoAdapter block. We use the Adam optimizer with a learning rate of 2e-5, a constant learning rate schedule with 100 warmup steps.`（L500–503）；`All experiments are conducted on 16 96-GB GPUs with a global batch size of 16. Training is performed in two stages: we first train for 2,500 iterations on480P clips, and then fine-tune the same model for another 2,500 itera-tions on720P clips. The total wall-clock training time is about 380 hours.`（L504–509）；`At inference time, we use 50 denoising steps and a classifier-free guidance scale of 5.0. Generat-ing an 81-frame720P video clip on 8 96-GB GPU takes about 1152 seconds, with a peak per-GPU memory usage of about 90 GB.`（L511–515）

**指标定义（原文）：** VBench-I2V 取八维均值（附录式 10）；`For camera control, we follow CameraC-trl [33] and report rotation error (RotErr) and translation er-ror (TransErr). For object-motion control, we adopt ObjMC proposed in MotionCtrl [98].`（L516–520）；`ObjMC is com-puted as the average Euclidean distance between the esti-mated and ground-truth 3D Gaussian means over all con-trolled objects and frames.`（L524–527；附录补充：未匹配物体惩罚 λ=10.0m，匈牙利匹配，式 13–15，L1044–1077）

**Table 1（联合控制，L534–538）：**

| 方法 | Overall↑ | ImagingQ↑ | AestheticQ↑ | DynamicDeg↑ | MotionSm↑ | BgCons↑ | SubjCons↑ | I2VBg↑ | I2VSubj↑ | RotErr↓ | TransErr↓ | ObjMC↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perception-as-Control | 83.66 | 66.81 | 53.34 | 73.91 | 96.89 | 93.19 | 94.02 | 96.35 | 94.78 | 5.006 | 8.767 | 6.556 |
| Yume | 85.47 | 71.16 | 52.39 | 72.24 | 98.96 | 95.66 | 96.43 | 98.51 | 98.39 | 7.560 | 8.735 | 7.959 |
| Uni3C | 83.55 | 68.06 | 53.16 | 66.09 | 98.94 | 93.74 | 94.19 | 97.19 | 97.05 | 1.361 | 7.731 | 5.883 |
| **Ours** | **88.10** | **72.70** | **57.49** | **86.26** | 98.79 | **95.69** | **96.48** | **98.76** | **98.65** | **0.890** | **3.103** | **2.507** |

**Table 2（仅相机控制、静态场景，L543–546）：**

| 方法 | Overall↑ | ImagingQ↑ | AestheticQ↑ | DynamicDeg↑ | MotionSm↑ | BgCons↑ | SubjCons↑ | I2VBg↑ | I2VSubj↑ | RotErr↓ | TransErr↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ViewCrafter | 84.04 | 69.56 | 55.52 | 68.02 | 97.86 | 92.09 | 94.25 | 97.70 | 97.29 | 2.101 | 9.868 |
| Voyager | 78.12 | 55.48 | 49.80 | 65.34 | 99.39 | 92.31 | 91.55 | 86.02 | 85.03 | 3.557 | 3.880 |
| FlashWorld | 85.33 | 71.68 | 58.74 | 73.46 | 98.35 | 94.27 | 92.47 | 95.38 | 98.32 | 1.792 | 3.257 |
| **Ours** | **86.80** | **74.57** | 54.78 | **80.34** | 97.62 | **94.88** | **95.55** | **97.86** | **98.79** | **0.650** | **2.587** |

（注：Table 2 中 Ours 的 Aesthetic Quality 54.78 低于 FlashWorld 的 58.74，Motion Smoothness 97.62 也非最高；原文只说 "maintaining motion smoothness compara-ble to prior methods"，L599–600。全文未给出显著性检验或置信区间。）

**Table 3（消融，L614–618）：**

| 变体 | Overall↑ | ImagingQ↑ | AestheticQ↑ | DynamicDeg↑ | MotionSm↑ | BgCons↑ | SubjCons↑ | I2VBg↑ | I2VSubj↑ | RotErr↓ | TransErr↓ | ObjMC↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (3D Bounding Box) | 85.45 | 69.23 | 55.70 | 78.57 | 98.70 | 92.92 | 93.27 | 97.74 | 97.48 | 1.350 | 3.805 | 4.520 |
| Ours (3D Point Trajectory) | 85.57 | 70.29 | 55.27 | 78.23 | 98.63 | 94.00 | 92.75 | 97.85 | 97.55 | 1.298 | 3.281 | 6.896 |
| Ours (w/o depth) | 85.64 | 70.19 | 55.00 | 80.60 | 98.66 | 92.07 | 92.83 | 98.07 | 97.69 | 1.177 | 3.900 | 4.929 |
| Ours (BG & FG Merged) | 85.72 | 69.19 | 54.86 | 83.72 | 98.65 | 91.15 | 92.86 | 97.93 | 97.41 | 1.080 | 3.803 | 3.726 |
| **Ours（全模型）** | **88.10** | **72.70** | **57.49** | **86.26** | **98.79** | **95.69** | **96.48** | **98.76** | **98.65** | **0.890** | **3.103** | **2.507** |

正文结论句：`On 3D control metrics, VerseCrafter substan-tially reduces rotation, translation, and object-motion er-rors compared with the best-performing baseline, reflect-ing much tighter alignment with the target 4D geometric control.`（L566–570）

## 6) 消融/局限（原文逐字）

三个消融设计（L635–640）：`We conduct ablations to analyze three key design choices in VerseCrafter: (i) the 3D representations for object motion, (ii) the use of depth in control maps, and (iii) the decoupling of background and foreground controls.` 替代表示的具体做法：`(1) an oriented3D bounding boxwhose axes follow the Gaus-sian's principal directions and whose side lengths are scaled by its principal spreads; and (2) a3D point trajectory that retains only the Gaussian centroid.`（L643–647）；结论：`replacing Gaussians with boxes slightly hurts both visual quality and control accu-racy, while point trajectories give the weakest object-motion consistency.`（L650–653）；无深度：`This variant yields a lower Overall Score and significantly worse 3D control (higher RotErr and ObjMC values).`（L659–661）；合并通道：`it consistently underperforms the full model, with a particularly noticeable drop in object-motion accuracy.`（L674–675）

**官方局限（§H，L1296–1328，四条全部原文）：**
1. 单高斯只给椭球级控制：`our current object representation provides only ellipsoid-levelcontrol through a single 3D Gaussian per ob-ject, which limits fine-grained pose and part-level articula-tion, especially for human-like or near-symmetric objects.`（L1300–1303）
2. 背景点云静态：`our background point cloud is reconstructed from the first frame and serves as a mostly static geometric scaffold, which limits controllability for highly non-rigid and texture-dominant scene dynamics such as waterfalls.`（L1307–1310）
3. 无物理约束：`although VerseCrafter enforces 4D geometric consistency through explicit camera control and 3D Gaus-sian trajectory control, it does not impose explicit phys-ical constraints during generation.`（L1314–1317）
4. 算力贵：`VerseCrafter remains computationally expensive at high resolution and long temporal horizons because it conditions a large frozen video diffusion backbone and ren-ders multi-channel 4D controls for all frames.`（L1322–1325）

附录补充失败模式：`VerseCrafter is most effective when the underlying 4D geometric cues are suf-ficiently informative, while failures mainly arise in under-constrained cases such as subtle human orientation changes or highly non-rigid background dynamics.`（L1247–1251）；对深度噪声不敏感：`the reconstructed point cloud acts as acoarsegeometric scaf-fold rather than a per-pixel hard constraint.`（L1239–1241）

## 7) 叙事关联【推断】（≤3 条）

1. 【推断】"4D 几何场景状态 = 世界模型的显式世界接口"。VerseCrafter 把相机轨迹与多物体运动统一进一个可编辑状态（L209–213），并强调可编辑性（Blender 拖拽椭球，L303–311），若本项目叙事是「可交互的 4D 世界模拟/编辑平台」，这篇论文是「显式几何状态 + 冻结扩散先验」路线的直接证据。附原句：`This editable representation consists of a static background point cloud P bg and per-object 3D Gaussian trajectories{G to}, all de-fined in a shared world coordinate frame.`（L210–213）
2. 【推断】"解耦渲染通道 = 控制可分解性"。背景/前景分通道渲染（L328–331）与推理三模式（camera-only/object-only/joint，L360–361）证明统一状态上可实现控制维度的解耦，若叙事涉及「统一世界模型支持多模态控制入口」，可引用此设计。附原句：`Background and 3D Gaussian maps share the same 4D geometric scene state but are rendered throughdecoupledchannels, disentangling camera motion from object motion while preserving geometric consistency.`（L328–331）
3. 【推断】"轻几何脚手架 + 强扩散先验"是低成本真实世界视频生成路线：几何只作粗约束、细节交给 Wan2.1 先验（L1239–1241 的 "coarse geometric scaffold"），且数据引擎把无标注真实视频自动转为训练对（L373–386），说明可扩展性瓶颈在扩散推理而非几何构造（L1293–1295）。附原句：`the reconstructed point cloud acts as acoarsegeometric scaf-fold rather than a per-pixel hard constraint.`（L1239–1241）

## 8) 主要图注抄录

- **Fig. 1**（L8–10）：`VerseCrafterenables precise control of camera motion and multi-object motion via a 4D Geometric Control representation built from a static background point cloud and per-object 3D Gaussian trajectories, producing videos that better follow the desired motion than Yume [63] and Uni3C [11] and more closely match the ground-truth video.`
- **Fig. 2**（L286–291）：`Framework of VerseCrafter.Given an input image and a text prompt, we estimate depth and obtain user-specified object masks to construct4D Geometric Control…rendered into per-frame background RGB/depth, 3D Gaussian trajectory RGB/depth, and a soft merged mask, forming multi-channel 4D control maps. The 4D control maps are encoded and fed into the proposed GeoAdapter, which conditions a frozen Wan2.1-14B backbone together with text embeddings from umT5…`
- **Fig. 3**（L430–437）：`Construction pipeline of VerseControl4D.Starting from Sekai-Real-HQ and SpatialVID-HQ, we extract 81-frame clips and apply quality filtering. For each retained clip, Qwen2.5-VL-72B, Grounded-SAM2, and MegaSAM provide captions, ob-ject masks, depth, and camera trajectory, which are lifted into background/object point clouds, from which 3D Gaussian trajecto-ries are fitted, and then rendered into background/trajectory maps and a soft merged mask…`
- **Fig. 4**（L479–482）：`Qualitative comparison of joint camera and object motion control.Perception-as-Control and Uni3C exhibit noticeable human deformation, while Yume roughly follows the text-described motion but lacks precise camera control. Uni3C is also limited to single human. In contrast, VerseCrafter more faithfully follows both the camera trajectory and multi-object motion…`
- **Fig. 5**（L489–491）：`ViewCrafter and V oyager exhibit distorted facades, drifting structures, or inaccurate camera motion, while FlashWorld tends to produce blurred scene boundaries and imprecise camera motion. In contrast, VerseCrafter better follows the target camera trajectory while preserving sharp details and globally consistent 3D geometry.`
- **Fig. 6**（L556–562）：`Ablation on 3D representations for object motion control…3D point trajectory and 3D bounding box often cause scale drift and misaligned motion (red boxes), whereas 3D Gaus-sian trajectory better follows the intended object motion while pre-serving plausible shapes and background interactions.`
- **Fig. 7**（L582–589）：`Ablation on depth-aware control…Without depth, the model often produces incorrect foreground-background ordering, e.g., lampposts are pulled in front of distant buildings, and occlusion boundaries drift over time (red boxes). With RGB+depth, the model recovers consistent parallax and oc-clusion, producing geometry much closer to the ground truth.`
- **Fig. 8**（L624–630）：`Ablation on decoupled background and foreground controls…When the controls are merged, object motion control degrades significantly (red boxes), whereas the decoupled design better preserves the static back-ground and produces more accurate and stable object motion.`
- **Fig. 9**（L834–839）：`Detailed architecture of VerseCrafter…all geometry latents are then concatenated along the channel dimension to form a unified spatio-temporal geometry feature…GeoAdapter outputs are passed through zero-initialized linear layers and added to the backbone tokens as residual modulations…`
- **Fig. 10**（L904–906）：`VerseControl4D dataset examples…we visualize the input image and target camera trajectory (left), followed by several frames of ground-truth video and our rendered control signals (right): background RGB/depth, 3D Gaussian trajectory RGB/depth for controlled objects, and the final merged mask.`
- **Fig. 15**（L1219–1222）：`Additional analysis of control fidelity and boundary cases. (a) Orientation controllability.Two success cases on rigid anisotropic objects and one failure case on a human-like subject.(b) Dynamic background modeling.Two success cases on moderate background dynamics and one failure case on highly non-rigid background motion.(c) Articulated and non-rigid object controllability.`
- **Fig. 16**（L1256–1258）：`(a) Single-view vs. multi-view input.Multi-view reconstruction improves geometry coverage and novel-view faithfulness.(b) Robustness to monocular-depth errors.Even with noisier depth and distorted point clouds, the generated videos remain visually similar and preserve the main scene structure.`

## 附：数据核对备注

- 正文与附录数字一致：35K 训练 / 1K 验证（L463–464 与 L879–880）；26%/74% 来源占比（L465–466 与 L881–882）；Table 5 的 Train 合计 9,071+7,000+18,929=35,000，Val 合计 468+250+282=1,000，吻合。
- 推理成本两处口径：正文 L513–515 "takes about 1152 seconds, with a peak per-GPU memory usage of about 90 GB"；附录 Table 7 拆解 Diffusion Inference 866 s（50 步）+ 模型加载 203 s + 状态构建 23 s + 渲染 60 s ≈ 1152 s，自洽；Table 6 另报告 FSDP 可将峰值内存 90→70 GB、加 CPU offload 57 GB（L1262–1265）。
- 论文全文无与"CVPR 2026"相关的自述信息；评测为 VBench-I2V 自动指标 + 自建数据集轨迹误差，未做用户研究（原文未写）。
