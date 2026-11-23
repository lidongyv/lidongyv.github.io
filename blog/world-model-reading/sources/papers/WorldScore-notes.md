# WorldScore 精读笔记

> 数据源：`auto_docs\papers\txt\WorldScore.txt`（82,005 字符，21 页，arXiv 2504.00983v2）。本次只精读正文（L1–659：Abstract→Conclusion），参考文献 L665–1089 与附录 L1090–1677 略读。所有事实性陈述附论文原句（英文，逐字取自 txt，注明行号）；论文未写处标「原文未写」；本人解读标【推断】。txt 为 PDF 抽取，行尾连字符断词（如 L5 "uni-fied"）在引用中按原文拼合，其余逐字保留。

## 1) 原文信息

- 标题（L1）："WorldScore: A Unified Evaluation Benchmark for World Generation"
- 作者（L2）："Haoyi Duan∗ Hong-Xing Yu∗ Sirui Chen Li Fei-Fei Jiajun Wu"；脚注（L41）："*Equal contribution."；单位（L3）："Stanford University"
- 版本（L82）："arXiv:2504.00983v2 [cs.GR] 29 Nov 2025"
- 项目页（L19–21）："Our dataset, evaluation code, and leaderboard can be found at https://haoyi-duan.github.io/WorldScore/."

## 2) Abstract 全文逐字（L4–21）

（合并 txt 换行，其余逐字）

> We introduce the WorldScore benchmark, the first unified benchmark for world generation. We decompose world generation into a sequence of next-scene generation tasks with explicit camera trajectory-based layout specifications, enabling unified evaluation of diverse approaches from 3D and 4D scene generation to video generation models. The WorldScore benchmark encompasses a curated dataset of 3,000 test examples that span diverse worlds: static and dynamic, indoor and outdoor, photorealistic and stylized. The WorldScore metric evaluates generated worlds through three key aspects: controllability, quality, and dynamics. Through extensive evaluation of 20 representative models, including both open-source and closed-source ones, we reveal key insights and challenges for each category of models. Our dataset, evaluation code, and leaderboard can be found at https://haoyi-duan.github.io/WorldScore/.

## 3) 问题与动机（为何需要统一的世界生成基准）

论文对"世界生成"的定义（L23–26）：

> "Recent advances in visual generation have sparked growing interest in world generation—the creation of large-scale, diverse worlds with various scenes, which finds wide applications in entertainment, education, simulation, and embodied AI."

需求端变化（L31–35）：

> "However, as the concept of world generation expands, users demand to generate more comprehensive worlds that seamlessly integrate multiple varied scenes with detailed spatial layout controls rather than disconnected individual environments."

缺口声明（L36–39）：

> "Achieving this vision requires a unified evaluation benchmark that systematically assesses different types of world generation models across large-scale, diverse worlds, which is currently absent."

现有基准的两个具体缺陷（L64–73）：

> "For example, VBench [26] primarily evaluates text-to-video (T2V) tasks using curated prompts without explicit spatial layout control, restricting their evaluations to single scenes (Figure 1). Moreover, despite the promising potential of 3D and 4D scene generation methods for world generation, current benchmarks lack essential components such as camera specifications and reference images, making them incompatible with many state-of-the-art 3D/4D scene generation methods that require an image or a camera trajectory as inputs [11, 16, 39, 90, 91]."

Related Work 中再强调（L148–157）：现有基准 "focus on evaluating video generation models based on single-scene video quality without layout control and multi-scene generation. Furthermore, their designs are incompatible with 3D/4D scene generation methods that require camera specification. In contrast, our WorldScore benchmark is designed to focus on evaluating world generation approaches with multi-scene generation tasks, and it is designed to accommodate 3D, 4D, I2V and T2V models."

## 4) 方法

### 4.1 next-scene 任务分解（核心设计）

Intro 版（L74–82）：

> "Our key design is to decompose world generation into a sequence of next-scene generation tasks, where each step is characterized by a triplet of (current scene, next scene, layout). For unified evaluation across different methods, we provide both an image and a text prompt for a current scene, as well as both camera matrices and a textual description for a layout specification."

形式化定义与式 (1)（L232–243）：

> "We decompose the world generation task into a sequence of next-scene generation tasks, where each step is specified by a triplet of (C,N,L), where C={I,P} denotes the current scene given by a scene image I and a text prompt P, N denotes the next-scene text prompt, and L={T,Y} denotes the layout given by a camera trajectory T=(C1,C2,···,CN) where Ci denotes a camera matrix and a text prompt of camera movement Y. Then, a world generation model is instructed to generate a video: V=gworld(wproc(C,N,L)), (1) where V denotes a video, gworld denotes the world generation model, and wproc denotes a model-specific pre-processing which we detail in Supp. A."

统一评测的前提——所有方法输出同一格式（L101–103）："All methods are evaluated on a common output format, i.e., rendered or generated videos, to enable direct comparison of generation across different types of approaches."

静态/动态两类任务显式解耦（L245–248）："Static and dynamic worlds. We explicitly disentangle the evaluation of dynamics aspect from the controllability and quality aspects due to their distinct natures." 静态任务（L249–253）："We instruct a model to generate varying-length scene sequences for controllability and quality assessment. Here, the next-scene text prompt N describes the new scene contents, and the layout L describes large camera movements." 动态任务（L294–299）："We instruct a model to generate in-scene motion for dynamics assessment. Here, the next-scene text prompt N describes the same scene content as C but with dynamics changes, e.g., an animal moving. The layout L explicitly specifies a fixed camera position without any camera motion."

### 4.2 相机轨迹布局 L={T,Y}

（L366–380）：

> "A layout L={T,Y} is given by a camera trajectory T=(C1,C2,···,CN) and a text prompt of camera movement Y. We curate a set of 8 camera movements (left of Figure 5) which are widely used in movie industry. This design achieves two objectives: Firstly, it covers all spatial directions; secondly, it facilitates text-to-video models to take the instruction Y as most of them are trained on movie clips that often contain these camera movement descriptions. These movements include both intra-scene movements, such as moving into a scene, as well as inter-scene transitions, such as pulling out the camera. For each static scene generation example, we randomly assign a layout L to a next-scene generation task. ... When the assigned layout is intra-scene, we perform a replacement of N with P."

### 4.3 三维指标（controllability / quality / dynamics，共 10 项）

总纲（L104–111）：

> "Our evaluation metric, WorldScore, is computed by aggregating three key aspects: controllability, which measures the adherence of the generated worlds w.r.t. control inputs; quality, which measures the fidelity and consistency; dynamics, which measures how much the generated worlds exhibit accurate and stable motions. Each of these aspects comprises a few distinct metrics, leading to a total of 10 metrics that contribute to computing the WorldScore."

两个总分（L383–388）："Our WorldScore metrics include two overall scores: WorldScore-Static which measures only the static world generation capability, and WorldScore-Dynamic which measures dynamic world generation capability in addition to static worlds."

- Controllability（3 项，L391–413）：相机可控性用相机误差，式 (3)（L394–397）："we compute camera errors as follows: ecamera=√eθ·et, (3) where eθ and et are scale-invariant rotation and translation errors with respect to the ground truth trajectory T, respectively."；物体可控性（L400–403）："We evaluate whether the objects specified in the next-scene prompt N appear in the generated next scene. To this end, we measure the success rate of object detection."（用 open-set 检测模型 [44]）；内容对齐（L410–413）："we also assess whether the generated scenes are aligned with the entire text N using CLIPScore [22]."
- Quality（4 项，L430–471）：3D 一致性用 DROID-SLAM（L433–439）："we use DROID-SLAM [72], a standard SLAM method, to estimate dense pixel-wise depth for each frame, and then we compute the reprojection error between a pair of co-visible pixels in consecutive frames."；光度一致性（L454–456）："our photometric consistency metric estimates the optical flow between consecutive frames and computes the Average End-Point Error (AEPE)."；风格一致性（L459–462）："computing the differences (F-norm) between the Gram matrices [17] of the first frame and the last frame"；主观质量（L467–471）："we perform a human study of 400 participants, enumerate different metric combinations, and we pick the combination (CLIP-IQA+ [75] with CLIP Aesthetic [63]) that best matches human preference."
- Dynamics（3 项，L472–517）：运动准确性（L473–483）："motion accuracy, which measures whether the motion specified in the next-scene prompt N occurs in the designated regions. ... the score is calculated by comparing optical flow within the intended region with the flow outside the region. We need to consider the outside flow as it cancels out the global motion caused by unintended camera movements."；运动幅度（L511–512）："ability to create large motions by estimating the optical flow between the consecutive frames of the generated video."；运动平滑度（L513–517）："Temporal jittering is a common failure mode in dynamic world generation. We utilize a standard video frame interpolation model [93] to generate smooth interpolation as ground truth to evaluate the temporal smoothness of generated videos V."
- 聚合（L518–528）："we apply a linear normalization and mapping process based on empirical bounds (Supp. C.8) to ensure that the final scores fall within the range between zero to one, and then we scale it by 100. Then, we compute the arithmetic mean of the dimension scores within control and quality aspects to obtain our WorldScore-Static. Additionally, we further incorporate three dynamics dimension scores into the aggregation, resulting in WorldScore-Dynamic. For 3D scene generation models that do not support dynamic tasks, we assign 0 to each dynamics metric."

## 5) 实验与主结果（Table 2 数值，20 行模型）

被评模型构成（L534–547）："We evaluate 20 available world generation models ... We assess 13 video generation models, including two leading commercial closed-source I2V models—Gen-3 [58] and Hailuo [20], along with 7 well-known open-source I2V models ... and 4 open-source T2V models ... Additionally, we evaluate six well-known 3D scene generation models ... Moreover, we include an open-source 4D generation model, 4D-fy [3]."（派单写"19 模型"与原文不符：Abstract L16–17 与 L534 均写 20 个模型；Table 2 共 20 行——其中 VideoCrafter1 与 CogVideoX 各分 T2V/I2V 两行。以原文为准。）

Table 2 数值（L488–507，列序 = Model | -Static | -Dynamic | CameraCtrl | ObjectCtrl | ContentAlign | 3DConsist | PhotoConsist | StyleConsist | SubjectiveQual | MotionAcc | MotionMag | MotionSmooth）：

| Model | Static | Dynamic | CamCtrl | ObjCtrl | Align | 3D | Photo | Style | SubjQ | MotAcc | MotMag | MotSmooth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gen-3 [58] | 60.71 | 57.58 | 29.47 | 62.92 | 50.49 | 68.31 | 87.09 | 62.82 | 63.85 | 54.53 | 27.48 | 68.87 |
| Hailuo [20] | 57.55 | 56.36 | 22.39 | 69.56 | 73.53 | 67.18 | 62.82 | 54.91 | 52.44 | 63.46 | 27.20 | 70.07 |
| DynamiCrafter [84] | 52.09 | 47.19 | 25.15 | 47.36 | 25.00 | 72.90 | 60.95 | 78.85 | 54.40 | 41.11 | 39.25 | 26.92 |
| VideoCrafter1-T2V [9] | 47.10 | 43.54 | 21.61 | 50.44 | 60.78 | 64.86 | 51.36 | 38.05 | 42.63 | 11.76 | 75.00 | 18.87 |
| VideoCrafter1-I2V [9] | 50.47 | 47.64 | 25.46 | 24.25 | 35.27 | 74.42 | 73.89 | 65.17 | 54.85 | 55.63 | 25.00 | 42.49 |
| VideoCrafter2 [9] | 52.57 | 47.49 | 28.92 | 39.07 | 72.46 | 65.14 | 61.85 | 43.79 | 56.74 | 47.12 | 30.40 | 29.39 |
| T2V-Turbo [41] | 45.65 | 40.20 | 27.80 | 30.68 | 69.14 | 38.72 | 34.84 | 49.65 | 68.74 | 34.87 | 40.09 | 7.48 |
| EasyAnimate [86] | 52.85 | 51.65 | 26.72 | 54.50 | 50.76 | 67.29 | 47.35 | 73.05 | 50.31 | 75.00 | 31.16 | 40.32 |
| Allegro [97] | 55.31 | 51.97 | 24.84 | 57.47 | 51.48 | 70.50 | 69.89 | 65.60 | 47.41 | 54.39 | 40.28 | 37.81 |
| Vchitect-2.0 [14] | 42.28 | 38.47 | 26.55 | 49.54 | 65.75 | 41.53 | 42.30 | 25.69 | 44.58 | 33.59 | 33.81 | 21.31 |
| LTX-Video [19] | 55.44 | 56.54 | 25.06 | 53.41 | 39.73 | 78.41 | 88.92 | 53.50 | 49.08 | 76.22 | 29.95 | 71.09 |
| CogVideoX-T2V [88] | 54.18 | 48.79 | 40.22 | 51.05 | 68.12 | 68.81 | 64.20 | 42.19 | 44.67 | 25.00 | 47.31 | 36.28 |
| CogVideoX-I2V [88] | 62.15 | 59.12 | 38.27 | 40.07 | 36.73 | 86.21 | 88.12 | 83.22 | 62.44 | 69.56 | 26.42 | 60.15 |
| SceneScape [16] | 50.73 | 35.51 | 84.99 | 47.44 | 28.64 | 76.54 | 62.88 | 21.85 | 32.75 | 0.00 | 0.00 | 0.00 |
| Text2Room [24] | 62.10 | 43.47 | 94.01 | 38.93 | 50.79 | 88.71 | 88.36 | 37.23 | 36.69 | 0.00 | 0.00 | 0.00 |
| LucidDreamer [11] | 70.40 | 49.28 | 88.93 | 41.18 | 75.00 | 90.37 | 90.20 | 48.10 | 58.99 | 0.00 | 0.00 | 0.00 |
| WonderJourney [90] | 63.75 | 44.63 | 84.60 | 37.10 | 35.54 | 80.60 | 79.03 | 62.82 | 66.56 | 0.00 | 0.00 | 0.00 |
| InvisibleStitch [12] | 61.12 | 42.78 | 93.20 | 36.51 | 29.53 | 88.51 | 89.19 | 32.37 | 58.50 | 0.00 | 0.00 | 0.00 |
| WonderWorld [91] | 72.69 | 50.88 | 92.98 | 51.76 | 71.25 | 86.87 | 85.56 | 70.57 | 49.81 | 0.00 | 0.00 | 0.00 |
| 4D-fy [3] | 27.98 | 32.10 | 69.92 | 55.09 | 0.85 | 35.47 | 1.59 | 32.04 | 0.89 | 22.22 | 22.88 | 80.06 |

### 4.1 节结论（Observations and Challenges，L549–645）

① 3D 模型静态世界最强（L552–557）："From the WorldScore-Static results, we observe that 3D scene generation models generally perform better, e.g., WonderWorld [91] (72.69) and LucidDreamer [11] (70.40) are the top-2, much better than the best video model CogVideoX-I2V [88] (62.15)." 原因（L591–596）："high camera controllability and, thus, better content alignment due to the larger space they can create, as well as high 3D and photometric consistency. However, they do not allow for the generation of dynamic worlds. When extended to 4D for dynamics, 4D-fy [3] does not perform well, likely due to the intrinsic difficulty in 4D scene generation."

② 视频模型相机可控性弱（L597–600）："Even CogVideoX-T2V [88], the best video generation model in camera controllability (40.22), scored much lower than any 3D/4D generation model. This is the main challenge for video generation models to achieve good static world generation."

③ 开源最优视频模型追平闭源（L604–612）："Comparing CogVideoX-I2V [88], with Gen-3 and Hailuo [20], we observe that CogVideoX-I2V scored even higher than both closed-source models in both WorldScore-Static (62.15) and WorldScore-Dynamic (59.12). However, CogVideoX-I2V is not better than them in every aspect. For instance, we observe that CogVideoX-I2V is better at camera controllability yet worse at object controllability and content alignment."

④ 幅度与平滑的权衡（L613–617）："Comparing motion smoothness and motion magnitude metrics for each method, we observe that larger motion often comes at the cost of lower smoothness, revealing current challenge for video models in maintaining both significant motion and natural transitions."

⑤ 幅度大 ≠ 位置准（L619–624）："The correlation between the motion magnitude and accuracy is weak. This implies that models that can produce large motion do not guarantee correct motion placement to follow instructions. Instead, they could hallucinate unintended camera motion or irrelevant motion."

⑥ 视频模型长序列与室外场景弱（L627–634）："We observe that video generation models struggle significantly with long-sequence (large world generation) tasks. In addition, video models are significantly weaker than 3D models in outdoor scenes, while the gap is smaller in indoor scenes."

⑦ T2V 比 I2V 更易被引导（L635–643）："we observe that T2V models generally have higher scores in the controllability aspect and larger motion magnitude, while I2V models have higher scores in quality aspect. Through empirical examination, we find that this is because T2V models are willing to generate larger camera motion, while I2V models tend to stick to the input image viewpoint. This reveals a challenging in controlling I2V models to generate new scene contents."

人类验证（L530–533）："We validate our metrics by human study. Our results suggest that WorldScore's metrics align with human preference, and WorldScore is robust to different video resolutions and aspect ratios."

## 6) 局限（原文逐字）

原文没有独立的 Limitations 一节（「原文未写」benchmark 自身局限，如评测成本、指标对动态世界之外维度的覆盖等）。最接近"局限"的表述是 Conclusion（L646–656）：

> "The WorldScore benchmark reveals current limitations in world generation approaches. For 3D models, while they excel in static world generation, extending them to 4D representations and incorporating dynamics remains challenging. For video models, the main challenges include controllability, long-sequence generation, and generating outdoor scenes. These insights point to directions for future research: bridging the gap between 3D and 4D representations, developing more robust controllability mechanisms, and designing architectures capable of handling extended scene sequences. We believe the WorldScore benchmark will serve as a valuable tool for measuring progress toward more capable and versatile world generation systems."

（注意：这里说的是被评模型的局限，而非基准本身的局限。）

## 7) 叙事关联【推断】：「dynamics」维度测什么——是否触及力学

依据的原句：

- 定义（L107–109）："dynamics, which measures how much the generated worlds exhibit accurate and stable motions."
- 任务设计（L294–299，见 4.1）：动态任务要求 "the layout L explicitly specifies a fixed camera position without any camera motion" —— 即固定机位、只测场景内运动。
- 运动准确性（L477–483）："motion accuracy, which measures whether the motion specified in the next-scene prompt N occurs in the designated regions ... the score is calculated by comparing optical flow within the intended region with the flow outside the region."
- 运动幅度（L511–512）与平滑度（L513–517）：均基于光流与视频插帧模型。
- 类别（L117–118）："For dynamic worlds, we include 5 distinct types of dynamics such as rigid motion and fluid motion."（Fig. 3 标签：Articulated / Deformable / Fluid / Rigid body / Multi-motion，L264–268）

【推断】① dynamics 维度只测"运动现象"三件事——指定区域有没有动（accuracy）、动得大不大（magnitude）、时序顺不顺（smoothness），全部通过光流（SEA-RAFT/SAM2 掩码）与插帧实现；不测力、质量、动量、碰撞、因果等物理量，因此未触及力学。佐证：正文中 "physical" 仅出现在 L315 "they are either real photos or physically-based rendered images"（描述数据集图像来源）以及参考文献标题（L669 "platform for physical ai"、L786 "A physical law perspective"、L872 "Crafting physical commonsense-based benchmark"）；全文无 force/gravity/collision/mass 字样。

【推断】② 论文在设计上已把"相机运动"与"场景运动"解耦：相机运动归入 controllability（跟地面真值轨迹 T 比旋转/平移误差），dynamics 只针对固定机位下的场景内运动（并用区域外光流抵消意外相机运动，见 L481–483）。这与"世界模拟"叙事中"生成内容应符合物理规律"是不同层级——WorldScore 测的是运动放置正确性，不是物理正确性。

【推断】③ 5 类运动标签（rigid/fluid 等）只是数据集分层，不构成物理一致性评估；判据是"指定对象是否动"，而非"动得是否符合物理"。若后续论文想借 WorldScore 论证"力学/物理涌现"，需要另行补指标——原文未提供。

## 8) 图注抄录（主要）

Figure 1（L57–63，图中标签文字 L55–56）：

> "Figure 1. While existing video benchmarks like VBench [26] rate Models A and B similarly based on single-scene video quality, our WorldScore benchmark differentiates their world generation capabilities by identifying that Model B fails to generate a new scene or follow the instructed camera movement. In https://haoyi-duan.github.io/WorldScore/, we show the videos to explain our WorldScore metrics."

（图中关键数字，L55–56："Model A (80.28) ≈ Model B (83.06)" vs "Model A (70.70) > Model B (46.28)"。）

Figure 2（L190–194）：

> "Figure 2. Overview of the WorldScore benchmark design. Top left: World generation is decomposed into a sequence of next-scene generation tasks, where each step follows a structured world specification defining both spatial layout and semantic content. Bottom left: The unified world specification is used to instruct different types of models, including video generation and 3D/4D generation models. Bottom right: All models output videos for evaluation. Top right: Output videos are evaluated using the WorldScore metrics, which assess three fundamental aspects including controllability, quality, and dynamics."

Figure 3（L269–271）：

> "Figure 3. Showcasing of the current scene images. Top two rows: Static world generation examples are categorized into indoor (first row) and outdoor (second row) scenes, each containing 5 categories. Bottom row: Dynamic world generation examples are divided into 5 motion types. Each dynamic example comes with an annotation of motion mask that indicates where the motion should happen."

Figure 4（L292–293）："Figure 4. Curation on the current scene C. Top: Photorealistic worlds. Bottom: Stylized counterparts."

Figure 5（L325–326）："Figure 5. Curation on layouts L. Left: Camera paths T and text Y. Right: A move-right example."

Figure 6（L425–429）：

> "Figure 6. Typical examples. Top: 3D consistency. The bad example on the right-hand-side has a sudden change in geometry rather than smooth transition. Middle: Photometric consistency. The bad example exhibits severe texture shift in the mountain grassland. Bottom: Motion accuracy. In the good example, the octopus moves while the jellyfish remains static. For bad example on the right, the jellyfish moves while the octopus remains static. A full version of all metrics is in Figure S3 and Figure S4 in supplementary material. In https://haoyi-duan.github.io/WorldScore/, we show videos to explain our WorldScore metrics."

Figure 7（L590）："Figure 7. WorldScore-Static across different subdomains."

Table 1（L94–97）：

> "Table 1. Comparison of Benchmarks. Our WorldScore benchmark is designed to evaluate various world generation approaches including 3D, 4D, I2V and T2V models. It is designed to generate multiple scenes with varying sequence lengths. Our benchmark also features multiple visual styles, accurate camera control evaluation, and 3D consistency evaluation, all of which are important factors in world generation yet currently missing in existing benchmarks."

（Table 1 行：WorldScore 为 3000 例且 Multi-Scene/Unified/Long Seq./Image Cond./Multi-Style/Camera Ctrl./3D Consist. 全 ✓；对照 9 个基准全 ✗，见 L83–93。）

Table 2（L508–510）：

> "Table 2. WorldScore evaluation of 20 world generation models. Top: Close-source video models. Middle: Open-source video models. Bottom two rows: 3D and 4D models. Abbreviations: Ctrl=Controllability, Align=Alignment, Consist=Consistency, Photo=Photometric, Qual=Quality, Acc=Accuracy, Mag=Magnitude, Smooth=Smoothness."

（注："Bottom two rows: 3D and 4D models" 原文如此，但实际表中 3D/4D 共 7 行（6 个 3D + 1 个 4D）。）

附录略读要点（L1090–1677，Supp. A–E，无 F 节）：A. 预处理 wproc 把参考图按模型分辨率裁切、相机轨迹 T 给支持相机输入的模型而文字 Y 给视频模型（L1103–1124）；B. 数据筛选五准则 Quality/Perspective/Similarity/Brightness/Human Judgment（L1136–1154）、7 种风格（"anime, cyberpunk, Chinese ink painting, ukiyo-e, impressionism, post-impressionism, and minecraft"，L1163–1165）、GPT-4o 生成 N 的完整 prompt（L1173–1182 静态、L1215–1223 动态）；C. 各指标公式（S1–S8）与经验界（C.8）——其中内容对齐/主观质量/运动准确/幅度四项用 z-score 重标定，"we apply z-score rescaling, setting the empirical best and worst values so that the performance of selected models falls within the 25 to 75 range"（L1465–1467）；D. 400 人 2AFC 人类验证：CLIP-IQA+ & CLIP Aesthetic 一致性最高 0.637（Table S5，L1584），分辨率鲁棒性差异 ≤0.83（L1554–1555）；E. 可视化。
