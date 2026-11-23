# UniWorld 精读笔记（arXiv:2308.07234）

> 引用规范：英文引用逐字复制自 `auto_docs/papers/txt/UniWorld.txt`，行号按该 txt（共 766 行、15 页）。因换行/OCR 产生的拆词与多余空格已合并（如 "World Mod-els"→"World Models"、"consis-tent"→"consistent"、"mA VE"→"mAVE"）；原文拼写错误（gird、Fundational、ImageNe 等）原样保留。

## 1) 原文信息

- 标题：UniWorld: Autonomous Driving Pre-training via World Models（[1-2]）
- 作者：Chen Min（[3]）；机构：Peking University（[4]）；邮箱 minchen@stu.pku.edu.cn（[5]）
- ID：arXiv:2308.07234v1 [cs.CV] 14 Aug 2023（[46]）
- 发表状态：txt 中未见会议/期刊收录信息，**原文未写**（仅 arXiv v1 标注）。
- 代码：https://github.com/chaytonmin/UniWorld（[24-25]）

## 2) Abstract 全文逐字（[6-25]）

> "In this paper, we draw inspiration from Alberto Elfes' pioneering work in 1989, where he introduced the concept of the occupancy grid as World Models for robots [1]. We imbue the robot with a spatial-temporal world model, termed UniWorld, to perceive its surroundings and predict the future behavior of other participants. UniWorld involves initially predicting 4D geometric occupancy as the World Models for foundational stage and subsequently fine-tuning on downstream tasks. UniWorld can estimate missing information concerning the world state and predict plausible future states of the world. Besides, UniWorld's pre-training process is label-free, enabling the utilization of massive amounts of image-LiDAR pairs to build a Foundational Model. The proposed unified pre-training framework demonstrates promising results in key tasks such as motion prediction, multi-camera 3D object detection, and surrounding semantic scene completion. When compared to monocular pre-training methods on the nuScenes dataset, UniWorld shows a significant improvement of about 1.5% in IoU for motion prediction, 2.0% in mAP and 2.0% in NDS for multi-camera 3D object detection, as well as a 3% increase in mIoU for surrounding semantic scene completion. By adopting our unified pre-training method, a 25% reduction in 3D training annotation costs can be achieved, offering significant practical value for the implementation of real-world autonomous driving. Codes are publicly available at https://github.com/chaytonmin/UniWorld."

## 3) 问题与动机（世界模型预训练的意义）

现状是主流模型用单目预训练，忽视多相机时空关联："current multi-camera 3D perception models [7, 8, 9, 10, 11, 12] usually rely on pre-trained ImageNet models [13] or depth estimation models [7] on monocular images. These models fail to take into account the inherent spatial and temporal correlations presented in multi-camera systems. Additionally, while monocular pre-training enhances the capability of image feature extraction, it does not address the pre-training requirements of subsequent tasks."（[29-33]）

可用的无标注数据："Autonomous driving vehicles collect vast amounts of image-LiDAR pairs, which contain valuable spatial and temporal information. Thus, effectively utilizing these unlabeled image-LiDAR pairs can be beneficial for enhancing the pre-training performance of autonomous driving systems."（[34-36]）

深度估计不够、占用栅格才是统一世界模型："depth estimation methods typically focus on estimating the depth of object surfaces, neglecting the holistic 3D structure of objects and occluded elements. For robot systems, geometric occupancy gird provides a unified and consistent world model for robotic tasks, such as obstacle avoidance, path planning, and navigation [1]."（[40-42]）由此结论："the pre-training of models would yield greater benefits by prioritizing the reconstruction of the 4D geometric occupancy of the surrounding scene, rather than solely emphasizing depth prediction."（[45-46, 76-77]）

人类类比："With an inherent World Model, humans possess the remarkable ability to mentally reconstruct the complete 3D geometry of occluded scenes and anticipate the future motion trajectories of objects in the surrounding scene, which is crucial for recognition and understanding."（[78-80]）

世界模型思想谱系（Elfes 1989 → Ha 2018）："The concept of employing world models by humans, animals, and intelligent systems has a historical foundation in psychology, dating back to the work of Craik in 1943 [45]. Alberto Elfes proposed the geometric occupancy grid as a world model for robot perception and navigation in 1989 [1]. David Ha proposed that world model can be trained quickly in an unsupervised manner to learn a compressed spatial and temporal representation of the environment [46]."（[200-204]）自身定位："we follow the concept of the occupancy grid as a robot's world model introduced in 1989 [1], and propose a label-free spatiotemporal fused world model for autonomous driving by integrating future prediction techniques inspired by methods in reinforcement learning [46] and MILE [53]."（[209-212]）

与 Occ-BEV 的关系（本文是其在时间维上的扩展）："Our previous work Occ-BEV [44] defines the task of multi-camera unified pre-training and reconstructes the 3D static surrounding scene as the fundamental stage. In this work, we extend Occ-BEV to utilize 4D geometric occupancy prediction to learn spatial-temporal world models for vision-based perception."（[195-198]）

作者自述四项贡献（[151-161]）：① "We propose to learn spatial-temporal World Models for unified autonomous driving pre-training, which involves initially reconstructing the 4D surrounding scene as the fundamental stage and subsequently fine-tunes on downstream tasks."（[152-154]）② "UniWorld has the capability to estimate missing information concerning the 3D world state and predict plausible future states of the 4D world."（[155-156]）③ "UniWorld's pre-training process is label-free, enabling the utilization of massive amounts of image-LiDAR pairs collected by autonomous vehicles to build a Fundational Model."（[157-158]）④ "By adopting our unified pre-training method, a 25% reduction in costly 3D annotation costs can be achieved, offering significant practical value for the implementation of real-world autonomous driving."（[159-161]）

## 4) 方法逐机制

**(a) 总体流程**（预训练→解码器丢弃→初始化下游）："the input multi-camera images are transformed to the BEV space using advanced techniques like LSS [16] or Transformer [7], and then a geometric occupancy prediction head is incorporated to learn the 4D occupancy distribution, thereby enhancing the model's understanding of the 4D surrounding scene."（[85-88]）"The decoder was solely used for pre-training, while the well-trained model was utilized to initialize the multi-camera perception models."（[90-91]）方法不绑定特定视角变换："our method is not restricted to any specific view transformation technique."（[223]）

**(b) 4D 占用解码器（3D 世界表示）**："we begin by transforming the BEV features into F′bev ∈ RC′×D×H×W, where D represents the number of height channels, and C = C′ × D. Subsequently, we utilize a 3D decoder specifically designed to generate 4D geometric occupancy. This decoder consists of lightweight 3D convolution layers, with the final layer providing the probability of each voxel containing points. The output of the decoder is denoted as P = {Pi ∈ RD×H×W×1, i = 1, 2, ..., m}. During the pre-training phase, the main objective of the decoder is to reconstruct occupied voxels."（[246-258]）

**(c) 占用标签生成（多帧 LiDAR 融合）**："Considering the sparsity of single-frame LiDAR point clouds and the potential inaccuracies that arise from fusing a large number of frames due to dynamic objects, we adopt a strategy of fusing LiDAR point clouds from selected keyframes to generate occupancy labels."（[260-263]）"The 4D ground truth T = {Ti ∈ {0, 1}D×H×W×1, i = 1, 2, ..., m} is generated based on the occupancy of the voxels, indicating whether each voxel contains points or not. A value of 1 represents an occupied voxel, while a value of 0 indicates a free voxel."（[265-268]）未来可用 NeRF/MVS 替代 LiDAR："In the future, it is also feasible to utilize point clouds generated from 3D scene reconstructions using techniques such as NeRF [59, 60, 61, 62] or MVS [63, 64, 65, 66]."（[290-292]）

**(d) 自监督预训练目标（focal loss 二分类）**："We propose the binary geometric occupancy classification task as part of the pre-training process for multi-camera perception models."（[269-270]）"due to the significant number of empty voxels, predicting occupancy grids presents an imbalanced binary classification challenge. To address this, we employ focal loss for binary occupancy classification"（[271-274]）

公式（[276-285]）：loss = −(1/m)(1/n) Σᵢ₌₁ᵐ Σⱼ₌₁ⁿ αₜ(1 − Pₜⁱʲ)^γ log(Pₜⁱʲ)

"where P ij is the predicted probability of voxel j in the i-th occupancy. n = D × H × W is the total number of voxels and batch is the number of batch sizes. The weighting factor α for positive/negative examples is set as 2 and the weighting factor γ for easy/hard examples is 0.25. αt = α and P ijt = P ij for class 1. αt = 1− α and P ijt = 1− P ij for class 0."（[286-289]）

**(e) 下游迁移（语义占用等）**："we propose to extend our multi-camera unified pre-training algorithm to include the surrounding semantic scene completion task. This involves initially performing geometric occupancy prediction and subsequently fine-tuning the model on the semantic scene completion task."（[298-301]）前置理由："directly predicting the 3D semantics of multi-view images requires a significant amount of 3D semantic annotations for training, which can be expensive and time-consuming."（[296-298]）

**(f) 相对单目预训练的三点优势**（[306-314]）："(1) Spatial-Temporal Integration: By leveraging spatial and temporal information from multiple camera views, the model gains a better understanding of the dynamic nature of the environment, leading to more accurate predictions. (2) Unified Representation: The unified pre-training approach enables the model to learn a shared representation across different camera views, facilitating improved knowledge transfer and reducing the necessity for task-specific pre-training. (3) Perception of Occluded Areas: Monocular depth estimation can only predict surface positions of objects, while our proposed multi-camera unified pre-training method enables comprehensive 3D reconstruction of occluded objects."

**(g) 相对知识蒸馏**："Our unique pre-training algorithm eliminates the need for annotations or pre-trained LiDAR detection models, significantly reducing the 3D annotation requirements."（[320-321]）

**(h) 实验设置**："We performed pre-training for a total of 24 epochs. The occupancy decoder consists of two layers of 3D convolutional layers."（[347-348]）"All experiments were conducted using 8 Nvidia Tesla A40 GPU cards."（[350]）

## 5) 实验与主结果（数值全部照抄，未四舍五入）

数据集："We conducted extensive experiments on the nuScenes dataset [17]."（[324]）

**Table 1**（[328-334]）"Quantitative multi-task learning performance on the nuScenes validation set."（[328]）列：NDS↑ / MAP↑ / Divider↑ / Ped Cross↑ / Boundary↑ / mIoU↑ / IoU↑ / VPQ↑（[330]）

| Method | Pre-train | NDS↑ | MAP↑ | Divider↑ | Ped Cross↑ | Boundary↑ | mIoU↑ | IoU↑ | VPQ↑ |
|---|---|---|---|---|---|---|---|---|---|
| BEVerse [78] | ImageNet [13] | 0.466 | 0.321 | 53.2 | 39.0 | 53.9 | 48.7 | 38.7 | 33.3 |
| ImageNet + UniWorld-3D | | 0.483+1.7% | 0.334+1.3% | 54.8+1.6 | 40.4+1.4 | 55.0+1.1 | 50.0+1.3 | 39.4+0.7 | 34.1+0.8 |
| ImageNet + UniWorld-4D | | 0.484+1.8% | 0.331+1.0% | 54.7+1.5 | 40.1+1.1 | 54.9+1.0 | 49.9+1.2 | 40.5+1.8 | 35.0+1.7 |

正文运动预测结论："In the motion prediction task, UniWorld's improvement over Occ-BEV results in a 1% increase in IoU and VPQ."（[358-359]）Intro 中写 "our multi-camera unified pre-training algorithm exhibits a 1.8% increase in IoU and a 1.7% improvement in VPQ compared to monocular approaches"（[100-101]，即 Table 1 UniWorld-4D 行）。注：Abstract 称约 1.5% IoU（[20]），与正文 1.8%（[100]）不一致，两处均系原文。

**Table 2**（[335-346]）"Quantitative multi-camera 3D object detection performance on the nuScenes validation set."（[335]）列：mAP↑ / NDS↑ / mATE↓ / mASE↓ / mAOE↓ / mAVE↓ / mAAE↓（[336]）

| Type | Method | Pre-train | Backbone | Image Size | CBGS | mAP↑ | NDS↑ | mATE↓ | mASE↓ | mAOE↓ | mAVE↓ | mAAE↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer [7] | DETR3D [7] | FCOS3D [79] | R101-DCN | 900×1600 | ✓ | 0.349 | 0.434 | 0.716 | 0.268 | 0.379 | 0.842 | 0.200 |
| | FCOS3D + UniWorld-3D | R101-DCN | 900×1600 | ✓ | 0.360+1.1% | 0.461+2.7% | 0.701 | 0.260 | 0.372 | 0.730 | 0.188 |
| BEVFormer [8] | ImageNet [13] | R101-DCN | 900×1600 | × | 0.371 | 0.479 | 0.688 | 0.275 | 0.441 | 0.382 | 0.195 |
| | ImageNe+UniWorld-3D（原文如此）| R101-DCN | 900×1600 | × | 0.397+2.6% | 0.500+2.1% | 0.680 | 0.272 | 0.392 | 0.380 | 0.193 |
| | ImageNe+UniWorld-4D（原文如此）| R101-DCN | 900×1600 | × | 0.390+1.9% | 0.501+2.2% | 0.683 | 0.280 | 0.394 | 0.374 | 0.192 |
| | FCOS3D [79] | R101-DCN | 900×1600 | × | 0.416 | 0.517 | 0.673 | 0.274 | 0.372 | 0.394 | 0.198 |
| | FCOS3D + UniWorld-3D | R101-DCN | 900×1600 | × | 0.438+2.2% | 0.534+1.7% | 0.656 | 0.271 | 0.371 | 0.348 | 0.183 |
| | FCOS3D + UniWorld-4D | R101-DCN | 900×1600 | × | 0.432+1.6% | 0.530+1.3% | 0.659 | 0.274 | 0.375 | 0.344 | 0.188 |

正文："Occ-BEV surpassed FCOS3D [79] on DETR3D [7] by achieving a 2.7% increase in NDS and 1.1% in mAP. Additionally, it outperformed BEVFormer [8] with a 2.6% improvement in NDS and 2.1% in mAP for ImageNet baseline, a 1.7% improvement in NDS and 2.2% in mAP for FCOS3D baseline."（[363-366]）注：正文把这些数字归给 Occ-BEV，但表 2 对应行标注为 UniWorld-3D（[338-346]），原文不一致。收敛曲线："Our unified pre-training Occ-BEV significantly enhances BEVFormer [8] at the initial epoch, achieving a 4% increase in NDS."（[367-369]，图 3）4D vs 3D："Compared to the 3D reconstruction pretraining of Occ-BEV [44], the introduction of 4D occupancy prediction in UniWorld resulted in slightly inferior performance in the 3D object detection task. This discrepancy is likely attributed to the fact that 4D prediction allows the model to learn future information but introduces uncertainty in the object's position."（[370-374]）

**Table 3**（[381-384]）"Quantitative multi-camera 3D object detection performance on the nuScenes test set."（[381]）

| Method | Pre-train | Backbone | Image Size | CBGS | mAP↑ | NDS↑ | mATE↓ | mASE↓ | mAOE↓ | mAVE↓ | mAAE↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DETR3D [7] | DD3D [14] | v2-99 | 900×1600 | ✓ | 0.412 | 0.479 | 0.641 | 0.255 | 0.394 | 0.845 | 0.133 |
| DD3D + UniWorld-3D | v2-99 | 900×1600 | ✓ | 0.431+1.9% | 0.496+1.7% | 0.621 | 0.257 | 0.407 | 0.783 | 0.123 |

正文："our multi-camera unified pre-training method demonstrated a significant improvement of about 1.8% in both mAP and NDS compared to the DETR3D [7] pre-trained on DD3D [14] for depth estimation"（[378-379]）——表内实际标注 +1.9%/+1.7%（[384]），原文不一致。且 "As UniWorld's performance in the 3D object detection task is slightly inferior to Occ-BEV, we do not present its results on the test set here."（[413-414]）

**Table 4**（[385-406]）"Quantitative segmentation performance on the 3D occupancy prediction challenge [80]."（[385-386]）类别列（[387-403]，其中 "others" 出现两次，原文如此）：mIoU, others, barrier, bicycle, bus, car, construction, motorcycle, pedestrian, traffic-cone, trailer, truck, driveable, others, sidewalk, terrain, manmade, vegetation

| Methods | Pre-train | Backbone | Image Size | mIoU↑ | others | barrier | bicycle | bus | car | construction | motorcycle | pedestrian | traffic-cone | trailer | truck | driveable | others | sidewalk | terrain | manmade | vegetation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BEVFormer [8] | FCOS3D [79] | R101-DCN [54] | 900×1600 | 23.70 | 10.24 | 36.77 | 11.70 | 29.87 | 38.92 | 10.29 | 22.05 | 16.21 | 14.69 | 27.44 | 23.13 | 48.19 | 33.10 | 29.80 | 17.64 | 19.01 | 13.75 |
| BEVStereo [28] | BEVDet4D [9] | Swin-B [81] | 512×1408 | 42.78 | 22.45 | 47.95 | 28.13 | 40.29 | 53.79 | 27.60 | 35.18 | 29.64 | 31.69 | 45.49 | 37.71 | 81.88 | 49.16 | 55.03 | 51.00 | 50.87 | 39.44 |
| BEVStereo [28] | UniWorld-3D | Swin-B [81] | 512×1408 | 45.92+3.14% | 26.21 | 53.06 | 33.41 | 42.77 | 56.57 | 28.99 | 39.92 | 32.31 | 34.89 | 49.59 | 40.28 | 82.88 | 52.29 | 57.77 | 53.58 | 53.94 | 42.25 |

正文："As shown in Table 4, on the 3D occupancy prediction challenge [80], our algorithm achieved a 3% improvement in mIoU compared to BEVStereo [28]"（[426-427]）；注：正文又称该方法为 Occ-BEV（[421-422]），表 4 行标注为 UniWorld-3D，原文不一致。

**Table 6**（[467-471]）"Comparison with BEVDistill [75]."（[471]）

| Methods | mAP↑ | NDS↑ |
|---|---|---|
| BEVFormer [8] | 0.352 | 0.423 |
| BEVDistill [75] | 0.386 | 0.457 |
| UniWorld-3D | 0.389 | 0.459 |

正文："our method demonstrates comparable performance to the knowledge distillation method trained on annotated LiDAR point clouds data. It is worth noting that our approach offers higher efficiency and broader applicability since it does not rely on data annotation or training of LiDAR point clouds models as BEVDistill [75]."（[416-419]）注：表 6 的 BEVFormer 基线（0.352/0.423）与表 2/7（0.416/0.517）不同，原文未解释。

## 6) 消融（全部抄录）

**4.3.1 数据效率（图 4）**："when BEVFormer is trained with 75% of the labeled data, it achieves the same performance as it trained on the complete dataset. Moreover, even with only 25% of the samples available for fine-tuning, our UniWorld model outperforms BEVFormer by 1% in mAP, highlighting its remarkable data efficiency and its potential to reduce the reliance on expensive human-annotated 3D data."（[486-490]）

**4.3.2 多帧融合（Table 5，[454-465]）**："Table 5: Impacts of multi-frame fusion. The occupancy labels were obtained by merging LiDAR point clouds from the current frame, preceding keyframes, succeeding keyframes, and their corresponding non-keyframes."（[459-465]）

| Frames | mAP↑ | NDS↑ | mATE↓ |
|---|---|---|---|
| 0 | 0.416 | 0.517 | 0.673 |
| 1 | 0.417 | 0.523 | 0.661 |
| 3 | 0.438 | 0.534 | 0.656 |
| 5 | 0.429 | 0.525 | 0.659 |

注：表内帧数标注为 0/1/3/5（原文如此），正文称比较 "single-frame fusion, 3-frame fusion and 5-frame fusion (including the corresponding non-key frames)"（[493-494]），原文不一致。结论："the model's accuracy initially improved with an increasing number of fused point clouds but started to decline afterwards... excessive fusion of frames introduces uncertainty due to the presence of dynamic objects. This uncertainty can lead to errors in the fusion process and subsequently lower the accuracy of the model."（[495-500]）

**4.3.3 显式语义监督（Table 7，[472-476]）**："Table 7: Semantic occupancy supervision."（[476]）

| Methods | mAP↑ | NDS↑ | mATE↓ |
|---|---|---|---|
| BEVFormer [8] | 0.416 | 0.517 | 0.673 |
| UniWorld-3D | 0.438 | 0.534 | 0.656 |
| Supervision | 0.445 | 0.544 | 0.648 |

结论："The results in Table 7 demonstrate that incorporating explicit supervision leads to a notable improvement of 3% in mAP and NDS compared to BEVFormer [8]. Furthermore, when compared to unlabeled multi-frame fused point cloud pre-training, there is a 1% increase in mAP. These findings highlight the potential of leveraging labeled data for explicit occupancy prediction supervision."（[505-508]）

**定性评估（图 5）**："using single-frame point clouds as the supervision for occupancy grid generation results in incomplete reconstructions due to the sparse nature of the LiDAR point clouds. On the other hand, using three keyframes and their corresponding non-key frames as the supervision information allows for more complete reconstructions of scenes."（[512-514, 521-522]）

## 7) 局限/未来工作（[523-531] 原文逐字）

> "Limitations. Although our multi-camera unified pre-training approach has demonstrated promising results, there are several limitations to consider: (1) Currently, UniWorld-4D's performance in the 3D object detection task is slightly inferior to UniWorld-3D. It is necessary to address the issue of decreased 3D static detection performance caused by temporal information prediction. (2) The 3D convolutions in the decoder limit its applicability to tasks requiring high-resolution occupancy reconstruction. We will explore the cascade refine strategy. (3) We rely on LiDAR to obtain ground truth occupancy grids. In the future, we will explore the NeRF [59, 60, 61, 62] and MVS [63, 64, 65, 66] algorithms to reconstruct 3D scenes solely from multi-view images and obtain ground truth data."

结论段未来工作："Future work should focus on addressing the limitations mentioned and further improving the performance and applicability of our approach in real-world autonomous driving scenarios."（[539-541]）

## 8) 叙事关联【推断】：世界模型作为预训练范式（而非直接部署）的例证

1.【推断】UniWorld 把世界模型明确定位为「基础阶段表征学习 + 下游微调」，即预训练范式，而非部署时直接使用的组件。支持原句："Our approach leverages the intuitive concept of using the multi-camera system to learn a compressed spatial and temporal representation of the environment (i.e., World models) as the foundational stage, followed by fine-tuning downstream tasks."（[82-85]）；"UniWorld involves initially predicting 4D geometric occupancy as the World Models for foundational stage and subsequently fine-tuning on downstream tasks."（[10-12]）

2.【推断】世界模型解码器只服务于预训练，预训练后即被丢弃，最终部署/下游保留的是编码器——证明其作用是「表征注入」而非「行为生成」。支持原句："The decoder was solely used for pre-training, while the well-trained model was utilized to initialize the multi-camera perception models."（[90-91]）；"After pre-training, the lightweight decoder is discarded, and the encoder is used to warm up the backbones of downstream tasks."（[144-145]）

3.【推断】与 RL 中依赖奖励与在线环境交互的世界模型用法形成对照：UniWorld 不需要 reward 或在线交互，仅以「预测未来占用」作离线自监督代理任务。支持原句："Methods in [47, 48, 49] presuppose access to rewards and online interaction with the environment from predictions in the compact latent space of a world model in reinforcement learning."（[204-206]）；"UniWorld's pre-training process is label-free, enabling the utilization of massive amounts of image-LiDAR pairs collected by autonomous vehicles to build a Fundational Model."（[157-158]）

4.【推断】与 MILE（以世界模型直接驱动模仿学习驾驶）不同，UniWorld 仅借用「预测未来」这一机制做预训练，世界模型不直接参与驾驶决策。支持原句："MILE [53] proposed to build the world model by predicting the future BEV segmentation from high-resolution videos of expert demonstrations for autonomous driving."（[207-209]）；"we follow the concept of the occupancy grid as a robot's world model introduced in 1989 [1], and propose a label-free spatiotemporal fused world model for autonomous driving by integrating future prediction techniques inspired by methods in reinforcement learning [46] and MILE [53]."（[209-212]）

5.【推断】预训练收益在检测、运动预测、语义占用等多下游任务上同时体现，表明世界模型被当作通用基础表征来源（Foundational Model 路线），而非为某一部署任务定制。支持原句："The proposed unified pre-training framework demonstrates promising results in key tasks such as motion prediction, multi-camera 3D object detection, and surrounding semantic scene completion."（[15-18]）；"Pre-training via World Model with unlabeled image-LiDAR pairs offers promising opportunities for reducing the 25% dependency on annotated 3D data and establishing a Foundational Model for autonomous driving and robotics."（[537-539]）

## 9) 图注/表注逐字

- Fig. 1（[72-75]）："Figure 1: Comparison between monocular pre-training and our unified multi-camera pre-training. Monocular pre-training only enhances the capability of the feature extraction from a single view, whereas our proposed multi-view unified pre-training enables the incorporation of temporal and spatial information from multi-view images through World Models for pre-training."
- Fig. 2（[140-145]）："Figure 2: The overall architecture of the proposed multi-camera unified pre-training method UniWorld. We first transform the multi-frame large-scale irregular LiDAR point clouds into volumetric representations as the 4D geometric occupancy labels, then add an occupancy decoder with some layers of 3D convolutions to the BEV encoder. We apply binary occupancy classification as the pretext task to distinguish whether the 4D voxel contains points. After pre-training, the lightweight decoder is discarded, and the encoder is used to warm up the backbones of downstream tasks."
- Fig. 3（[440]）："Figure 3: Performance curves."（对应正文 [366-367]：BEVFormer + FCOS3D 基线收敛曲线）
- Fig. 4（[453]）："Figure 4: Label-efficiency."
- Fig. 5（[520]）："Figure 5: Visualization of scene reconstruction via occupancy prediction."
- 图内坐标轴文字（OCR 自 txt）：Fig. 3 横轴 "2 6 10 14 18 22 26"（[430]）、纵轴 "28 32 36 40 44 48 mAP"（[432-438]）；Fig. 4 横轴 "0.25 0.50 0.75 1.00 Percentage of labeled data"（[441-442]）、纵轴 "42 … 58 NDS"（[443-452]）。
- 表注：Table 1（[328]）、Table 2（[335]）、Table 3（[381]）、Table 4（[385-386]）、Table 5（[459-465]）、Table 6（[471]）、Table 7（[476]）——均已在上文逐字抄录。
