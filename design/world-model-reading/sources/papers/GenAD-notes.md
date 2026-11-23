# GenAD 精读笔记

> 数据源：`auto_docs/papers/txt/GenAD.txt`（arXiv:2402.11502v3，共 930 行 / 10 页）。
> 引用行号为 txt 行号；论文未写的内容一律标注「原文未写」，解读标注【推断】。
> 结构要求的必抄部分（Abstract 全文、图注/表注、数值表）不计入讲解字数。

## 1. 原文信息

- 标题（L1 逐字）："GenAD: Generative End-to-End Autonomous Driving"
- 作者（L2 逐字）："Wenzhao Zheng1,* Ruiqi Song2,3,* Xianda Guo2,*,† Chenming Zhang2 Long Chen2,3,†"
- 机构（L3-4 逐字）："1University of California, Berkeley 2Waytous 3Institute of Automation, Chinese Academy of Sciences"
- 通讯/共一（L41 逐字）："∗ Equal contributions; † Corresponding authors."（L2 中 † 标于 Xianda Guo 与 Long Chen【推断】）
- ID（L85 逐字）："arXiv:2402.11502v3  [cs.CV]  7 Apr 2024"
- 发表状态：会议/期刊出处 原文未写（txt 仅含 arXiv 版本信息）
- 代码（L36 逐字）："Code: https://github.com/wzzheng/GenAD."

## 2. Abstract 全文逐字（L7-36，仅拼接原换行断词）

> "Directly producing planning results from raw sensors has been a long-desired solution for autonomous driving and has attracted increasing attention recently. Most existing end-to-end autonomous driving methods factorize this problem into perception, motion prediction, and planning. However, we argue that the conventional progressive pipeline still cannot comprehensively model the entire traffic evolution process, e.g., the future interaction between the ego car and other traffic participants and the structural trajectory prior. In this paper, we explore a new paradigm for end-to-end autonomous driving, where the key is to predict how the ego car and the surroundings evolve given past scenes. We propose GenAD, a generative framework that casts autonomous driving into a generative modeling problem. We propose an instance-centric scene tokenizer that first transforms the surrounding scenes into map-aware instance tokens. We then employ a variational autoencoder to learn the future trajectory distribution in a structural latent space for trajectory prior modeling. We further adopt a temporal model to capture the agent and ego movements in the latent space to generate more effective future trajectories. GenAD finally simultaneously performs motion prediction and planning by sampling distributions in the learned structural latent space conditioned on the instance tokens and using the learned temporal model to generate futures. Extensive experiments on the widely used nuScenes benchmark show that the proposed GenAD achieves state-of-the-art performance on vision-centric end-to-end autonomous driving with high efficiency. Code: https://github.com/wzzheng/GenAD."

## 3. 问题与动机：生成式端到端的论证

中文讲解：两条核心论证。①串行管线无法建模「自车未来动作 ↔ 他车未来动作」的高阶双向交互：自车变道影响后车行为、后车行为又反过来影响自车规划。②真实轨迹高度结构化、共享共同先验，但现有方法直接从特征解码轨迹、忽略该先验。GenAD 的解法是把自动驾驶整体建模为「轨迹生成问题」。

- L12-16："However, we argue that the conventional progressive pipeline still cannot comprehensively model the entire traffic evolution process, e.g., the future interaction between the ego car and other traffic participants and the structural trajectory prior."
- L78-86："the serial design of prediction and planning of existing pipelines ignores the possible future interactions between the ego car and the other traffic participants. We argue that this type of interaction is important for accurate planning. For example, the lane shift of the ego car would affect the action of the rear cars, and further affects the planning of the ego car."
- L88-92："Also, future trajectories are highly structured and share a common prior (e.g., most trajectories are continuous and straight lines). Still, most existing methods fail to consider this structural prior, leading to inaccurate predictions and planning."
- L93-97："In this paper, we propose a Generative End-to-End Autonomous Driving (GenAD) framework ..., which models autonomous driving as a trajectory generation problem to unleash the full potential of end-to-end methods."

解读：两条论证指向同一个解法——把「自车与他车未来轨迹的联合演化」整体当作生成对象，而不是先预测他车、再串行规划自车。这意味着预测与规划的监督可以共享同一个轨迹输出空间，为第 8 节的叙事关联埋下伏笔。

## 4. 方法逐机制

### 4.1 实例中心场景表征（Instance-Centric Scene Representation，Sec 3.1）

目标形式化（L265-274）：给定过去 p 帧传感器输入与自车轨迹，规划未来 f 帧自车轨迹 "T(T − p, p+ 1), S → T(T, f), (1)"。
- Image→BEV：沿用 BEVFormer。"We basically follow BEVFormer [27] to obtain the BEV features."（L283-284）；"We then initialize H × W BEV tokens B0 as queries and use deformable cross-attention [56] to transfer information from the multi-scale image feature F: B = DA(B0, F, F), (2)"（L287-290）。过去 p 帧 BEV 对齐到当前坐标系并拼接（L294-297）。
- BEV→Map：可学习 map token 经全局 cross-attention 更新："we use map tokens M ∈ R^{Nm∗C} to represent semantic maps"（L300-301）；"M = CA(M0, B, B), (3)"（L310）。三类地图元素（L305-307 逐字）："we consider three categories of map elements (i.e., lane divider, road boundary, and pedestrian crossing)"。
- BEV→Agent："A = DA(A0, B, B), (4)"（L320）；检测头解码位置/朝向/类别（L322-324）。
- 实例中心表征（L325-353）：先加 ego token，"We first add an ego token e to the learned agent tokens A to construct a set of instance tokens I = concat(e, A)."（L329-331）；再自注意力 "I ← SA(I, I, I), (5)"（L340）实现 ego-agent 高阶交互；最后注入地图 "I ← CA(I, M, M). (6)"（L350）。意义（L351-353）："The learned instance tokens I incorporate high-order agent-ego interactions and are aware of the learned semantic maps, which are compact yet contain all the necessary map and instance information to perform motion prediction and planning."

解读：机制分工上，deformable cross-attention 用于 BEV→agent（以几何定位聚合特征），global cross-attention 用于 BEV→map（全局语义更新），ego token 加入后经自注意力实现 ego 与 agent 的双向交互，再用 cross-attention 把地图信息注入实例 token——三者组合得到「紧凑但完整」的场景描述。这里 L329-331 的 ego token 拼接正是「预测+规划联合」的结构入口。

### 4.2 轨迹先验建模（Trajectory Prior Modeling，Sec 3.2，VAE）

预测与规划被论证为同一问题（L381-385）："We find that the objectives of motion prediction of other agents and planning of the ego vehicle share the same output space and are essentially the same. They both aim to produce a high-quality realistic trajectory of the concerned instance, given semantic maps and interactions with other agents." 用 VAE 把真实轨迹编码为潜空间高斯分布："we adopt a variational autoencoder (V AE) [24] architecture to learn a latent space Z to model this trajectory prior."（L399-401）；"The encoder ef outputs two vectors µf and σf representing the mean and variance of the Gaussian distribution: p(z|T(T, f)) ∼ N (µf , σf ), (7)"（L406-408）。作用（L411-414）："The learned distribution p(z|T(T, f)) contains the prior of the ground-truth trajectories, which can be leveraged to improve the authenticity of motion prediction and planning for traffic agents and ego vehicles."

### 4.3 潜空间未来轨迹生成（Latent Future Trajectory Generation，Sec 3.3，自回归）

分解联合分布为逐帧条件概率（L424-426，Eq 8）："p(T(T, f)|zT ) = p(wT +1|zT ) · p(wT +2|wT +1, zT ) · · · p(wT +f |wT +1, · · ·, wT +f −1, zT )"；采样起点（L427-428）："We sample a vector from the distribution N (µf , σf ) as the latent state at the current time stamp zT."；MLP 解码单点（L429-431）："Instead of decoding the whole trajectory at once, we adopt a simple MLP-based decoder dw to decode a waypoint w = dw(z) from the latent space Z"；GRU 自回归推进（L433-437）："We then adopt a gated recurrent unit (GRU) [7] as the future trajectory generator to model the temporal evolutions of instances. Specifically, the GRU model g takes as inputs the current latent representation zt and transforms it into the next state g(zt) = zt+1."

### 4.4 总体框架与训练/推理（Sec 3.4）

解读：选择 VAE 而非直接从特征解码轨迹，依据是未来轨迹本身具有不确定性——L389-392："Different from existing methods which directly output the trajectory using a simple decoder, we model it as a trajectory generation problem T ∼ p(T|I) considering its uncertain nature." 高斯潜变量一方面编码结构先验，另一方面允许推理时多次采样得到多条可能未来（L114-116）。

### 4.4 总体框架与训练/推理（Sec 3.4）

- 轨迹重建损失（L485-493，Eq 9）："Jprior = Ltra( ˆTe, Te) + 1/Na Ltra( ˆTa, Ta) + λcLf ocal( ˆCa, Ca)"；"Ltra denotes the trajectory losses [21] including the L1 discrepancy, ego-agent collision constraint, ego-boundary overstepping constraint, and the ego-Lane directional constraint."
- 分布匹配：实例编码器 ei 输出 "p(z|I) ∼ N (µi, σi). (10)"（L502）；KL 损失（L523-527，Eq 11）："Jplan = DKL (p(z|I), p(z|T(T, f)))"；依据（L503-505）："With the learned latent trajectory space to model realistic trajectory prior, the motion prediction and planning can be unified and formulated as a distribution matching problem between the instance distribution p(z|I) and the ground-truth distribution p(z|T(T, f))."
- 辅助任务：map 分割 Jmap 与 3D 检测 Jdet（L529-537）。总损失（L540，Eq 12）："JGenAD = Jprior + λplanJplan + λmapJmap + λdetJdet"。
- 推理（L543-547）："For inference, we discard the future trajectory encoder ef and sample a latent state following the instance distribution p(z|I) as input for the trajectory generator g and waypoint decoder dw."

## 5. 实验与主结果

数据（L557-565）：nuScenes，"composed of 1000 driving scenes, where each scene provides RGB and LiDAR video of 20 seconds. The ego vehicle is equipped with 6 surrounding cameras with 360◦ horizontal FOV and a 32-beam LiDAR sensor. nuScenes provides semantic map and 3D object detection annotations for keyframes at 2Hz. It includes 1.4M 3D bounding boxes of objects from 23 categories." 划分 700/150/150。
指标（L567-576）：L2 displacement error 与 collision rate；"By default, we take as inputs 2s history (i.e., 5 frames) and evaluate the planning performance at the 1s, 2s, and 3s future."
实现（L578-610）：ResNet50 骨干、640×360 输入、200×200 BEV；BEV/map/agent token 数 100×100/100/300，map token 含 20 point tokens；隐维 256，潜空间与 GRU 隐维 512；每注意力块 3 层；AdamW、cosine 调度、lr 2×10−4、weight decay 0.01；60 epochs、8×RTX 3090、总 batch 8。

### Table 1（L510-522）逐数值：L2 (m)↓ 1s/2s/3s/Avg；Collision Rate (%)↓ 1s/2s/3s/Avg；FPS

| Method | Input | L2 1s | 2s | 3s | Avg | CR 1s | 2s | 3s | Avg | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| IL [43] | LiDAR | 0.44 | 1.15 | 2.47 | 1.35 | 0.08 | 0.27 | 1.95 | 0.77 | - |
| NMP [52] | LiDAR | 0.53 | 1.25 | 2.67 | 1.48 | 0.04 | 0.12 | 0.87 | 0.34 | - |
| FF [14] | LiDAR | 0.55 | 1.20 | 2.54 | 1.43 | 0.06 | 0.17 | 1.07 | 0.43 | - |
| EO [23] | LiDAR | 0.67 | 1.36 | 2.78 | 1.60 | 0.04 | 0.09 | 0.88 | 0.33 | - |
| ST-P3 [15] | Camera | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 | 1.6 |
| UniAD [16] | Camera | 0.48 | 0.96 | 1.65 | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 | 1.8 |
| OccNet [47] | Camera | 1.29 | 2.13 | 2.99 | 2.14 | 0.21 | 0.59 | 1.37 | 0.72 | 2.6 |
| VAD-Tiny [21] | Camera | 0.60 | 1.23 | 2.06 | 1.30 | 0.31 | 0.53 | 1.33 | 0.72 | 6.9‡ |
| VAD-Base [21] | Camera | 0.54 | 1.15 | 1.98 | 1.22 | 0.04 | 0.39 | 1.17 | 0.53 | 3.6‡ |
| GenAD | Camera | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 | 6.7‡ |
| VAD-Tiny† | Camera | 0.46 | 0.76 | 1.12 | 0.78 | 0.21 | 0.35 | 0.58 | 0.38 | 6.9‡ |
| VAD-Base† | Camera | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.17 | 0.41 | 0.22 | 3.6‡ |
| GenAD† | Camera | 0.28 | 0.49 | 0.78 | 0.52 | 0.08 | 0.14 | 0.34 | 0.19 | 6.7 |

（†=对所有预测帧求平均后的指标；Aux. Sup. 列：UniAD 为 "Map & Box & Motion & Tracklets & Occ"，OccNet 为 "3D-Occ & Map & Box"，其余 Camera 方法为 "Map & Box & Motion"，L515-519。）
结论（L615-617）："We see that our GenAD achieves the best L2 errors among all the methods with an efficient inference speed." 对 UniAD 碰撞率更低的说明（L617-623）："Though UniAD [16] outperforms our method with respect to the collision rates, it employs additional supervision signals during training such as tracking and occupancy information, which has been verified to be vital to avoid collision [16]. However, these labels in the 3D space are difficult to annotate, rendering it not trivial to use less labels to achieve competitive performance." 效率（L624-625）："Our GenAD is also more efficient than UniAD, demonstrating a strong performance/speed trade-off."

### Table 2（L583-585）逐数值：感知/预测/规划

| Method | mAP | mAP@0.5 | mAP@1.0 | mAP@1.5 | EPA(car) | EPA(ped.) | Avg L2 | Avg CR | FPS |
|---|---|---|---|---|---|---|---|---|---|
| VAD [21] | 0.27 | 0.15 | 0.44 | 0.61 | 0.56 | 0.29 | 1.30 | 0.72 | 6.9† |
| GenAD | 0.29 | 0.24 | 0.56 | 0.71 | 0.59 | 0.34 | 0.91 | 0.43 | 6.7† |

结论（L646-652）："We observe that GenAD outperforms V AD on all the tasks with a similar inference speed. Specifically, GenAD achieves better motion prediction performances by considering the influence of the ego vehicle on the other agents. GenAD also demonstrates superior performance in 3D detection and map segmentation, showing a better consistency between perception, prediction, and planning."

## 6. 消融（全部抄录）

### Table 3（L589-593）：实例中心场景表征消融（E → A = ego-to-agent 交互）

| Setting | L2 1s | 2s | 3s | Avg | CR 1s | 2s | 3s | Avg |
|---|---|---|---|---|---|---|---|---|
| VAD [21] | 0.60 | 1.23 | 2.06 | 1.30 | 0.31 | 0.53 | 1.33 | 0.72 |
| w/ E → A | 0.39 | 0.99 | 1.91 | 1.10 | 0.14 | 0.39 | 1.35 | 0.63 |
| GenAD | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |
| w/o E → A | 0.40 | 0.93 | 1.74 | 1.02 | 0.70 | 0.98 | 1.97 | 1.22 |

结论（L656-665）："We first added the ego-to-agent interaction with the proposed method to V AD-tiny [21], and observe a large improvement in both the L2 error and the collision rate."；"We also removed the ego-to-agent interaction in our GenAD model by masking the self-attention matrix to dissect its effect. We see that the collision rate performance drops greatly. We think this is because without considering the high-order interactions between the ego car and the other agents, it becomes very difficult to learn the true underlying distributions of trajectories."

### Table 4（L631-635）：生成式框架消融（TPM=轨迹先验建模；LFTG=潜空间未来轨迹生成）

| TPM | LFTG | L2 1s | 2s | 3s | Avg | CR 1s | 2s | 3s | Avg |
|---|---|---|---|---|---|---|---|---|---|
| × | × | 0.42 | 1.02 | 1.89 | 1.11 | 0.06 | 0.43 | 1.60 | 0.70 |
| ✓ | × | 0.38 | 0.92 | 1.76 | 1.02 | 0.23 | 0.37 | 1.19 | 0.60 |
| × | ✓ | 0.39 | 0.92 | 1.74 | 1.02 | 0.18 | 0.45 | 1.05 | 0.56 |
| ✓ | ✓ | 0.36 | 0.83 | 1.55 | 0.91 | 0.06 | 0.23 | 1.00 | 0.43 |

结论（L675-683）："When using TPM alone, we directly decode the entire trajectory from the latent space. With only the LFTG module, we use a gated recurrent unit to gradually produce waypoints given the instance-centric scene representation. We see that both modules are effective and improve the planning performance. Combining the two modules further improves the performance by a large margin. This verifies the importance of factorizing the joint distribution as in (8) to release the full potential of latent trajectory prior modeling."

可视化结论（L689-694）："We see that GenAD produces better and safer trajectories than V AD in various scenarios including going straight, overtaking, and turning. For the challenging scenarios when multiple agents are involved in complex traffic scenes, our GenAD still demonstrates good results while V AD cannot safely move through."

## 7. 局限 / 未来工作

论文无独立 Limitations 节（原文未写）。隐性局限说明见 L621-623（3D 标签难标注）与 L617-621（UniAD 靠 tracking/occupancy 额外监督才在碰撞率上更好）。未来工作（L706-708 逐字）："In the future, it is interesting to explore other generative modeling methods such as generative adversarial networks or diffusion models for end-to-end autonomous driving." 结论段自述贡献（L699-702）："We have investigated the conventional serial design of perception, prediction, and planning for autonomous driving and proposed a generative framework to enable high-order ego-agent interactions and produce more accurate future trajectories with learned structural prior."

## 8. 叙事关联【推断】：预测（T1）+ 规划（T5）联合的代表

1. 【推断】GenAD 是「预测与规划联合建模」的代表作：二者被论证为同一输出空间的同一生成任务，而非两个串联模块。支持原句 L381-385（见 4.2 引用）。
2. 【推断】联合的动机是 ego-agent 未来交互闭环——T1 的他车预测依赖 T5 的自车未来动作、T5 的规划又受 T1 的他车反应影响，串行管线结构上无法表达。支持原句 L78-86（见第 3 节引用）。
3. 【推断】联合的机制载体是「结构化潜轨迹空间 + 轨迹先验」：预测与规划共享同一 VAE 潜空间，通过实例分布与真值分布的 KL 匹配统一训练。支持原句 L503-505 与 L523-527（见 4.4 引用）。
4. 【推断】轨迹结构先验（连续、直线/近常曲率）是 T1 与 T5 共享的领域先验，生成式框架把它显式编码为潜空间分布，是二者质量共同提升的机制性来源。支持原句 L393-399："The trajectories of both the ego vehicle and other agents are highly structured (e.g., continuous) and follow certain patterns. For example, most of the trajectories are straight lines as the vehicle driving at a constant speed, and some of them are curved lines with a near-constant curvature when the vehicle turning right or left. Only in very rare cases will the trajectories be zig-zagging."
5. 【推断】消融数值表明「联合」的收益集中在碰撞率与长时域误差：去掉 ego→agent 交互 Avg CR 由 0.43 恶化到 1.22（Table 3），去掉生成模块 Avg L2 由 0.91 恶化到 1.11（Table 4）——为叙事中「T1+T5 联合带来安全与一致性收益」提供了直接证据。支持原句 L661-665（见第 6 节引用）。

## 9. 图注/表注逐字抄录

- Figure 1（L55-62）："Figure 1. Comparisons of the proposed generative end-to-end autonomous driving framework with the conventional pipeline. Most existing methods follow a serial design of perception, prediction, and planning. They usually ignore the high-level interactions between the ego car and other agents and the structural prior of realistic trajectories. We model autonomous driving as a future generation problem and conduct motion prediction and ego planning simultaneously in a structural latent trajectory space."
- Figure 2（L236-241）："Figure 2. Framework of our generative end-to-end autonomous driving. Given surrounding images as inputs, we employ an image backbone to extract multi-scale features and then use a BEV encoder to obtain BEV tokens. We then use cross-attention and deformable cross-attention to transform BEV tokens into map and agent tokens, respectively. With an additional ego token, we use self-attention to enable ego-agent interactions and cross-attention to further incorporate map information to obtain the instance-centric scene representation. We map this representation to a structural latent trajectory space which is jointly learned using ground-truth future trajectories. Finally, we employ a future trajectory generator to produce future trajectories to simultaneously complete motion prediction and planning."
- Figure 3（L371-377）："Figure 3. Illustration of the proposed trajectory prior modeling and future generation. We use a future trajectory encoder to map ground-truth trajectories to a latent trajectory space, where we use the Gaussian distribution to model the trajectory uncertainty. We then employ a gate recurrent unit (GRU) to progressively predict the next future in the latent space and use a decoder to obtain explicit trajectories."
- Figure 4（L673-674）："Figure 4. Visualizations of the results of GenAD with comparisons with V AD [21]. We provide perception, motion prediction, and planning results with surrounding camera inputs."
- Table 1（L506-508，原文如此，含两个 † 说明）："Table 1. Comparisons with state-of-the-art methods in motion planning performance on the nuScenes [2] val dataset. † represents that the metrics are computed with an average of all the predicted frames. † denotes FPS measured with the same environment on our machine with a single RTX 3090 GPU."
- Table 2（L581-582）："Table 2. Comparisons of perception, prediction, and planning performance. † denotes FPS measured with the same environment on our machine with a single RTX 3090 GPU card."
- Table 3（L586-588）："Table 3. Effect of the instance-centric scene representation. E → A represents the proposed ego-to-agent interaction to obtain the instance-centric scene representation."
- Table 4（L628-630）："Table 4. Effect of the generative framework for autonomous driving. TPM and LFTG denote the trajectory prior modeling and latent future trajectory generation, respectively."
