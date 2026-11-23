# Drive-WM 精读笔记

> 唯一数据源：`auto_docs/papers/txt/DriveWM.txt`（arXiv 2311.17918，全文 1321 行）。所有事实性陈述附原文逐字引用（英文）+ 行号；论文未写处标「原文未写」；解读标【推断】。

## 1) 原文信息

- **标题**（行 1–2）："Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving"
- **作者**（行 3，∗ 为共同一作，B 为通讯标注）："Yuqi Wang1∗ Jiawei He1∗ Lue Fan1∗ Hongxin Li1∗ Yuntao Chen2B Zhaoxiang Zhang1,2B"
- **机构**（行 4）："1CASIA 2CAIR, HKISI, CAS"
- **项目页**（行 5）："Project Page: https://drive-wm.github.io"；**代码**（行 6）："Code: https://github.com/BraveGroup/Drive-WM"
- **arXiv ID 与版本**（行 58）："arXiv:2311.17918v1 [cs.CV] 29 Nov 2023"
- **发表状态**：txt 中仅有 arXiv v1 预印本标注（行 58）；会议/期刊录用信息**原文未写**。

## 2) Abstract 全文逐字（行 31–49）

> "In autonomous driving, predicting future events in advance and evaluating the foreseeable risks empowers autonomous vehicles to better plan their actions, enhancing safety and efficiency on the road. To this end, we propose Drive-WM, the first driving world model compatible with existing end-to-end planning models. Through a joint spatial-temporal modeling facilitated by view factorization, our model generates high-fidelity multiview videos in driving scenes. Building on its powerful generation ability, we showcase the potential of applying the world model for safe driving planning for the first time. Particularly, our Drive-WM enables driving into multiple futures based on ∗Equal contribution. distinct driving maneuvers, and determines the optimal trajectory according to the image-based rewards. Evalua-tion on real-world driving datasets verifies that our method could generate high-quality, consistent, and controllable multiview videos, opening up possibilities for real-world simulations and safe planning."

（注：行 43 的 "∗Equal contribution." 是夹在 Abstract 中间的脚注，属作者行共同一作说明，非摘要正文。）

## 3) 问题与动机（多视角视觉预测 + 规划联合的论证）

**出发点**：端到端自动驾驶受到关注，但其 OOD 泛化存疑。行 51–56：

> "These approaches take multi-sensor data as input and directly output planning results in a joint model, allowing for joint optimization of all modules. However, it is questionable whether an end-to-end planner trained purely on expert driving trajectories has sufficient generalization capabilities when faced with out-of-distribution (OOD) cases."

Figure 2 给出实证：自车横向偏移 0.5m 后 VAD 失效。行 60–63 + 行 116–119：

> "As illustrated in Figure 2, when the ego vehicle's position deviates laterally from the center line, the end-to-end planner struggles to generate a reasonable trajectory."

> "We shift the ego location 0.5m to the right to create an out-of-domain case. (a) shows the reasonable trajectory prediction of the V AD [30] method under normal data, and (b) shows the irrational trajectory when encountering out-of-distribution cases."

**核心论证（预测 → 反馈 → 规划闭环）**，行 63–72：

> "To alleviate this problem, we propose improving the safety of autonomous driving by developing a predictive model that can foresee planner degradation before decision-making. This model, known as a world model [19, 20, 35], is designed to predict future states based on current states and ego actions. By visually envisioning the future in advance and obtaining feedback from different futures before actual decision-making, it can provide more rational planning, enhancing generalization and safety in end-to-end autonomous driving."

**三个技术挑战**（行 77–92）：(1) 高分辨率像素空间建模——"The driving world model requires modeling in high-resolution pixel space. The previous low-resolution image [20] or vectorized state space [4] methods cannot effectively represent the numerous fine-grained or non-vectorizable events in the real world."；(2) 多视角一致视频生成困难——"Previous and concurrent works are limited to single view video [28, 31, 63] or multiview image generation [17, 53, 69], leaving multiview video generation an open problem"；(3) 异构条件（天气/光照/动作/布局）难以灵活适配——"It is challenging to flexibly accommodate various heterogeneous conditions like changing weather, lighting, ego actions, and road/obstacle/vehicle layouts."

**解决方案概述**（行 93–105）："we introduce multiview and temporal modeling for jointly generating multiple views and frames. To further enhance multiview consistency, we propose factorizing the joint modeling to predict intermediate views conditioned on adjacent views… We also introduce a simple yet effective unified condition interface enabling flexible use of heterogeneous conditions like images, text, 3D layouts, and actions… Finally, building on the multiview world model, we explore end-to-end planning applications to enhance autonomous driving safety."

**「首个」声明**（行 124–126）："We are the first to explore the potential application of the world model in end-to-end planning for autonomous driving. We experimentally show that our method could enhance the overall soundness of planning and robustness in out-of-distribution situations."

## 4) 方法逐机制

### 4.1 联合建模（Sec 3.1，行 197–210）

行 197–210 给出形式化：视频 `x ∈ R^{T×K×3×H×W}`，潜变量 `z ∈ R^{T·K×C×Ĥ×Ŵ}`，去噪目标：

> "A denoising model fθ,ϕ,ψ (parameterized by spatial parameters θ, temporal parameters ϕ and multiview parameters ψ) receives the diffused zτ as input and is optimized by minimizing the denoising score matching objective E z∼pdata,τ∼pτ,ϵ∼N(0,I)[∥y − fθ,ϕ,ψ(zτ ; c, τ)∥2 2], (1)"

**时间层**（行 211–223）："The temporal encoding layer is attached after the 2D spatial layer in each block, following established practice in VideoLDM [2]… we rearrange the latent to hold out the temporal dimension, denoted as (TK)CHW → KCTHW, to apply the 3D convolution in spatio-temporal dimensions THW. Then we arrange the latent to (KHW)TC and apply standard multi-head self-attention to the temporal dimension."

**多视角层**（行 224–233）："we rearrange the latent as (KHW)TC → (THW)KC to hold out the view dimension. Then a self-attention layer parameterized by ψ in Eq. 1 is employed across the view dimension. Such multiview attention allows all views to possess similar styles and consistent overall structure."

**训练策略**（行 234–240）："we first train a standard image diffusion model with single-view image data and conditions… Then we freeze the parameters θ and fine-tune the additional temporal layers (ϕ) and multiview layers (ψ) with video data."

### 4.2 视角分解（Sec 3.2，行 241–344）

行 250–253 分解联合分布：

> "p(x1,...,K) = p(x1)p(x2|x1) . . . p(xK|x1, . . . , xK−1). (2)"

因自回归低效（行 309–315："such an autoregressive generation is inefficient, making such full factorization infeasible in practice"），将视角分为 reference views `xr` 与 stitched views `xs`（行 316–323）："in nuScenes, reference views can be the {F, BL, BR}2, and stitched views can be {FL, B, FR}. We use the term 'stitched' because a stitched view appears to be 'stitched' from its two neighboring reference views."（脚注 2：F: front, B: back, L: left, R: right，行 331）

考虑时间连贯后（行 334–338）：

> "p(x) = p(xs, xr|xpre) = p(xr|xpre)p(xs|xr, xpre), (4)"

`p(xr|xpre)` 由 Sec 3.1 流程实现；`p(xs|xr, xpre)` 额外加入相邻 reference views 作条件（行 339–344）。

### 4.3 统一条件接口（Sec 3.3，行 345–425）

- **图像条件**（行 366–372）：初始上下文帧与 reference views 用 ConvNeXt [39] 编码："A given image condition I ∈ R3×H×W is encoded and flattened to a sequence of d-dimension embeddings i = (i1, i2, ..., in) ∈ Rn×d, using ConvNeXt as encoder [39]."
- **布局条件**（行 373–380）："we project the 3D boxes and HD maps into a 2D perspective view"，嵌入 `l ∈ R^{k×d}`。
- **文本条件**（行 381–385）："we combine view information, weather, and light to derive a text description"，用预训练 CLIP [42] 编码为 `e ∈ R^{m×d}`。
- **动作条件**（行 386–391）："we define the action in a time step as (∆x, ∆y), which represents the movement of ego location to the next time step. We use an MLP to map the action into a d-dimension embedding a ∈ R2×d."
- **统一接口**（行 392–425）："we take the concatenation of required embeddings as input for the denoising UNet"，式 (5)：`ct = [i0, l0, e0, at] ∈ R^{(n+k+m+2)×d}`；"ct interacts with the latent zt in 3D UNet by cross attention in a frame-wise manner."

### 4.4 树式 rollout 规划（Sec 4.1，行 427–452）

> "Blindly planning actions without anticipating consequences is dangerous. Leveraging our world model enables comprehensive evaluation of possible futures for safer planning."（行 427–429）

流程（行 433–438）：

> "At each time step, we leverage the world model to generate predicted future scenarios for trajectory candidates sampled from the planner, evaluate the futures using an image-based reward function, and select the optimal trajectory to extend the planning tree."

动作定义（行 444–448）："we define its action at at time t as (xt+1 − xt, yt+1 − yt) for each trajectory, where xt and yt are the ego locations at time t." 生成与决策循环（行 448–452）："Given the actions, we adopt the condition combination in Eq. 5 for the video generation. After the generation, we leverage an image-based reward function to choose the optimal trajectory as the decision. Such a generation-decision process can be repeated to form a tree-based rollout."

### 4.5 图像奖励函数（Sec 4.2，行 453–484）

行 457–463：在生成视频上跑感知模型得到奖励："we utilize image-based 3D object detector [37] and online HDMap predictor [38] to obtain the perception results on the generated videos."

**map reward**（行 462–466）："The map reward includes two factors, distance away from the curb, encouraging the ego vehicle to stay in the correct drivable area, and centerline consistency, preventing ego from frequently changing lanes and deviating from the lane in the lateral direction." **object reward**（行 466–469）："The object reward means the distance away from other road users in longitudinal and lateral directions. This reward avoids the collision between the ego vehicle and other road users." **合成**（行 469–471）："The total reward is defined as the product of the object reward and the map reward. We finally select the ego prediction with the maximum reward."

**非向量化奖励（GPT-4V）**（行 474–483）："Since the proposed world model operates in pixel space, it can further get rewards from the non-vectorized representation to handle more general cases… Leveraging the recent powerful foundational models such as GPT-4V, the planning process can get more comprehensive rewards from the non-vectorized representation."

## 5) 实验与主结果（数值逐字抄录，不四舍五入）

**设置**（Sec 5.1）：nuScenes "a total of 700 training videos and 150 validation videos. Each video includes around 20 seconds captured by six surround-view cameras"（行 489–491）；图像 "crop and resize the original image from 1600 × 900 to 384 × 192"，"initialized with Stable Diffusion checkpoints [44]"，"conducted on A40 (48GB) GPUs"（行 492–495）。质量指标 FID/FVD（行 497–499）；多视角一致性指标 KPM（LoFTR [51] 匹配，行 500–512）；可控性用 CVT [72] 分割、3D 检测 [62]、在线地图 [38]（行 513–518）；规划用 L2 与碰撞率（行 519–521）。

**表 1a 生成质量**（行 548–555）：

| Method | Multi-view | Video | FID↓ | FVD↓ |
|---|---|---|---|---|
| BEVGen [53] | ✓ | | 25.54 | - |
| BEVControl [69] | ✓ | | 24.85 | - |
| MagicDrive [17] | ✓ | | 16.20 | - |
| Ours | ✓ | | 12.99 | - |
| DriveGAN [31] | | ✓ | 73.4 | 502.3 |
| DriveDreamer [63] | | ✓ | 52.6 | 452.0 |
| Ours | ✓ | ✓ | 15.8 | 122.7 |

【推断】txt 中每行 ✓ 的列位被压平；结合正文 "we make separate comparisons with previous methods in multi-view images and single-view videos, respectively"（行 531–533），上表列位按此恢复。正文陈述（行 536–541）："In single-view image generation, we achieve 12.99 FID, achieving a significant improvement over previous methods. For video generation, our method exhibits a significant quality improvement compared to past single-view video generation methods, achieving 15.8 FID and 122.7 FVD."

**表 1b 可控性**（行 557–564）：

| Method | mAPobj ↑ | mAPmap ↑ | mIoUfg ↑ | mIoUbg ↑ |
|---|---|---|---|---|
| GT | 37.78 | 59.30 | 36.08 | 72.36 |
| BEVGen [53] | - | - | 5.89 | 50.20 |
| LayoutDiffusion [71] | 3.68 | - | 15.51 | 35.31 |
| GLIGEN [36] | 15.42 | - | 22.02 | 38.12 |
| BEVControl [69] | 19.64 | - | 26.80 | 60.80 |
| MagicDrive [17] | 12.30 | - | 27.01 | 61.05 |
| Ours | 20.66 | 37.68 | 27.19 | 65.07 |

**表 3 规划主结果**（行 621–624）：

| Method | L2 (m) ↓ 1s | 2s | 3s | Avg. | Collision (%) ↓ 1s | 2s | 3s | Avg. |
|---|---|---|---|---|---|---|---|---|
| VAD (GT cmd) | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.17 | 0.41 | 0.22 |
| VAD (rand cmd) | 0.51 | 0.97 | 1.57 | 1.02 | 0.34 | 0.74 | 1.72 | 0.93 |
| Ours | 0.43 | 0.77 | 1.20 | 0.80 | 0.10 | 0.21 | 0.48 | 0.26 |

正文（行 643–649）："we sample planned trajectories from V AD according to the three commands 'Go straight', 'Turn left', and 'Turn right'… our tree-based planner outperforms random driving commands and even achieves performance close to the GT command."

**表 5 OOD 规划**（行 653–656）：

| OOD | World Model f.t. | L2 1s | 2s | 3s | Avg. | Collision 1s | 2s | 3s | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| | | 0.41 | 0.70 | 1.05 | 0.72 | 0.07 | 0.17 | 0.41 | 0.22 |
| ✓ | | 0.73 | 0.99 | 1.33 | 1.02 | 1.25 | 1.62 | 1.91 | 1.59 |
| ✓ | ✓ | 0.50 | 0.79 | 1.17 | 0.82 | 0.72 | 0.84 | 1.16 | 0.91 |

正文（行 659–669）："we shift the ego location laterally by 0.5 meters… the performance of the existing end-to-end planner V AD [30] undergoes a significant decrease… we fine-tune the planner with generated video supervised by the trajectory that the ego-vehicle drives back to the lane. Learning from these OOD data, the performance of the planner can be better and near normal levels."

**训练/推理细节（附录 B，供参考）**：图像条件模型 "trained for 60,000 iterations with a total batch size of 768. We use the AdamW optimizer with a learning rate 1 × 10−4"（行 1119–1121）；视频模型 "fine-tune this model for 40,000 iterations with a batch size of 32, with video frame length T = 8"，lr "5 × 10−5"（行 1123–1128）；推理 "the number of sampling steps is 50, and we use stochasticity η=1.0, CFG=5.0"（行 1135–1136）。数据策展（附录 C.2，行 1232–1259）：原始动作分布不均，"This process results in 1048 unique clips"；"we sample N = 36 clips from each bin of the 2D 32 × 11 grid… Consequently, 7272 clips are collected."

## 6) 消融（全部抄录）

**表 2a 统一条件消融**（行 568–571）：

| Temp emb. | Layout Cond. | FID↓ | FVD↓ | KPM(%)↑ |
|---|---|---|---|---|
| ✓ | | 20.3 | 212.5 | 31.5 |
| | ✓ | 18.9 | 153.8 | 44.6 |
| ✓ | ✓ | 15.8 | 122.7 | 45.8 |

正文（行 597–601）："we find that the layout condition has a significant impact on the model's ability, improving both the quality and consistency of the generated videos. Additionally, temporal embedding can enhance the quality of the generated videos."

**表 2b 多视角时序微调消融**（行 573–576）：

| Temp Layers | View Layers | FID↓ | FVD↓ | KPM(%)↑ |
|---|---|---|---|---|
| | | 23.3 | 228.5 | 40.8 |
| ✓ | | 16.2 | 127.1 | 40.9 |
| ✓ | ✓ | 15.8 | 122.7 | 45.8 |

**表 2c 分解生成消融**（行 578–580）：

| Method | KPM(%)↑ | FVD↓ | FID↓ |
|---|---|---|---|
| Joint Modeling | 45.8 | 122.7 | 15.8 |
| Factorized Generation | 94.4 | 116.6 | 16.4 |

正文（行 607–612）："factorized generation notably improves the consistency among multiple views, increasing from 45.8% to 94.4%, in contrast to joint modeling. This enhancement is achieved while ensuring the quality of both images and videos."

**表 4 奖励函数消融**（行 628–633）：

| Map Reward | Object Reward | L2 1s | 2s | 3s | Avg. | Collision 1s | 2s | 3s | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| | | 0.51 | 0.97 | 1.57 | 1.02 | 0.34 | 0.74 | 1.72 | 0.93 |
| ✓ | | 0.45 | 0.82 | 1.29 | 0.85 | 0.12 | 0.33 | 0.72 | 0.39 |
| | ✓ | 0.43 | 0.77 | 1.20 | 0.80 | 0.12 | 0.21 | 0.48 | 0.27 |
| ✓ | ✓ | 0.43 | 0.77 | 1.20 | 0.80 | 0.10 | 0.21 | 0.48 | 0.26 |

正文（行 649–652）："the combined reward outperforms each sub-reward, particularly in terms of the object collision metric."

## 7) 局限 / 未来工作（原文逐字）

**原文未写**独立的 Limitations 章节。最接近的表述为：

- 结论（行 678–689）："We introduce Drive-WM, the first multiview world model for autonomous driving. Our method exhibits the capability to generate high-quality and consistent multiview videos under diverse conditions… Besides, extensive experiments on nuScenes dataset show that our method could enhance the overall soundness of planning and robustness in out-of-distribution situations."
- 条件可扩展（行 350–351）："More conditions can be further included for better controllability."
- 数据增广潜力（行 1018–1023）："This ability has great potential for future data augmentation. By generating diverse scenes under various weather and lighting conditions, our model can significantly expand the training dataset, thereby improving the generalization performance and robustness of the model."
- GPT-4V 奖励（行 1080–1085）："By using GPT-4V to critique Drive-WM's forecasts, more robust feedback can be achieved to eventually improve autonomous driving safety under diverse real-world conditions."
- 反事实事件（行 670–677）："our Drive-WM can generate counterfactual events, such as turning around and running over non-drivable areas (Figure 7), which are significantly different from the training data. The ability to generate such counterfactual data reveals again that our Drive-WM has the potential to foresee and handle out-of-domain cases."

## 8) 叙事关联【推断，≤5 条】（T1=预测/世界模型，T5=规划/决策闭环；T1/T5 的确切定义以上级叙事框架为准，论文原文无此概念）

1. **「预测→评估→决策」闭环的代表作**：世界模型先想象未来，再以图像奖励选择轨迹，把预测与规划绑成单一循环。支持原句（行 63–72）："we propose improving the safety of autonomous driving by developing a predictive model that can foresee planner degradation before decision-making… By visually envisioning the future in advance and obtaining feedback from different futures before actual decision-making, it can provide more rational planning, enhancing generalization and safety in end-to-end autonomous driving."【推断】这是 T1（预测）与 T5（规划）串成闭环的核心机制证据。
2. **树式 rollout 是预测通往规划的接口**：生成候选未来 → 奖励评估 → 扩展规划树。支持原句（行 433–438）："we leverage the world model to generate predicted future scenarios for trajectory candidates sampled from the planner, evaluate the futures using an image-based reward function, and select the optimal trajectory to extend the planning tree."【推断】与 Dreamer 式 latent rollout 同构，但本文 roll out 在像素空间。
3. **像素空间世界模型使「非向量化风险」进入规划奖励**：这是 Drive-WM 相对向量空间世界模型的差异化卖点。支持原句（行 474–476）："Since the proposed world model operates in pixel space, it can further get rewards from the non-vectorized representation to handle more general cases."【推断】对应 T1 中「表征载体」之争（pixel vs vector）。
4. **用世界模型生成 OOD 数据反哺规划器（预测→数据→规划升级）**：支持原句（行 665–669）："we fine-tune the planner with generated video supervised by the trajectory that the ego-vehicle drives back to the lane. Learning from these OOD data, the performance of the planner can be better and near normal levels."【推断】这是「预测能力直接改善规划模型」的另一条闭环路径（仿真数据闭环）。
5. **反事实生成能力是预测模型预见 OOD 的证据**：支持原句（行 670–677）："Given an initial observation and the action, our Drive-WM can generate counterfactual events, such as turning around and running over non-drivable areas (Figure 7)… The ability to generate such counterfactual data reveals again that our Drive-WM has the potential to foresee and handle out-of-domain cases."【推断】训练集不含掉头样本（行 641–642："Note that our training set does not contain any turning-around samples"）却能生成，说明模型具备超出训练分布的想象能力——这是评估世界模型「预测泛化」的关键定性证据。

## 9) 图注 / 表注逐字抄录（主要部分）

- **Figure 1**（行 27–29）："Multiview visual forecasting and planning by world model. At time step T , the world model imagines the multiple futures at T + K, and finds it is safe to keep going straight at T . Then the model realizes that the ego car will be too close to the front car according to the imagination of time step T + 2K, so it decides to change to the left lane for a safe overtaking."
- **Figure 2**（行 114–119）："Ego vehicle's slight deviation from centerline causes motion planner to struggle generating reasonable trajectories. We shift the ego location 0.5m to the right to create an out-of-domain case. (a) shows the reasonable trajectory prediction of the V AD [30] method under normal data, and (b) shows the irrational trajectory when encountering out-of-distribution cases."
- **Figure 3**（行 306–308）："Overview of the proposed framework. (a) illustrates the training and inference pipeline of the proposed method. (b) visualizes the unified conditions leveraged to control the generation of multi-view video.(c) represents the probabilistic graph of factorized multiview generation. It takes the 3-view output from (a) as input to generate other views, enhancing the multi-view consistency."
- **Figure 4**（行 357–358）："Illustration of factorized multi-view generation. We take the sensor layout in nuScenes as an example."
- **Figure 5**（行 412–415）："End-to-end planning pipeline with our world model. We display the components of our planning pipeline at the top and illustrate the decision-making process in the planning tree using image-based rewards at the bottom."
- **Table 1**（行 566–567）："Multi-view video generation performance on nuScenes. For each task, we test the corresponding models trained on the nuScenes training set. Our Drive-WM surpasses all other methods in both quality and controllability evaluation."
- **Table 2**（行 582–583）："Ablations of the components in model design. The experiments are conducted under the layout-based video generation (See model variants in Sec. 5.1) from nuScenes validation set."
- **Table 3**（行 625–627）："Planning performance on nuScenes. Instead of using the ground truth driving command, we use our tree-based planning to select the best out of three commands."
- **Table 4**（行 634–635）："Image-based reward function design. We use two sub-rewards, map reward and object reward."
- **Table 5**（行 657–658）："Out-of-domain planning. We define OOD location with a lateral deviation of 0.5 meters from the ego vehicle."
- **Figure 6**（行 639–640）："Qualitative results of factorized multiview generation. For each compared pair, the upper row is generated without factorization, and the lower row is generated with factorization."
- **Figure 7**（行 641–642）："Counterfactual events generation. Top: turning around at a T-shape intersection on a rainy day. Note that our training set does not contain any turning-around samples. Bottom: running over a non-drivable area."
- **Figure 8**（行 1058–1061）："Generation of multiple plausible futures based on the planning. Here we only show the front-view videos for better illustration. The video samples are generated based on the first frames from the nuScenes val set. In the first row, we show examples of lane changing to the left and straight ahead. In the second row, we show cases of driving towards the roadside and driving straight ahead. In the last row, we present examples of making left and right turns at the intersection."
- **Figure 17**（行 1292–1293）："GPT-4V as a reward function. GPT-4V could give more reasonable rewards in cases such as there is a puddle ahead on the road."
- **Figure 22**（行 1318–1320）："Illustration of keypoint matching in KPM calculation. The blue points are the image keypoints in the overlapping regions on the left/right side of the image. The green lines are the matched keypoints between the current view and its two adjacent views using the LoFTR [51] matching algorithm."

---

*本笔记全部内容依据 `auto_docs/papers/txt/DriveWM.txt`（1321 行）逐字核对；行号即该 txt 文件行号。*
