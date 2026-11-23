# Genie: Generative Interactive Environments — 精读笔记

> 唯一数据源：`auto_docs/papers/txt/Genie.txt`（1444 行，27 页）。所有英文引文逐字复制自该 txt 并标注行号；txt 由 PDF 提取，个别词间空格缺失（如 "firstgenerative"）保持原样。论文未写之处标「原文未写」；本人解读标【推断】。

## 1. 原文信息

- 标题（L3）："Genie: Generative Interactive Environments"
- 作者（L4–8）：Jake Bruce*、Michael Dennis*、Ashley Edwards*、Jack Parker-Holder*、Yuge (Jimmy) Shi*、Edward Hughes、Matthew Lai、Aditi Mavalankar、Richie Steigerwald、Chris Apps、Yusuf Aytar、Sarah Bechtle、Feryal Behbahani、Stephanie Chan、Nicolas Heess、Lucy Gonzalez、Simon Osindero、Sherjil Ozair、Scott Reed、Jingwei Zhang、Konrad Zolna、Jeff Clune、Nando de Freitas、Satinder Singh、Tim Rocktäschel*。
- 机构（L9）："*Equal contributions,1Google DeepMind,2University of British Columbia"
- 通讯作者（L34）："Corresponding author(s): Ashley Edwards (edwardsashley@google.com), Jack Parker-Holder (jparkerholder@google.com)."
- 版权（L35）："© 2024 Google DeepMind. All rights reserved"
- ID/状态（L36）："arXiv:2402.15391v1  [cs.LG]  23 Feb 2024"。会议录用信息原文未写。各页页眉含 "Proprietary + Confidential"（L2、L37），其含义原文未说明。

## 2. Abstract 全文逐字（L24–32）

> L24–25："We introduce Genie, the firstgenerative interactive environmenttrained in an unsupervised manner from unlabelled Internet videos. The model can be prompted to generate an endless variety of action-controllable virtual worlds described through text, synthetic images, photographs, and even sketches."
> L26–28："At 11B parameters, Genie can be considered afoundation world model. It is comprised of a spatiotemporal video tokenizer, an autoregressive dynamics model, and a simple and scalable latent action model."
> L28–30："Genie enables users to act in the generated environments on a frame-by-frame basisdespite training without any ground-truth action labelsor other domain-specific requirements typically found in the world model literature."
> L30–32："Further the resulting learned latent action space facilitates training agents to imitate behaviors from unseen videos, opening the path for training generalist agents of the future."

关键词（L33）："Keywords: Generative AI, Foundation Models, World Models, Video Models, Open-Endedness"

## 3. 问题与动机

**为什么「可交互环境」生成值得做**：视频生成与语言工具之间在交互性上存在鸿沟。L56–59："Still, there remains a gulf between the level of interactions and engagement of video generative models and language tools such as ChatGPT, let alone more immersive experiences." 由此提出核心问题。L60–63："What if, given a large corpus of videos from the Internet, we could not only train models capable of generating novel images or videos, but entire interactive experiences? We proposegen- erative interactive environments, a new paradigm for generative AI whereby interactive environments can be generated from a single text or image prompt."

Genie 的定位（L67–72）："Our approach, Genie, is trained from a large dataset of over 200,000 hours of publicly available Internet gaming videos and, de- spite trainingwithout action or text annotations, is controllable on a frame-by-frame basis via a learned latent action space"；L73–77："At 11B parameters, Genie exhibits properties typically seen in foun- dation models—it can take an unseen image as a prompt making it possible to create and play entirely imagined virtual worlds (e.g Figure 2)."

**潜动作接口的论证**：要可控生成需以动作为条件，但互联网视频没有动作标注。L182–189："To achieve con- trollable video generation, we condition each fu- ture frame prediction on the action taken at the previous frame. However, such action labels are rarelyavailableinvideosfromtheInternetandac- tion annotation can be costly to obtain. Instead, we learn latent actions in a fully unsupervised manner (see Figure 5)." 码本被限制得很小以保可玩性（L199–203）："We limit the vocabulary size|A| of the VQ codebook, i.e. the maximum number of possible latent actions, to a small value to permit human playability and further enforce controllability (we use|A| = 8 in our experiments)." 信息论依据（L203–207）："As the decoder only has access to the history and latent action,˜a_t should encode the most meaningful changes between the past and the future for the decoder to successfully reconstruct the future frame." 脚注 2 解释潜动作如「新手柄按键」（L342–348）："However, wefoundthatthemeaningofeachaction remained consistent across different inputs. Hence, interpreting the mapping of latent actions is akin to learning the buttons on a new controller."

## 4. 方法逐机制

**ST-transformer 骨架**（L146–167）："Unlike a traditional transformer where every token attends to all others, an ST-transformer contains L spatiotemporal blocks with interleaved spatial and temporal attention layers, followed by a feed-forward layer (FFW) as standard at- tention blocks." L151–157："The self-attention in the spatial layer attends over the1×H×W tokens within each time step, and in the temporal layer attends over T×1×1 tokens across theT time steps. Similar to sequence transformers, the temporal layer assumes a causal structure with a causal mask." 复杂度论证（L157–162）："the dominating factor of compu- tation complexity (i.e. the spatial attention layer) in our architecture scales linearly with the num- ber of frames rather than quadratically"；L163–167："we include only one FFW after both spatial and temporal compo- nents, omitting the post-spatial FFW to allow for scaling up other components of the model, which we observe to improve results significantly."

**三组件与两阶段训练**（L168–181）："our model contains three key components: 1) alatent action modelthat infers the latent actiona between each pair of frames and 2) avideo tokenizerthat converts raw video frames into discrete tokensz and 3) a dynamics modelthat, given a latent action and past frame tokens, predicts the next frame of the video."；"we train the video tokenizer first, which is used for the dynamics model. We then co-train the latent action model (directly from pixels) and the dynamics model (on video tokens)."

**潜动作模型 LAM**（L190–200）："First, an encoder takes as inputs all previous frames x1:t =(x1,···xt) as well as the next frame xt+1, and outputs a corresponding set of continuous latent actions˜a1:t =(˜a1,···˜at). A decoder then takes all previous frames and latent actions as input and predicts the next frameˆxt+1. To train the model, we leverage a VQ-VAE-based objective (van den Oord et al., 2017), which enables us to limit the number of predicted actions to a small discrete set of codes." 关键细节（L227–230）："Note that this de- coder exists only to give the LAM training signal. In fact, apart from the VQ codebook, the entire LAM is discarded at inference time and replaced with actions from the user."；L231–235："The causal mask in the temporal layer allows us to take the entire video x1:T as input and generate all latent actions be- tween each frame˜a1:T−1."

**视频 tokenizer**（L236–247）："we compress videos into discrete tokens to re- duce dimensionality and enable higher quality video generation (see Figure 6). We again make use of VQ-VAE, which takes inT frames of video x1:T =(x1,x2,···,xT)∈R^{T×H×W×C} as input, gen- erating discrete representations for each frame z1:T =(z1,z2,···,zT)∈I^{T×D}, whereD is the size of the discrete latent space."（txt L246 作 "VQ-VQAE"，系提取笔误，照录）L250–259："we utilize the ST-transformer in both the encoder and decoder to incorporate temporal dynamics in the encodings, which improves the video gen- eration quality. By the causal nature of the ST- transformer, each discrete encodingz_t contains informationfromallpreviouslyseenframesofthe video x1:t." 与 C-ViViT 对比（L259–266）："our ST-transformer based tokenizer (ST- ViViT) is much more compute efficient with the dominating factor in its cost increasing linearly with the number of frames."（C-ViViT "the cost grows quadratically with the number of frames"）

**动力学模型**（L274–288）："The dynamics model is a decoder-only MaskGIT (Chang et al., 2022) trans- former (Figure 7). At each time stept∈[1,T], it takes in the tokenized videoz1:t−1 and stopgrad latent actions ˜a1:t−1 and predicts the next frame tokens ˆz_t."；"The model is trained with a cross-entropy loss between the predicted tokens ˆz2:T and ground-truth tokensz2:T. At train time we randomly mask the input tokensz2:T−1 ac- cording to a Bernoulli distribution masking rate sampled uniformly between0.5 and 1."

**掩码策略+加性嵌入**（L283–296）：同上掩码率 0.5–1；L289–296："a common practice for training world-models ... is to concate- nate the action at timet to the corresponding frame (Micheli et al., 2023; Robine et al., 2023). However, we found that treating the latent ac- tions asadditive embeddingsfor both the latent action and dynamics models helped to improve the controllability of the generations."

**推理**（L299–328）："A player first prompts the model with an imagex1 that serves as the initial frame... The player then specifies a discrete latent actiona1 to take by choosing any integer value within[0,|A|). The dynamics model takes the frame tokensz1 and corresponding latent action ˜a1, which is obtained by indexing into the VQ codebook with the discrete inputa1, to predict the next frame tokensz2."；L319–328："This process is repeated to generate the rest of the sequenceˆz2:T in an au- toregressive manner as actions continue to be passed to the model... Note that we can regenerate ground truth videos from the dataset by passing the model the starting frame and inferred actions from the video, or generate completely new videos (or tra- jectories) by changing the actions."

## 5. 实验与主结果（数值逐字抄录，不四舍五入）

**数据集**：Platformers（L330–338）："yielding 55M 16s video clips at 10FPS, with 160x90 resolution. The final dataset contains 6.8M 16s video clips (30k hours)"。附录 B.1（L1211–1213）："Our resulting dataset contains 55M videos, which totals around 244k hours."；L1233："Our final dataset is 6.8M videos for a total of over 30k hours." 主模型（L342–343）："Unless otherwise spec- ified, results are with a 11B-parameter model trained on this dataset." Robotics（L398–407）："combining their dataset of∼130k robot demonstrations with a separate dataset of simulation data and the 209k episodes of real robot data from prior work (Kalashnikov et al., 2018). Note that we do not use actions from any of these datasets, and simply treat them as videos."

**指标**（L408–434）：FVD 用于视频保真；可控性用 Δ_tPSNR："ΔtPSNR = PSNR(xt,ˆxt)−PSNR(xt,ˆx′t)"；"For all experiments we reportΔtPSNR witht = 4."

**训练细节**（L435–453）：tokenizer 200M 参数、patch size 4、码本 embedding size 32、1024 码；LAM 300M 参数、patch size 16、码本 8 码；序列 16 帧 @10FPS；bfloat16 + QK norm；推理 "we perform 25 MaskGIT steps for the sampling of each frame with a temperature of 2 using random sampling."

**缩放实验**（L460–506）：模型规模 40M→2.7B："we train a series of dynamics models ranging from 40M to 2.7B parameters"；"with each increase in size corresponding to a consis- tent decrease in the final training loss"；batch 缩放："considering a 2.3B model with batch sizes of 128, 256, and 448, equating to 1.9M, 3.8M and 6.6M tokens"；最终模型："we train a 10.1B dynamics model with a batch size of 512, for a total of 125k steps, using 256 TPUv5p. When combined with the tokenizer and action model this brings the total to 10.7B parameters, trained on 942B tokens, which we refer to as the Genie model." 注：摘要称 11B（L26–27、L342、L511），与 10.7B 的差异原文未解释；【推断】或含网站版 360p 解码器（L506–508 "we train a larger decoder mapping tokens to 360p videos, adding additional parameters"）。

| 表（出处） | 内容（逐字数值） |
|---|---|
| Table 1（L127–130） | World Models：Video + Actions / Frame-level；Video Models：Video + Text / Video-level；Genie：Video / Frame-level |
| Table 10（L1312–1319）模型缩放 | 41M(18/8/512/64, 64 TPUv2, 3 days, 2.05×10^20 FLOPs)；96M(16/16/768/64, 64 TPUv2, 6 days, 3.58×10^20)；192M(20/18/1024/64, 64 TPUv2, 9 days, 6.4×10^20)；404M(21/12/1536/128, 64 TPUv2, 18 days, 1.2×10^21)；811M(20/20/2048/128, 128 TPUv3, 7 days, 2.2×10^21)；1.6B(28/22/2560/128, 128 TPUv3, 12 days, 4.04×10^21)；2.7B(36/22/3072/128, 256 TPUv3, 16 days, 6.91×10^21)（列序：Parameters/num_layers/num_heads/d_model/kq size/training hardware/training time/FLOPs；均 200k steps、batch 256、750B tokens） |
| Table 12（L1345–1346）Genie 动力学模型 | 10.1B；num_layers 48、num_heads 36、d_model 5120、k/q size 128；FLOPs 6.6×10^22 |
| Table 6（L1263–1265）tokenizer batch 缩放 | batch 64：64 TPUv2、4.22×10^20 FLOPs、PSNR 35.7；batch 384：64 TPUv3、2.57×10^21 FLOPs、PSNR 36.5 |

**平台游戏生成（定性）**（L516–524）："we qualitatively eval- uate our Platformers-trained model usingonly out-of-distribution (OOD) image prompts, including those generated from text-to-image models, hand-drawn sketches, and even realistic photos. The ability to generalize to such significantly OOD inputs underscores the robustness of our approach and the value of training on large-scale data, which would not have been feasible with real actions as input." 涌现能力：视差（L549–556）、可变形物体物理（Figure 11）。

**机器人模型**（L557–580）："We trained a 2.5B- parameter model on the Robotics dataset using the same hyperparameters found to be best on Platformers, achieving an FVD of 82.7 on the test split."；"our model learns not only the controls of the robotic arm but also the interactions and deformations of various objects (Figure 11). We believe this shows our approach presents a path to using larger video datasets from the Internet to create a founda- tional world model for robotics, with low-level controllable simulation that could be used for a variety of applications."

**训练智能体（BC）**（L592–623）："We use a frozen LAM to label a sequence of expert videos from a target environment with discrete latent actions and then train a policy that predicts the likelihood of the expert taking a latent action given an observation. We then use a small dataset with expert ground-truth actions for mapping latent to real actions"；"The LAM-based policy achieves the same score as the oracle given as few as 200 ex- pert samples to adapt, despite almost certainly never seeing CoinRun before."（CoinRun 分 hard/easy 两档；对比上界 oracle BC、下界 random agent；具体百分比数值原文只在 Figure 15 图中，未以文字给出。）

## 6. 消融（数值全部抄录）

**LAM 输入消融**（L624–652）：文字结论 L637–645："While this model achieved a slightly lower FVD score on the Platformers dataset, it did not main- tain this advantage on the Robotics dataset. More importantly, in both environments, the token- input model exhibited worse controllability (as measured by ΔtPSNR). This suggests that some information about video dynamics and movement might have been lost during tokenization, and as a result it is beneficial for the latent action model to take in raw videos as input."

| Table 2（L649–652） | #Params | FVD(↓) | ΔtPSNR(↑) |
|---|---|---|---|
| Token-input, Platformers | 2.3B | 38.8 | 1.33 |
| Pixel-input (Genie), Platformers | 2.5B | 40.1 | 1.91 |
| Token-input, Robotics | 1B | 257.8 | 1.65 |
| Pixel-input (Genie), Robotics | 1B | 136.4 | 2.07 |

**Tokenizer 架构消融**（L653–688）：文字 L674–688："Our proposed ST-ViViT architecture provides both improved video generation (FVD) and ΔtPSNR, for a reasonable trade-off in memory, as compared to to C-ViViT and the spatial-only ViT."；"In fact, C-ViViT exhibits a tendency towards overfit- ting, necessitating strong regularization during training, which might explain its considerably lower performance."（三者 patch size 10、batch 128、序列长 16，参数量相近）

| Table 3（L667–669） | #Params | Memory | FVD(↓) | ΔtPSNR(↑) |
|---|---|---|---|---|
| ViT | 230M | 0.3GB | 114.5 | 1.39 |
| C-ViViT (Villegas et al., 2023) | 225M | 1.6GB | 272.7 | 1.37 |
| ST-ViViT (ours) | 205M | 0.9GB | 81.4 | 1.66 |

**数据策展消融**（附录 B.1，L1230–1237）："having high quality data outweighs the quantity of data – even though the curated datasaet is only just over 10% the size of the original dataset, the model trained on the curated dataset outperforms in terms of FVD"。

| Table 4（L1235–1237） | #Params | FVD(↓) |
|---|---|---|
| Original dataset (55M videos) | 580M | 61.4 |
| Curated dataset (6.8M videos) | 580M | 54.8 |

## 7. 局限 / 未来工作（原文逐字，L786–806）

> L786–796："Thereareclearimprovementsthatcanbemade to the model. Genie inherits some of the weak- nesses of other autoregressive transformer mod- els, and can hallucinate unrealistic futures. And while we have made progress with spatiotem- poral representations, we are still limited to 16 frames of memory which makes it challenging to get consistent environments over long hori- zons. Finally, Genie currently operates around 1FPS and requires future advances to achieve an efficient frame rate for interaction."
> L797–806："Still, we believe Genie opens up vast poten- tial for future research. Given its generality, the model could be trained from an even larger pro- portion of Internet videos to simulate diverse, re- alistic, and imagined environments. Furthermore, we only briefly touched upon the capabilities of using Genie for training agents, but given that the lack of rich and diverse environments is one of the key limitations in RL, we could unlock new paths to creating more generally capable agents."

补充（Broader Impact，L820–827）："We have chosen not to release the trained model checkpoints, the model's training dataset, or examples from that data to accompany this paper or the website."；复现（L828–838）："we describe a smaller scale, fullyreproducibleexampleinAppendixFthatcan run on a single mid-range TPU (or GPU)."

## 8. 叙事关联【推断，≤5 条，各附支持原句】

1. **「从视频学潜动作」是「动作接口再设计」潮流的源头之一**：Genie 把「动作」从人工定义的接口变成数据中无监督涌现的离散码本，并用它作为人/智能体与世界模型的交互协议。支持："This is made possible via a latent action interface, learned fully unsupervised from Internet videos."（L22–23）；"interpreting the mapping of latent actions is akin to learning the buttons on a new controller."（L346–348）；"we seek a more agentic goal and explicitly learn alatent action spacefrom data, allowing users or agents to "play" the model using latent action-conditioned predictions."（L730–733）【推断】后世「token-as-action」「动作接口重设计」等思路可在此找到直接原型：接口不再由人类预先设计，而由数据+重建目标确定。
2. **对机器人领域的迁移**：作者专门用 RT1 数据做免动作实验，把「潜动作世界模型」定位为机器人的基础世界模型路径。支持："we also train a separate model on action-free robot videos from the RT1 dataset (Brohan et al., 2023), learning a generative environment with consistent latent actions."（L107–110）；"this approach presents a path to using larger video datasets from the Internet to create a foundational world model for robotics, with low-level controllable simulation that could be used for a variety of applications."（L576–580）
3. **潜动作作为策略学习的中介表征**：LAM 输出的潜动作可直接给未见环境的专家视频打标签、训 BC 策略，再以小样本完成 latent→real 映射。支持："The LAM-based policy achieves the same score as the oracle given as few as 200 expert samples to adapt... This provides evidence that the learnt latent actions are consistent and meaningful for transfer, as the mapping from latent to real contains no information about the current observation."（L616–623）
4. **免动作标注 ⇒ Internet 视频可规模化的论证**：这是该路线成立的经济学理由。支持："despite training without action or text annotations"（L69–70）；"which would not have been feasible with real actions as input."（L522–524）；"avoiding the need for ground-truth actions that are costly and may not generalize."（L775–777）
5. **与 VPT 的路线分野**：VPT 用「人工动作标注的 IDM」给视频打动作标签，Genie 反其道：让潜动作自己涌现。支持："VPT (Baker et al., 2022) is a recent ap- proach that uses an inverse dynamics model learnt from human-provided action labeled data, tolabelInternet-scalevideoswithactionsthatcan then be used for training a policy. We showed, in contrast, that we can uselatent actions learnt from Internet videos to infer policies for arbitrary environments"（L765–777）

## 9. 图注 / 表注逐字抄录（主要部分）

- Figure 1（L20–23）："A whole new world: Genie is capable of converting a variety of different prompts into interactive, playable environments that can be easily created, stepped into, and explored. This is made possible via a latent action interface, learned fully unsupervised from Internet videos. On the right we see a few generated steps for taking two latent actions. See more examples on our website."
- Figure 2（L86–93）："Diverse trajectories: Genie is a gen- erative model that can be used as an interactive environment. The model can be prompted in var- ious ways, either with a generated image (top) or a hand-drawn sketch (bottom). At each time step, the model takes a user-provided latent action to generate the next frame, producing trajectories with interesting and diverse character actions."
- Figure 3（L213–216）："Genie model training: Genie takes inT frames of video as input, tokenizes them into discrete tokens z via the video tokenizer, and infers the latent actions˜a between each frame with the latent action model. Both are then passed to the dynamics model to generate predictions for the next frames in an iterative manner."
- Figure 4（L217–224）："ST-transformer architecture. The ar- chitectureiscomposedof Lspatiotemporalblocks, each containing a spatial layer, temporal layer and feed-forward layer. Each color represents a single self-attention map, with the spatial layer attending over theH×W tokens from within a single time step, and temporal the same token from across theT time steps."
- Figure 5（L225–226）："Latent action model: learns actionsa_t unsupervised from unlabelled video frames."
- Figure 6（L248–249）："Video tokenizer: a VQ-VAE with ST- transformer."
- Figure 7（L271–273）："Dynamics model: takes in video to- kens and action embeddings, and predicts future masked video tokens."
- Figure 8（L309–314）："Genie Inference: the prompt frame is tokenized, combined with the latent action taken by the user, and passed to the dynamics model for iterative generation. The predicted frame tokens are then decoded back to image space via the tokenizer's decoder."
- Figure 9（L394–396）："Scaling results. Left: Training curves for different model sizes,Middle: Final training loss for each model size, averaged over the last 300 updates,Right: Final training loss for a 2.3B model with different batch sizes."
- Figure 10（L481–484）："Playing from Image Prompts: We can prompt Genie with images generated by text-to- image models, hand-drawn sketches or real-world photos. In each case we show the prompt frame and a second frame after taking one of the latent actions four consecutive times. In each case we see clear character movement, despite some of the images being visually distinct from the dataset."
- Figure 11（L537–539）："Learning to simulate deformable objects: we show frames from a ten step trajectory in the model, taking the same action. Genie is capable of learning the physical properties of objects such as bags of chips."
- Figure 12（L544–548）："Emulating parallax, a common fea- ture in platformer games. From this initial text- generated image, theforeground moves more than thenear and far middle ground, while the background moves only slightly."
- Figure 13（L568–575）："Controllable, consistent latent ac- tions in Robotics: trajectories beginning from three different starting frames from our Robotics dataset. Each column shows the resulting frame from taking the same latent action five times. De- spite training without action labels, the same ac- tions are consistent across varied prompt frames and have semantic meaning:down, up and left."
- Figure 14（L607–609）："Playing from RL environments: Ge- nie can generate diverse trajectories given an im- age of an unseen RL environment."
- Figure 15（L634–636）："BC results. Mean percentage of levels solved out of 100 samples, averaged over5 seeds with 95% confidence intervals."
- Figure 16（L1184–1186）："More example trajectories: the model is prompted with either hand-drawn sketches, images generated from text-to-image generative models or realistic photos. Actions that drive the dynamics of the trajectory are provided by human input."
- Figure 17（L1199–1203）："Controllable, consistent latent actions in Platformers: trajectories beginning from four different starting frames from our Platformers dataset. Each column shows the resulting frame from taking the same latent action five times. Despite training without action labels, not only are the same actions consistent across varied prompt frames, but also have semantic meaning:left, right, jump, and no-op."
- Table 1（L123–126）："Anewclassofgenerativemodel : Genie is a novel video and world model that is control- lable on a frame-by-frame basis, which requires only video dataat train time."
- Table 2（L646–647）："Latent action model input ablation. We see that Genie achieves higher controllability."
- Table 3（L663–665）："Tokenizer architecture ablation: Our ST-ViViT architecture results in the best perform- ing tokenizer."
- Table 4（L1234）："Effect of dataset curation."

## 附：关键超参数（附录，主要）

- LAM（Table 5，L1242–1252）：Encoder 20 层/d_model 1024/16 heads；Decoder 20 层/1024/16；Codebook num_codes 8、patch_size 16、latent_dim 32。
- Tokenizer（Table 7，L1266–1278）：Encoder 12 层/d_model 512/8 heads/kq 64；Decoder 20 层/1024/16/kq 64；Codebook num_codes 1024、patch_size 4、latent_dim 32；300k 步 AdamW 余弦衰减（Table 8：max_lr 3e-4、min_lr 3e-4、β1 0.9、β2 0.9、weight_decay 1e-4、warmup 10k）。
- 动力学模型优化器（Table 9，L1302–1309）：max_lr 3e-5、min_lr 3e-6、β1 0.9、β2 0.9、weight_decay 1e-4、warmup 5k。
- 可复现案例（附录 F）：CoinRun，"we collect data using a random policy... for a total of 10M transitions"（L1393–1396）；LAM 码本 num_codes 6（Table 16）；"We find this model generates consistent playable latent actions, resembling the original environment."（L1421–1422）
