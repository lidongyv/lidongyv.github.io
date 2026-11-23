# lidongyv 自有仓库盘点（非 fork）

> 数据源：GitHub REST API `/users/lidongyv/repos`（未认证，8 页 × `per_page=17`，全量落盘后解析）。
> 抓取时间：2026-11-23。口径：`public_repos = 134`，其中 **fork = 95，非 fork（自己创建）= 39**。
> star 数为抓取时快照；"最后 push"取 `pushed_at`。

## 0. TL;DR

- **真正属于自己的仓库只有 39 个**；其中 23 个属课程作业/玩具/占位，这 23 个里有 16 个 push 停在 2015 年，且**有 9 个是空仓库**（`size = 0`，无任何文件）。年份分布：2015 年 16 个、2016 年 7 个、2017 年 4 个、2018 年 1 个、2019 年 3 个、2020 年 7 个、2026 年 1 个（空仓库噪音）。
- 有技术含量的集中在 **11 个 research 仓库（2017–2020）**：立体匹配（stereo matching）、单目深度估计（monocular depth）、单目标跟踪（SOT/MOT）、World Models、GC-Net 复现。全部围绕本人 bio 写的 depth estimation / navigation / tracking。
- **`finance` 类为 0**：量化/交易只出现在 fork 里（vnpy、AlphaGPT 等），不是自己写的，不要放到个人主页当作品。
- **2020-09 之后没有新的自有代码仓库**（唯一"2026 push"的 `3d` 是空仓库的噪音记录），个人主页应主打 2018–2020 那批 depth/stereo 工作。
- 39 个仓库的 star **合计只有 10 个**（最高 3，为 `tensorflow_trick`），所以**排序不该按 star，应按"主题相关性 + 是否有报告/项目页"**。

## 1. 非 fork 仓库总表（按最后 push 倒序）

| # | 仓库（链接） | 语言 | 最后 push | ★ | 描述（原文，未改写） |
|---:|---|---|---|---:|---|
| 1 | [3d](https://github.com/lidongyv/3d) | — | 2026-01-05 | 0 | so difficult experiment |
| 2 | [Reppoint-Tracking](https://github.com/lidongyv/Reppoint-Tracking) | Jupyter Notebook | 2020-09-17 | 0 | mmdet and reppoint tracking framework |
| 3 | [SOT-Projects](https://github.com/lidongyv/SOT-Projects) | Python | 2020-09-17 | 0 | （描述为空） |
| 4 | [Explicit-Context-Mapping-for-Stereo-Matching](https://github.com/lidongyv/Explicit-Context-Mapping-for-Stereo-Matching) | Python | 2020-09-16 | 1 | context mapping function |
| 5 | [Interpretable-stereo-matching-with-region-support](https://github.com/lidongyv/Interpretable-stereo-matching-with-region-support) | Python | 2020-09-16 | 0 | pixel-set stereo matching |
| 6 | [Monocular-depth-esitimation-with-region-support-cvpr](https://github.com/lidongyv/Monocular-depth-esitimation-with-region-support-cvpr) | Python | 2020-09-16 | 0 | region support coarse-to-fine network |
| 7 | [RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture](https://github.com/lidongyv/RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture) | Python | 2020-05-31 | 1 | reference aggregation network |
| 8 | [Thomas-S-Huang-Papers](https://github.com/lidongyv/Thomas-S-Huang-Papers) | — | 2020-04-27 | 1 | （描述为空） |
| 9 | [3dndiv](https://github.com/lidongyv/3dndiv) | Python | 2019-11-20 | 0 | 3d representation modeled by ndiv loss |
| 10 | [gym-car](https://github.com/lidongyv/gym-car) | Python | 2019-08-07 | 0 | test of the navie perception and controller system |
| 11 | [world-pytorch](https://github.com/lidongyv/world-pytorch) | Python | 2019-06-14 | 0 | world model realized by pytorch |
| 12 | [Monocular-depth-esitimation-with-region-support-nips](https://github.com/lidongyv/Monocular-depth-esitimation-with-region-support-nips) | Python | 2018-08-23 | 0 | Region Support for Depth Estimation from a single image |
| 13 | [tensorflow_trick](https://github.com/lidongyv/tensorflow_trick) | Python | 2017-12-26 | 3 | the tricks for tensorflow |
| 14 | [GCNet](https://github.com/lidongyv/GCNet) | Python | 2017-11-02 | 1 | GC |
| 15 | [DM-SearchForImages](https://github.com/lidongyv/DM-SearchForImages) | — | 2017-04-13 | 0 | DM final turotial |
| 16 | [DM-HomeWork-HorseIllness](https://github.com/lidongyv/DM-HomeWork-HorseIllness) | Python | 2017-04-13 | 0 | DM homework |
| 17 | [matlab](https://github.com/lidongyv/matlab) | Matlab | 2016-08-08 | 0 | some matlab trick |
| 18 | [matlab-rename](https://github.com/lidongyv/matlab-rename) | — | 2016-08-08 | 0 | matlab读取文件，重命名文件，另存为文件 |
| 19 | [Projects](https://github.com/lidongyv/Projects) | — | 2016-02-18 | 0 | VS2013_PROJECT |
| 20 | [Visual-Studio-2013](https://github.com/lidongyv/Visual-Studio-2013) | — | 2016-02-11 | 0 | （描述为空） |
| 21 | [eclipse](https://github.com/lidongyv/eclipse) | — | 2016-02-11 | 0 | （描述为空） |
| 22 | [python_sample](https://github.com/lidongyv/python_sample) | — | 2016-02-04 | 0 | test for opencv3 |
| 23 | [matlab_sample](https://github.com/lidongyv/matlab_sample) | — | 2016-02-04 | 0 | test for matlab |
| 24 | [cg3DTrans-Final](https://github.com/lidongyv/cg3DTrans-Final) | C++ | 2015-08-06 | 0 | hard but helpful |
| 25 | [opengl](https://github.com/lidongyv/opengl) | C | 2015-08-06 | 0 | test model |
| 26 | [tooth-control-in-unity](https://github.com/lidongyv/tooth-control-in-unity) | ASP | 2015-08-06 | 1 | use unity to control maya model |
| 27 | [LibrarySystem](https://github.com/lidongyv/LibrarySystem) | — | 2015-08-06 | 0 | when i'm young,writen by c++ |
| 28 | [Xinge_back7](https://github.com/lidongyv/Xinge_back7) | Java | 2015-08-06 | 0 | no dataform ,complete version in privacy |
| 29 | [Robot](https://github.com/lidongyv/Robot) | C++ | 2015-08-06 | 0 | old version,the newest version for competition under construction |
| 30 | [TEST_MVC](https://github.com/lidongyv/TEST_MVC) | Java | 2015-08-06 | 0 | the mvc for java ;created for book store |
| 31 | [project](https://github.com/lidongyv/project) | — | 2015-08-06 | 0 | for gis system |
| 32 | [ShipDemo](https://github.com/lidongyv/ShipDemo) | HTML | 2015-03-10 | 2 | shipdemo addition done |
| 33 | [six](https://github.com/lidongyv/six) | C++ | 2015-03-10 | 0 | add sixqi |
| 34 | [my_rabbitmq_test](https://github.com/lidongyv/my_rabbitmq_test) | Java | 2015-03-03 | 0 | from rabbitmq.com |
| 35 | [Python-XML2Dict](https://github.com/lidongyv/Python-XML2Dict) | C | 2015-03-03 | 0 | for a test |
| 36 | [setup](https://github.com/lidongyv/setup) | — | 2015-02-10 | 0 | setup a respository |
| 37 | [test3](https://github.com/lidongyv/test3) | — | 2015-02-10 | 0 | test3 |
| 38 | [test2](https://github.com/lidongyv/test2) | — | 2015-02-10 | 0 | test2 |
| 39 | [test](https://github.com/lidongyv/test) | — | 2015-02-10 | 0 | test for information |

## 2. 分类建议（逐仓库判断 + 一句话说明）

说明：`archive` 严格按"课程作业/玩具/空壳"归，含 2017 年数据挖掘课程作业（超出 2016 线但性质一致，已标注）。下列 39 条与总表一一对应、无遗漏。

### research（11 个）

| 仓库 | 链接 | 一句话说明 |
|---|---|---|
| SOT-Projects | https://github.com/lidongyv/SOT-Projects | 单目标跟踪（SOT）+ 特征解耦，基于 PySOT 改写，README 指向 lidongyv.github.io/project6.html 与 assets/pdf/sot.pdf 技术报告。 |
| Reppoint-Tracking | https://github.com/lidongyv/Reppoint-Tracking | 把 mmdet 检测器和 RepPoints 表示接到多目标跟踪上的训练/评测框架，是目前体量最大的工程级仓库（43 MB）。 |
| Monocular-depth-esitimation-with-region-support-cvpr | https://github.com/lidongyv/Monocular-depth-esitimation-with-region-support-cvpr | 单目深度估计：用 region support 做由粗到细（coarse-to-fine）的深度回归网络，是"region support"系列的主仓库。 |
| Monocular-depth-esitimation-with-region-support-nips | https://github.com/lidongyv/Monocular-depth-esitimation-with-region-support-nips | 同一思路的早期 NIPS 版本原型，只保留核心实验代码（730 KB），属前者的前身。 |
| RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture | https://github.com/lidongyv/RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture | 深度立体匹配中的显式代价聚合子网络 RAN（Reference Aggregation Network），可插拔模块式贡献。 |
| Explicit-Context-Mapping-for-Stereo-Matching | https://github.com/lidongyv/Explicit-Context-Mapping-for-Stereo-Matching | 立体匹配里显式建模上下文映射（context mapping）的实现，处理弱纹理/重复纹理区域。 |
| Interpretable-stereo-matching-with-region-support | https://github.com/lidongyv/Interpretable-stereo-matching-with-region-support | 以像素集合（pixel-set）为单位的可解释立体匹配，强调中间结果可视化与可解释性。 |
| 3dndiv | https://github.com/lidongyv/3dndiv | 用 NDIV loss 学习 3D 表征的实验代码，全仓最大（334 MB，含数据/权重），偏研究实验记录。 |
| GCNet | https://github.com/lidongyv/GCNet | 复现 GC-Net（几何+上下文立体深度估计）的代码（README 明确写 "reproduce GC-Net"），描述字段只有一个 "GC"。 |
| world-pytorch | https://github.com/lidongyv/world-pytorch | World Models（Ha & Schmidhuber）的 PyTorch 复现，含 VAE + RNN 世界模型与 controller，对应 imitation learning 方向。 |
| gym-car | https://github.com/lidongyv/gym-car | 自建仿真环境里联调"感知 + 控制器"闭环的试验代码，是从感知走向导航/控制的雏形。 |

### media-blog（1 个）

| 仓库 | 链接 | 一句话说明 |
|---|---|---|
| Thomas-S-Huang-Papers | https://github.com/lidongyv/Thomas-S-Huang-Papers | 整理黄煦涛（Thomas S. Huang）31 篇最高引论文的纪念性资料库，topics 打了 collection/paperreading/proceedings，属知识库而非代码。 |

### tools（4 个）

| 仓库 | 链接 | 一句话说明 |
|---|---|---|
| tensorflow_trick | https://github.com/lidongyv/tensorflow_trick | TensorFlow 使用技巧/踩坑片段集合（3 star，自有仓库里最高），只有 1 KB，更像速查笔记。 |
| matlab | https://github.com/lidongyv/matlab | MATLAB 小技巧脚本合集（2 KB），日常工具箱性质。 |
| matlab-rename | https://github.com/lidongyv/matlab-rename | MATLAB 批量"读文件→重命名→另存"的脚本，纯实用工具；但仓库 `size = 0`，实际没内容。 |
| Python-XML2Dict | https://github.com/lidongyv/Python-XML2Dict | 把 XML 转成 Python dict 的小工具/试验代码（2015），语言被识别为 C，属"小而可用"的脚本。 |

### finance（0 个）

自有仓库里**没有任何**量化/交易/因子/行情类仓库。相关的 `vnpy`、`AlphaGPT`、`ziwei-doushu` 等在 API 里 `fork = true`，是他人的仓库，不应计入作品集。若主页要保留 finance 板块，只能写成"关注/学习"，不能写成"我做过的项目"。

### archive（23 个，只归档、不上主页）

| 仓库 | 链接 | 一句话说明 |
|---|---|---|
| DM-HomeWork-HorseIllness | https://github.com/lidongyv/DM-HomeWork-HorseIllness | 2017 年北理工数据挖掘课作业（马病分类数据集），KNN/决策树那一类课程练习。 |
| DM-SearchForImages | https://github.com/lidongyv/DM-SearchForImages | 同门数据挖掘课的期末 tutorial（图像检索），带 GitHub Pages 课件页。 |
| cg3DTrans-Final | https://github.com/lidongyv/cg3DTrans-Final | 2015 年计算机图形学课程大作业：C++ 实现的 3D 变换/渲染（50 MB），无 README。 |
| opengl | https://github.com/lidongyv/opengl | 2015 年 OpenGL 模型显示试验（C，21 MB），课程/自学练手。 |
| tooth-control-in-unity | https://github.com/lidongyv/tooth-control-in-unity | 用 Unity 驱动 Maya 牙齿模型的本科项目（ASP 标签，12 MB），跨界玩具项目。 |
| LibrarySystem | https://github.com/lidongyv/LibrarySystem | C++ 写的图书管理系统课程设计，描述自带"when i'm young"。 |
| TEST_MVC | https://github.com/lidongyv/TEST_MVC | Java MVC 模式练习：书店（book store）示例，课程作业。 |
| Xinge_back7 | https://github.com/lidongyv/Xinge_back7 | Java 后端练习项目（作者注明完整版在私有库），9 MB，2015。 |
| Robot | https://github.com/lidongyv/Robot | 竞赛机器人控制程序的旧版本（作者说新版在私有库），C++。 |
| six | https://github.com/lidongyv/six | "add sixqi"，从命名与 C++ 代码看是六子棋类对弈程序，无 README，20 MB。 |
| ShipDemo | https://github.com/lidongyv/ShipDemo | HTML 做的船舶演示/Demo 页（2 star，14 MB），2015 年前端试水。 |
| my_rabbitmq_test | https://github.com/lidongyv/my_rabbitmq_test | 照 rabbitmq.com 官方示例做的 Java 消息队列练手。 |
| project | https://github.com/lidongyv/project | 一句话"for gis system"的 GIS 课程项目，`size = 0`，实际空壳。 |
| Projects | https://github.com/lidongyv/Projects | 名为 "VS2013_PROJECT" 的占位仓库，`size = 0`。 |
| Visual-Studio-2013 | https://github.com/lidongyv/Visual-Studio-2013 | 空仓库，连描述都没有，仅作 IDE 相关占位。 |
| eclipse | https://github.com/lidongyv/eclipse | 空仓库，同上，占位用。 |
| python_sample | https://github.com/lidongyv/python_sample | "test for opencv3"，空仓库（0 KB），无内容。 |
| matlab_sample | https://github.com/lidongyv/matlab_sample | "test for matlab"，空仓库（0 KB）。 |
| setup | https://github.com/lidongyv/setup | 建仓测试（"setup a respository"），无实际代码。 |
| test3 | https://github.com/lidongyv/test3 | 建仓测试仓库，112 KB，无意义内容。 |
| test2 | https://github.com/lidongyv/test2 | 建仓测试仓库，空。 |
| test | https://github.com/lidongyv/test | 建仓测试仓库，空。 |
| 3d | https://github.com/lidongyv/3d | 2015 年建的 3D 实验占位，`size = 0`；2026-01 有一次空 push，导致它排在总表第一，属噪音。 |

## 3. 最有展示价值的 10 个（建议放在首屏）

排序逻辑：**主题是否贴合本人定位（depth / stereo / tracking / imitation learning）> 是否有配套报告或项目页 > 工程完整度 > star**。star 普遍很低（最高 3），所以不按 star 排。

| 序 | 仓库 | 理由 |
|---:|---|---|
| 1 | [SOT-Projects](https://github.com/lidongyv/SOT-Projects) | 唯一"代码 + 技术报告 PDF + 项目页（project6.html）"三件套齐全的仓库，跟踪 + 特征解耦主题清晰，招聘方 30 秒能看懂你在做什么。 |
| 2 | [Reppoint-Tracking](https://github.com/lidongyv/Reppoint-Tracking) | 把检测（mmdet）和 RepPoints 表征接成完整 MOT 训练框架，43 MB 工程体量，能证明"能搭框架"而不只是跑 demo。 |
| 3 | [Monocular-depth-esitimation-with-region-support-cvpr](https://github.com/lidongyv/Monocular-depth-esitimation-with-region-support-cvpr) | 单目深度估计主线成果（region support + coarse-to-fine），有 Pages 页，和 bio 里 "depth estimation" 直接对应。 |
| 4 | [RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture](https://github.com/lidongyv/RAN-Deep-Stereo-Matching-with-Explicit-Cost-Aggregation-Sub-Architecture) | 名字即论文贡献：显式代价聚合子网络 RAN，可插拔模块，方法层面最有"研究味"的一个。 |
| 5 | [Explicit-Context-Mapping-for-Stereo-Matching](https://github.com/lidongyv/Explicit-Context-Mapping-for-Stereo-Matching) | 立体匹配里显式上下文映射，解决弱纹理/重复纹理，与第 4 项构成一条完整 stereo 研究线。 |
| 6 | [Interpretable-stereo-matching-with-region-support](https://github.com/lidongyv/Interpretable-stereo-matching-with-region-support) | 以 pixel-set 为单位的可解释立体匹配，"可解释性"是加分标签，且有可视化可截图。 |
| 7 | [world-pytorch](https://github.com/lidongyv/world-pytorch) | World Models 的 PyTorch 复现（VAE + RNN + controller），展示 imitation learning / 表征学习理解深度。 |
| 8 | [3dndiv](https://github.com/lidongyv/3dndiv) | NDIV loss 建模 3D 表征，334 MB 是全部自有仓库里体量最大的，说明真跑过实验。 |
| 9 | [GCNet](https://github.com/lidongyv/GCNet) | GC-Net 立体深度复现，证明能读论文落到代码；描述字段只有 "GC"，上主页前务必补 README 摘要。 |
| 10 | [gym-car](https://github.com/lidongyv/gym-car) | 仿真里跑"感知 + 控制器"闭环，是把感知接到决策控制的证据，和自动驾驶叙事最贴。 |

**备选 2 个**（首屏放不下时可放第二屏）：`Thomas-S-Huang-Papers`（31 篇黄煦涛高引论文整理，体现学术品味，但非代码）、`tensorflow_trick`（3 star，自有仓库最高星，但内容仅 1 KB，说服力有限）。

**上主页前必须做的三件事**：
1. 给 `SOT-Projects`、`GCNet` 补一句 README 首段（现在一个描述为空、一个只有 "GC"）。
2. `region-support` 三兄弟（nips / cvpr / interpretable）建议合并成一个入口，用 cvpr 版当主仓库，另外两个 archive 或 archive=1，否则首屏会出现三个近重复项。
3. `3d` 的空 push 让它排在"最近更新"第一位，务必 fix（删仓库或让它 archive），否则访客第一眼看到的是 `so difficult experiment`。

## 4. 分类汇总计数

| 分类 | 数量 | 说明 |
|---|---:|---|
| research | 11 | 立体匹配 4 + 单目深度 2 + 跟踪 2 + 3D 表征 1 + 世界模型 1 + 仿真闭环 1 |
| finance | 0 | 自有仓库为空；量化相关全部是 fork（vnpy / AlphaGPT 等） |
| tools | 4 | matlab ×2、tensorflow_trick、Python-XML2Dict |
| media-blog | 1 | Thomas-S-Huang-Papers（论文资料库） |
| archive | 23 | 2015–2017 课程作业/玩具 14 个 + 空仓库 9 个 |
| **合计（非 fork）** | **39** | 占全部 134 个公开仓库的 29.1%；其余 95 个为 fork |
