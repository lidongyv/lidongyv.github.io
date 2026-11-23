---
title: 从模块化感知到单阶段 VLA：三次换代里反复出现的同一个坑
title_en: "From modular perception to one-stage VLA: the same trap, three generations in a row"
date: 2026-11-18
tags: VLA, 端到端, 感知
summary: 每一次「把感知融进端到端」，真正难的都不是模型结构，而是怎么在不破坏单任务指标的前提下，让多任务在同一份数据上共存。
summary_en: Every time we fold perception into end-to-end driving, the hard part is never the architecture — it is keeping multi-task training alive on the same data without breaking single-task metrics.
---

我把 2021 年以来的工作按「感知在不在策略里」分成三代：模块化感知、两阶段端到端、单阶段 VLA。三代的技术细节差别很大，但每一代我都撞在同一堵墙上：**多任务指标与单模型之间的取舍**。

## 第一代：模块化，指标好看但学不动

蔚来时期做的是标准模块化：Mono3D 检测跟踪 + 车道线 + 后处理，靠数据闭环和评测框架把每个模块的指标单独推上去。关联后处理上，我们用 7 相机 + 1 激光雷达的规则化关联，后来改成可学习的 cost metric 替代规则与二分图匹配，在 20% 三维误差率下做到 0.03 的 ID switch 和 86% 召回。

这一代的优点是每个指标都能解释、都能回滚。缺点也很清楚：**误差会累积**，而且模块之间的接口是人为定义的，模型学不到接口之外的信息。

## 第二代：两阶段端到端，感知被「重构成 token」

NVIDIA 时期的核心动作是把感知任务重新组织，让它们适配 E2E：Lidar 前融合、检测、把 MOT 升级成 object prediction、加一个支持 pre/post-training 的 future tokenizer。车道线检测走了一条很长的路：segment-based（所见即所得，靠超分、BEV 多尺度融合、动态多分辨率 DCN 提精度）→ query-based（引入 navi info，能过路口、能 split/merge、能换道）→ token-based（以自回归方式 tokenize 所有感知元素）。

token-based 那一步是分水岭。它让两阶段能升级到单阶段，因为**感知元素一旦变成 token，规划就可以直接把它们当语言来读**。

## 第三代：单阶段，规划直接吃图像流

到了单阶段，planning 从「基于感知结果」改成「基于图像流」，perception tokenizer 被去掉、降级成 debugger 任务。轨迹预测的形态也一路换：单模态 → 多模态回归 → 自回归 → 扩散，最后叠 RL post-training 和世界模型闭环训练。

## 反复出现的那个坑

三代下来，同一个问题出现了三次：**你怎么在同一个模型里放多个任务，而不让它们互相拖累？**

我们在新石器给出的答案是 group query-based 模型 + single-head 多任务：从 multi-head 改成 single-head，让数据配平变成模型自己可以配置的事；而且配平发生在 token 级别（BEV token / image token），不是在 task-loss 级别。query 分组这件事在地平线就验证过一遍——它把单任务与多任务的指标差距拉平了，离散要素（箭头、斑马线、路口）用 query-based 替换两阶段检测器后分数持平、计算效率提升 30% 以上。

我的结论是：

1. **多任务的关键不在 loss 加权，在表示层**。task-loss 级别的加权是在结果上补救，token 级别的重配平是在原因上解决。
2. **单任务是多任务的子集**。如果一个架构不能把单任务当作 N=1 的多任务来跑，它迟早会分裂成两套代码。
3. **能随时加数据比一次训到最好更重要**。frozen head 预训练 backbone、frozen backbone 微调 perception head，这套可控迭代方式在工程上的价值远大于刷点。

下一篇我会写数据闭环那部分——因为在 VLA 里，模型结构反而是最容易追平的部分。
