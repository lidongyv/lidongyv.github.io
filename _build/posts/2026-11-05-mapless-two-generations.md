---
title: 无图三年：从 HD map 到 Mapless 的三次模型换代
title_en: "Three years without a map: three model generations from HD map to Mapless"
date: 2026-11-05
tags: 无图, 车道线, 数据闭环
summary: 2023 年交付国内第一套 Mapless 方案时，真正的难点不是「不要地图」，而是让模型在没有地图先验时依然能稳定输出拓扑。
summary_en: When we shipped China's first Mapless solution in 2023, the hard part was not removing the map — it was getting stable topology out of a model that no longer had map priors.
---

「无图」这个词很容易被误解成「把地图模块删掉」。实际上删掉地图之后，所有原来由地图免费提供的东西——拓扑、路口结构、车道归属——都要由模型自己学出来。

## 第一次换代：segment-based，所见即所得

最早的做法是「所见即所得」：模型只输出车道线和路沿，不输出中心线；靠超分、BEV 多尺度融合、动态多分辨率 DCN 把几何精度顶上去。为了让 GT 对得上，我们搭了 HD map → local BEV 的 GT 生成系统（200 万 image、1200 万 lane instance），把 segment-based 车道线检测从 AP 13% 拉到 60%、EPE 从 1.5 米压到 0.18 米。

这条路的天花板很明显：**它只能看见它看见的**。路口被遮挡、远处弱可见、车道线断掉的地方，模型没有能力补。

## 第二次换代：query-based，从「看见」到「推理」

query-based 方案把车道表示成「车道–车道线–点链」的多级回归，去掉后处理、把人力的位置往前挪，计算效率持平而 F1 提升 15%。更重要的是它能引入 navi info：路口直行还是左转、要不要 split/merge、是否换道，这些上游决策成了模型输入，模型开始能「猜」出没看见的部分。

代价是数据规则要重写：query-based 对标注规则、数据配平、case 级精调都更敏感。我们为此建了道路结构全要素、多任务、同源时序的数据集，并把 issue 解决周期从 D+10 压到 D+3。

## 第三次换代：token-based，为端到端做准备

第三次换代的目的不再是刷指标，而是**让车道线能进端到端**。以自回归方式 tokenize 感知元素之后，车道检测从「输出一张 BEV 图」变成「输出一串 token」，于是它可以和轨迹预测共享同一个解码器。

这一步也顺手解决了长尾问题：拓扑学习有三条路线并行试过——基于 language model 的自回归关键点生成、基于 MaskFormer 的拓扑关键点分割、以及把 query-based 点链生成「逆」成 segment 算法的 reverse transformer。三条路线最终都指向同一件事：**拓扑要被显式地生成，而不是从分割图里后处理出来**。

## 复盘

如果只允许留三条经验：

1. **无图的本质是拓扑建模**，不是检测精度。精度再高，没有拓扑就无法支撑决策。
2. **换表示之前先换数据规则**。query-based 的收益有一半来自标注规则和数据配平的重写。
3. **换代的判据是「能不能进端到端」**，不是「这一代指标涨了多少」。token-based 之所以值得做，是因为它让后面两代成为可能。

再往后就是世界模型那条线了，我在另一篇里写。
