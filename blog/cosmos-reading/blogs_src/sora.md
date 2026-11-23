# Video generation models as world simulators (Sora)

Source: OpenAI technical report — https://openai.com/index/video-generation-models-as-world-simulators/
Date: February 15, 2024
Authors: Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, Aditya Ramesh

## Abstract
We explore large-scale training of generative models on video data. Specifically, we train text-conditional diffusion models jointly on videos and images of variable durations, resolutions and aspect ratios. We leverage a transformer architecture that operates on spacetime patches of video and image latent codes. Our largest model, Sora, is capable of generating a minute of high fidelity video. Our results suggest that scaling video generation models is a promising path towards building general purpose simulators of the physical world.

This technical report focuses on (1) our method for turning visual data of all types into a unified representation that enables large-scale training of generative models, and (2) qualitative evaluation of Sora's capabilities and limitations. Model and implementation details are NOT included in this report.

Prior work studied generative video modeling via recurrent networks, GANs, autoregressive transformers, and diffusion models — often on narrow categories, shorter videos, or fixed sizes. Sora is a generalist model of visual data — it can generate videos and images spanning diverse durations, aspect ratios and resolutions, up to a full minute of HD video.

## Turning visual data into patches
Inspired by LLMs that gain generalist capabilities from internet-scale data using tokens that unify diverse modalities of text. Whereas LLMs have text tokens, Sora has visual patches. At a high level, videos are turned into patches by first compressing videos into a lower-dimensional latent space, and subsequently decomposing the representation into spacetime patches.

## Video compression network
A network reduces the dimensionality of visual data: raw video in → latent representation compressed both temporally and spatially. Sora trains and generates within this compressed latent space. A corresponding decoder maps generated latents back to pixel space.

## Spacetime latent patches
Given a compressed input video, extract a sequence of spacetime patches which act as transformer tokens. Works for images too (image = single-frame video). Patch-based representation enables training on videos/images of variable resolutions, durations, aspect ratios. At inference, control the size of generated videos by arranging randomly-initialized patches in an appropriately-sized grid.

## Scaling transformers for video generation
Sora is a diffusion model: given input noisy patches (and conditioning like text prompts), it is trained to predict the original "clean" patches. Importantly, Sora is a diffusion transformer (DiT). Diffusion transformers scale effectively as video models; sample quality improves markedly as training compute increases.

## Variable durations, resolutions, aspect ratios
Training on data at its native size (rather than resize/crop to e.g. 256x256 4s) gives: sampling flexibility (1920x1080 to 1080x1920 and between); improved framing and composition (square-crop training sometimes shows subjects only partially in view).

## Language understanding
Applies the re-captioning technique from DALL-E 3: train a highly descriptive captioner model, then use it to produce captions for all training videos — improves text fidelity and overall quality. At inference, GPT turns short user prompts into longer detailed captions sent to the video model.

## Prompting with images and videos
Beyond text-to-video, Sora can be prompted with pre-existing images or video: animating DALL-E images; extending videos forward/backward in time (incl. seamless infinite loops); video-to-video editing (SDEdit, zero-shot style/environment transform); connecting/interpolating between two input videos. Image generation too (patches of Gaussian noise in a spatial grid, one-frame temporal extent), up to 2048x2048.

## Emerging simulation capabilities
When trained at scale, video models exhibit emergent capabilities — WITHOUT explicit inductive biases for 3D/objects (purely phenomena of scale):
- 3D consistency: dynamic camera motion; people/scene move consistently in 3D as camera shifts/rotates.
- Long-range coherence & object permanence: persists people/animals/objects even when occluded or leaving frame; multiple shots of same character.
- Interacting with the world: actions that affect state in simple ways (painter leaves strokes that persist; man eats burger and leaves bite marks).
- Simulating digital worlds: can control a Minecraft player with a basic policy while rendering the world in high fidelity, zero-shot via captions mentioning "Minecraft".

## Discussion / limitations
Sora currently exhibits many limitations as a simulator: does not accurately model physics of many basic interactions (e.g., glass shattering); interactions like eating food do not always yield correct changes in object state; incoherencies develop in long-duration samples; spontaneous appearances of objects. Still, the authors argue continued scaling of video models is a promising path towards capable simulators of the physical and digital world.

(References [1]-[32] omitted here; see original technical report. Notable: Ha & Schmidhuber "World Models" 2018; Peebles & Xie "Scalable Diffusion Models with Transformers" (DiT) 2023; Rombach et al. latent diffusion 2022.)
