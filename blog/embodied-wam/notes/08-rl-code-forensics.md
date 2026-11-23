# WAM RL post-training 代码取证：VAMPO 完整 RL 循环（08）

> 取证时间：2026-11-12 ｜ 仓库：https://github.com/OpenHelix-Team/VAMPO（main 分支）
> 铁律：以下全部为代码 verbatim（文件路径+关键行）；结论只基于这些代码事实。
> 取证文件：`grpo/eas_with_logprob.py`、`grpo/reward_fn.py`、`step1_train_svd_grpo.py`、`scripts/train_calvin_svd_grpo.sh`

---

## 1. 采样器：首步 SDE + log-prob（`grpo/eas_with_logprob.py`）

**Euler Ancestral 步骤 + 对数概率**（`eas_step_with_logprob`）：
```python
prev_sample_mean = sample + derivative * dt          # ODE 漂移部分
if prev_sample is None:
    noise = randn_tensor(...)                        # 首步：真正采样噪声（SDE）
    prev_sample = prev_sample_mean + noise * sigma_up
    ...
else:
    noise = (prev_sample - prev_sample_mean) / sigma_up   # 后续步：反推噪声（ODE，sigma_up=0）
log_prob_calculated = -0.5 * (noise ** 2) - torch.log(sigma_up) - 0.5 * math.log(2 * math.pi)
log_prob = log_prob_calculated.mean(dim=tuple(range(2, prev_sample.ndim)))  # 空间维平均 → (B,C)
```
→ 与论文 Eq.(14)「可重参数化的高斯转移密度」对应：`prev_sample = None` 分支是随机首步，其余步走确定性 ODE。

**EDM 式 SDE 变体**（`stochastic_eds_step_with_logprob`，`eta=0.1` 控制随机强度）：
```python
drift = (1.0 - eta) * derivative * delta_sigma
diffusion_std = eta * torch.sqrt(torch.abs(delta_sigma))
```

## 2. 奖励函数（`grpo/reward_fn.py`）

**潜空间奖励**（论文 r = −λ_L1‖x_σ0−x_0‖₁ + λ_cos cos 的代码本体，`latent_reward_fn`）：
```python
mae = F.l1_loss(pred_latents_chunk, true_latents_chunk, reduction='none').mean(dim=[2, 3, 4])
reward_mae = - mae                       # 负 MAE 作奖励
# cosine：空间维展平后 L2 归一化再点积
pred_latents_norm = F.normalize(pred_latents_flat, p=2, dim=-1)
cosine_sim = torch.sum(pred_latents_norm * true_latents_norm, dim=-1)   # (B, F)
final_reward += mae_weight * final_reward_mae
final_reward += cosine_weight * final_reward_cosine
```
默认 `mae_weight=1.0, cosine_weight=1.0, decode_chunk_size=8`——奖励在**潜空间**直接算，不解码像素。
另有 `pixel_reward_fn`（VAE 解码后 MAE + LPIPS 感知奖励）作备选。

## 3. GRPO 训练循环（`step1_train_svd_grpo.py`）

**模型冻结策略**：
```python
freeze_models([vae, text_encoder, unet])     # 先全冻结
# 再按 trainable_modules 解冻 UNet 层；LoRA 可选
target_modules = ['to_k', 'to_q', 'to_v','out','proj','ff.net.','ff_in.net.','conv_out','conv_in']
```
**采样阶段（inference_mode）**：
```python
sde_start = 0
sde_end = 1
train_num_timesteps = sde_end - sde_start          # =1：只有首步参与梯度
pred_latents, latents_trajectory, log_probs_trajectory, sigma_ups_trajectory, ... = \
    SamplingStableVideoDiffusionPipeline.__call__(
        pipeline, image=image_latent, text=text_token,
        num_videos_per_prompt=args.num_generations,  # G=8
        sde_start=sde_start, sde_end=sde_end, ...)
all_rewards = latent_reward_fn(pred_latents, true_latents.repeat(num_generations, 1,1,1,1), vae=vae)
```
**优势 = 组归一化（GRPO 特征：无 critic/价值网络）**：
```python
rewards = rewards.reshape(sample_batch_size, num_generations, args.num_frames)
advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)
```
**策略更新 = PPO 式截断代理目标**：
```python
ratio = torch.exp(log_prob - batch_log_probs)
clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * clipped_ratio
loss = torch.mean(torch.minimum(unclipped_loss, clipped_loss))
approx_kl = 0.5 * torch.mean((log_prob - batch_log_probs) ** 2)
clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
```

## 4. 启动脚本与超参（`scripts/train_calvin_svd_grpo.sh`，verbatim）

```bash
pretrained_model_path=checkpoints/svd-robot-calvin-ft   # 先 SFT 的 VPM 检查点
sample_batch_size=8
train_batch_size=8
learning_rate=1e-6
max_train_steps=10000
checkpointing_steps=50
validation_steps=50
train_args.use_lora=False
train_args.lora_rank=16
train_args.num_generations=8            # G=8 组采样
train_args.num_inner_epochs=1
train_args.num_inference_steps=20       # I=20 去噪步
train_args.guidance_scale=7.5
train_args.clip_range=0.2
train_args.adv_clip_max=2.0
train_args.max_grad_norm=0.5
```
8 GPU accelerate 启动；lr=1e-6；训练只更新 UNet（VPM 去噪器），VAE/文本编码器冻结。

## 5. 代码与论文一致性核对

| 论文表述 | 代码对应 | 一致 |
|---|---|---|
| 「first step SDE, remaining steps ODE」 | `sde_start=0, sde_end=1` + `eas_step_with_logprob` 的 `prev_sample is None` 分支 | ✓ |
| r = −λ_L1‖x_σ0−x_0‖₁ + λ_cos cos | `latent_reward_fn` 的 `reward_mae` + `reward_cosine` | ✓（λ 默认均 1.0） |
| GRPO（组归一化优势） | `advantages = (rewards - mean) / (std + 1e-8)` over G 维 | ✓ |
| J(θ) 目标（论文 §3.3 截断未取到） | PPO 式 `minimum(unclipped, clipped)` 截断代理 | 补全 |
| 「只 post-train VPM」 | `freeze_models([vae, text_encoder, unet])` 后按模块解冻 UNet；AGM 不在本仓库训练循环内 | ✓（AGM 属另一 policy 训练脚本） |

## 6. 未能获取/未核实

- 本仓库为 VPM（视频预测）侧训练；AGM（动作生成）的 RL 是否同时进行未在本文件核实。
- 论文 §4 的实验数值（GRPO 提升幅度）仍待 07 线取实验节。
- `pipeline_with_logprob_svd.py`、`SamplingStableVideoDiffusionPipeline` 内部实现未逐行读（本轮只取证 RL 目标与奖励核心）。
