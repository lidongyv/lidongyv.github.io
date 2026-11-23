# WAMotor 代码取证：投机推理实现（11）

> 取证时间：2026-11-21 ｜ 仓库：https://github.com/Overdue-Lin/WAMotor（main，HTTP 200 实抓确认）
> 核心文件：`src/wamotor/models/wan22/speculative/action.py`（投机动作去噪循环，verbatim 摘录）
> 铁律：以下全部为代码事实；伪代码「按官方代码结构」标注。

---

## 1. 函数签名与默认超参（verbatim）

```python
def speculative_denoise_action(model, latents_action,
    infer_timesteps_action, infer_deltas_action,
    context, context_mask, video_kv_cache, attention_masks, video_seq_len,
    w: int = 10,          # speculation window cap（投机窗口上限）
    epsilon: float = 1e-2,# 验证阈值系数
) -> torch.Tensor:
```

## 2. 时间步感知验证阈值的代码本体（与论文 §4.4 对应）

```python
abs_deltas = infer_deltas_action.float().abs()
reference_delta = abs_deltas.median().clamp_min(tiny)
thresholds = epsilon * (reference_delta / abs_deltas.clamp_min(tiny)) ** 2
```
→ 每步阈值 = ε × (中位步长 / 该步步长)² ：**步长越大阈值越紧**（"tightening verification for larger steps"），正是论文的「双非均匀」设计之二。

## 3. 投机循环的代码结构（关键行 verbatim）

```python
# Always take the first transition with ordinary Euler integration.
velocity = predict(latents_action, infer_timesteps_action[:1])
latents_action = model.infer_action_scheduler.step(velocity, infer_deltas_action[0], latents_action)
m = 1
...
draft = (latents_action.unsqueeze(0)
         + torch.cumsum(infer_deltas_action[m:end].float(), dim=0)...
         * velocity.unsqueeze(0))                     # 一个 anchor velocity 复用草拟 w 步
validated = predict(draft, candidate_timesteps)       # 一次 batched 调用验证全部候选
mse = (validated - velocity).square().mean(dim=-1)
valid = mse.lt(threshold).all(dim=-1).all(dim=-1)     # 逐步 MSE 与阈值比较
accepted = count if invalid.numel() == 0 else invalid[0].item()   # 接受最长合法前缀
```

## 4. 伪代码（按官方代码 action.py 结构）

```
speculative_denoise_action(model, a_0, {t_k}, {Δ_k}, c, w=10, ε=1e-2):
  ref = median(|Δ_k|)                        # 参考步长
  thr_k = ε · (ref / |Δ_k|)²                 # 时间步感知阈值（步长大→阈值紧）
  1. 首步强制普通 Euler：u ← v_θ(a_0, t_0, c)；a_1 ← step(u, Δ_0)；m ← 1
  2. while m < N-1:
       if velocity 为空: u ← v_θ(a_m, t_m, c)
       end ← min(m + w, N-1)
       draft_j ← a_m + u · Σ_{i=0}^{j-1} Δ_{m+i},  j = 1..end-m   # anchor velocity 线性外推
       v̂_j ← v_θ(draft_j, t_{m+j}, ĉ_j)        # 一次 batched 验证调用
       r_j ← u − v̂_j；accept j 当且仅当 ‖r_j‖² < thr_{m+j}（逐 token 判定）
       接受最长合法前缀 a_acc；m ← m + acc
       - 若全部接受：a ← draft_end，u ← v̂_end（缓存 velocity 供下一窗口复用）
       - 否则：在边界做一次标准 Euler 回退 step，u 失效置空
  3. 返回 a_N
```

## 5. 代码与论文一致性核对

| 论文表述 | 代码对应 | 一致 |
|---|---|---|
| Stage I「defer speculation，先做一次标准 Euler」 | `# Always take the first transition with ordinary Euler integration.` + `m = 1` | ✓ |
| Stage II「time-step-aware verification…按步长缩放阈值」 | `thresholds = ε * (median_delta / |Δ_k|)²` | ✓ |
| anchor velocity 草拟 q=min(w, N−1−m) 步（Eq.1） | `draft = a_m + cumsum(Δ)·velocity`，窗口 cap `w=10` | ✓ |
| 一次 batched 调用验证所有候选 | `validated = predict(draft, candidate_timesteps)` | ✓ |
| 接受最长合法前缀并 commit | `accepted = invalid[0]`（首个违例位置截断） | ✓ |

## 6. 仓库结构要点（集成范围证据）

- 支持三模型：`fastwam.py` / `fasterwam.py` / `jointwam.py`（配置 `configs/model/*.yaml`）；
- `speculative/joint.py`：联合视频-动作版的投机循环（8.3KB，未逐行取证）；
- 评测配置：LIBERO / LIBERO-Plus / RoboTwin（`configs/sim_*.yaml` + `scripts/eval_*.sh`）；
- 复用 Fast-WAM 数据处理（`processing/transforms/` 与 Fast-WAM 同名同结构）。

## 7. 待核实

- `speculative/joint.py` 的联合版验证逻辑（本次只取证 action-only 版）；
- 论文 §5 的 ε/w 敏感性数值与延迟分解（正文截断未取到）。
