# WAM 代码对质：6 篇论文动作空间/维度取证（06-code-forensics）

> 取证方法：GitHub API 列树（`/git/trees/main?recursive=1`）+ `raw.githubusercontent.com` 抓取 config/processor 文件；未知仓库先 web_search 定位官方仓库。
> 铁律遵守：每条结论只写代码事实并附 verbatim 关键行；无证据处一律标【未能获取/未找到】。
> 取证时间：2026-10-26。

---

## 1. DiT4DiT

- 仓库：https://github.com/Mondo-Robotics/DiT4DiT
- 取证文件：
  - `DiT4DiT/config/libero/dit4dit_libero.yaml`
  - `DiT4DiT/config/real_robot/dit4dit_g1.yaml`
  - `DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml`

【代码事实】LIBERO 配置 verbatim：

```yaml
  action_model:
    action_model_type: DiT-B
    hidden_size: 2560
    add_pos_embed: True
    max_seq_len: 1024
    action_dim: 8
    state_dim: 16
    future_action_window_size: 7
    action_horizon: 8
    past_action_window_size: 0
```

```yaml
    action_type: joint_position
```

```yaml
    max_state_dim: 16
    max_action_dim: 8
    video_delta_indices: [0,1,2,3,4,5,6,7,8]
    action_video_freq_ratio: 2  # pixel-space temporal downsampling: every Nth frame
```

G1 真机配置（`dit4dit_g1.yaml`）verbatim：

```yaml
    action_dim: 32
    state_dim: 64
    future_action_window_size: 15
    action_horizon: 16
```

RoboCasa（GR-1 机器人）配置（`dit4dit_robocasa_gr1.yaml`）同 G1：`action_dim: 32 / state_dim: 64 / action_horizon: 16`，另有：

```yaml
    data_mix: fourier_gr1_unified_1000 
    action_type: joint_position
    video_delta_indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    max_state_dim: 64
    max_action_dim: 32
    action_video_freq_ratio: 2
```

**结论**：动作空间 = 绝对关节位置（`action_type: joint_position`）；LIBERO 上 chunk=8 步、action_dim=8、state_dim=16；G1 真机与 RoboCasa GR-1 上 chunk=16 步、action_dim=32、state_dim=64。

---

## 2. τ0-WM（τ₀-World Model）

- 搜索过程：web_search「tau0-WM github official code repository」→ 官方仓库 `sii-research/tau-0-wm`（README 自称 "the official implementation of τ0-World Model: A Unified Video-Action World Model for Robotic Manipulation"）。
- 仓库：https://github.com/sii-research/tau-0-wm
- 取证文件：
  - `README.md`（"Action Space" 一节）
  - `configs/deployment/tau_pretrain_rela_eef6d.yaml`
  - `configs/tau_model/posttrain_taco_play_abs.yaml`
  - `configs/data/example_task/taco_abs_joint.yaml`
  - `configs/deployment/statistics.json`
  - `utils/action_space_utils.py`

【代码事实】README 动作空间定义 verbatim：

> *action* obtained from the server will be the **absolute** poses of end-of-effectors with shape {T, 16}.
> The order of output actions:
> - left end-effector (xyz + quaternion with order *xyzw*)
> - left gripper openness (ranging from 0 to 1, 0 for opening and 1 for close)
> - right end-effector
> - right gripper openness.
> In the pretraining stage, τ0-WM is optimized to predict the relative pose of end-effectors, including 20 channels (xyz and 6d-rotation for each arm). **The conversion between quaternion and 6d-rotation will be executed automatically.**

预训练部署配置 verbatim：

```yaml
# infer config
action_dim: 20
gripper_dim: 1
chunk: 9
action_chunk: 33

img_size: [192, 256]
action_type: "relative"
action_space: "eef6d"
norm_type: "meanstd"
```

```yaml
# diffusion
diffusion_model:
  config:
    action_dim: 1024
    action_ffn_dim: 2048
    action_in_dim: 20
    action_max_seq_len: 60
```

后训练配置（`posttrain_taco_play_abs.yaml`）verbatim：

```yaml
data:
  public_args:
    action_chunk: 33
    action_space: joint
    action_type: absolute
    chunk: 9
```

```yaml
diffusion_model:
  config:
    action_dim: 1024
    action_ffn_dim: 2048
    action_in_dim: 7
    action_max_seq_len: 60
```

任务数据配置（`taco_abs_joint.yaml`）verbatim：

```yaml
  action_type: absolute
  action_space: joint
  action_index: [0, 1, 2, 3, 4, 5]
  action_gripper_index: [6]
```

**结论**：Action DiT 内部隐层 `action_dim: 1024`，输入动作维度 `action_in_dim`：预训练（相对 EEF，eef6d）**20**（双臂 xyz+6d-rotation 18 + 2 夹爪）、后训练（绝对关节）**7**（6 关节 + 1 夹爪，`action_index: [0..5] + action_gripper_index: [6]`）；序列上限 `action_max_seq_len: 60`；动作块 `action_chunk: 33`、视频 `chunk: 9`。
【未能获取】代码未使用论文符号 H_a/H_v，33 与 9 只以 `action_chunk`/`chunk` 键名出现，与 H_a/H_v 的符号对应关系在仓库中未写明（仅数值事实）。

---

## 3. UWM（Unified World Models）

- 搜索过程：web_search「Unified World Models github」→ 官方仓库 `WEIRDLabUW/unified-world-model`（README："Unfied World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets"）。
- 仓库：https://github.com/weirdlabuw/unified-world-model
- 取证文件：
  - `configs/model/uwm.yaml`
  - `configs/dataset/template_robomimic.yaml`
  - `configs/dataset/droid.yaml`
  - `configs/template_train.yaml`
  - `models/uwm/uwm.py`

【代码事实】模型配置 verbatim：

```yaml
_target_: models.uwm.UnifiedWorldModel
action_len: 16
action_dim: ${dataset.shape_meta.action.shape[0]}
```

```yaml
latent_patch_shape: [2, 4, 4]
```

RoboMimic shape_meta（`template_robomimic.yaml`）verbatim：

```yaml
  action:
    shape: [7] # pos + quarternion + gripper
```

DROID shape_meta（`droid.yaml`）verbatim：

```yaml
  action:
    # 3(pos) + 6(rot) + 1(gripper)
    shape: [10]
```

```yaml
normalize_action: True
normalize_lowdim: True
```

训练模板（`template_train.yaml`）verbatim：

```yaml
num_frames: 19
obs_num_frames: 2
```

时间步切换实现（`models/uwm/uwm.py`）verbatim：

```python
class DualTimestepEncoder(nn.Module):
    ...
    def forward(self, t1, t2):
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        temb = torch.cat([temb1, temb2], dim=-1)
        return self.proj(temb)
```

训练时独立采样两个时间步：

```python
        # Sample diffusion timestep for action
        action_t = torch.randint(
            low=0, high=self.num_train_steps, size=(batch_size,), device=device
        ).long()
        # Sample diffusion timestep for next observation
        next_obs_t = torch.randint(
            low=0, high=self.num_train_steps, size=(batch_size,), device=device
        ).long()
```

推理时切换（另一模态时间步固定为常数 / 联合共用同一 t）：

```python
    def sample_forward_dynamics(self, obs_dict, action):
        ...
        action_t = self.noise_scheduler.timesteps[-1]      # action 时间步固定
```

```python
    def sample_marginal_next_obs(self, obs_dict):
        ...
        action_t = self.noise_scheduler.timesteps[0]       # action 时间步固定
```

```python
    def sample_marginal_action(self, obs_dict):
        ...
        next_obs_t = self.noise_scheduler.timesteps[0]     # 视频时间步固定
```

```python
    def sample_joint(self, obs_dict):
        ...
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, next_obs_noise_pred = self.noise_pred_net(
                obs_feat, action_sample, t, next_obs_sample, t
            )                                                # 同一 t
```

**结论**：动作 chunk `action_len: 16`；动作维度取自数据集 shape_meta —— RoboMimic **7**（pos+quaternion+gripper）、DROID **10**（3 pos + 6 rot + 1 gripper）；「时间步切换」= 双时间步编码器 `DualTimestepEncoder`（视频/动作各自 t）+ 边缘采样时将另一模态 t 固定为 `timesteps[0]` 或 `timesteps[-1]`、联合采样共用同一 t。

---

## 4. DreamZero

- 仓库：https://github.com/dreamzero0/dreamzero（直接可打开，列树成功、无截断）
- 取证文件：
  - `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
  - `groot/vla/configs/data/dreamzero/droid_relative.yaml`
  - `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml`
  - `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain.yaml`
  - `groot/vla/configs/model/dreamzero/vla.yaml`

【代码事实】48 步 chunk 主配置（`base_48_...`）verbatim：

```yaml
num_frames: 49
action_horizon: 48
state_horizon: 1
```

动作模态键（DROID，同文件）verbatim：

```yaml
  action:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    modality_keys:
      - action.joint_position
      - action.gripper_position
```

各本体动作维度（同文件注释 verbatim）：

```yaml
# agibot (AGIbot: state 32, action 22, 3 views)
```

```yaml
# yam (YAM: joint+gripper only from Dataset/YAM_play_data/meta/modality.json;
#       state 14 dims, action 14 dims, 3 views)
```

动作头配置（`wan_flow_matching_action_tf.yaml`）verbatim：

```yaml
num_frame_per_block: 1
num_action_per_block: 32
num_state_per_block: 1
frame_seqlen: 880
```

```yaml
    action_dim: ${max_action_dim}
    action_horizon: ${action_horizon}
```

模型变换配置（`dreamzero_cotrain.yaml`）verbatim：

```yaml
num_visual_tokens_per_frame: 16
max_state_dim: 44
max_action_dim: 32
```

DROID 数据配置（`droid_relative.yaml`）verbatim：

```yaml
max_state_dim: 64
relative_action: true
relative_action_per_horizon: false
relative_action_keys:
  - joint_position
max_chunk_size: 5
```

**结论**：48 步动作 chunk（`action_horizon: 48`、`num_frames: 49`）；动作空间 = 关节位置（`action.joint_position`）+ 夹爪位置（`action.gripper_position`），DROID 上为**相对**关节动作（`relative_action: true, relative_action_keys: [joint_position]`）；网络动作维按 `max_action_dim: 32` 对齐（pad），本体真实维度：DROID 7、AGIBOT 22（state 32）、YAM 14。

---

## 5. Motus

- 搜索过程：web_search「Motus github official code」→ 官方仓库 `thu-ml/Motus`（"Official code of Motus: A Unified Latent Action World Model"）。
- 仓库：https://github.com/thu-ml/Motus
- 取证文件：
  - `configs/latent_action.yaml`
  - `configs/ac_one.yaml`
  - `inference/robotwin/Motus/utils/robotwin.yml`
  - `inference/robotwin/Motus/deploy_policy.py`
  - `data/utils/stat.json`
  - `DATA_FORMAT.md`
  - `data/robotwin2/robotwin_data_convert/README.md`

【代码事实】潜动作配置（`latent_action.yaml`）verbatim：

```yaml
common:
  # Latent Action dimensions
  action_dim: 14
  state_dim: 14
```

真机微调配置（`ac_one.yaml`）verbatim：

```yaml
common:
  # Robot dimensions
  action_dim: 14         # Robot action dimension
  state_dim: 14          # Robot state dimension
```

RoboTwin 部署配置（`robotwin.yml`）verbatim：

```yaml
common:
  # Robot dimensions
  action_dim: 14         # Robot action dimension
  state_dim: 14          # Robot state dimension
```

RoboTwin 部署执行接口（`deploy_policy.py`）verbatim：

```python
        actions_real = predicted_actions.squeeze(0).cpu().numpy()
```

```python
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')
```

动作归一化（`deploy_policy.py`）verbatim：

```python
    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]."""
        ...
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
```

各本体统计（`data/utils/stat.json`）verbatim：

```json
    "robotwin2": {
        "min": [..., 0.0, ..., 0.0],
        "max": [..., 1.0, ..., 1.0],
        "action_dim": 14,
```

（`aloha_agilex_1` / `aloha_agilex_2` / `ac_one` 三段同样 `"action_dim": 14`。）

RoboTwin 数据格式（`data/robotwin2/robotwin_data_convert/README.md`）verbatim：

```text
│       ├── qpos/0.pt       # Robot positions (T, 14)
```

**结论**：14 维潜动作（`latent_action.yaml: action_dim: 14`）对应的真实执行动作空间 = **14 维机器人关节位置（qpos）**：RoboTwin2.0 数据为 `(T, 14)` 的 qpos；AC-One、Aloha-Agilex-1/2、RoboTwin2 四个真机/仿真统计文件 `action_dim` 均为 14；执行接口为 `take_action(action, action_type='qpos')`；动作以 stat.json 的 min/max 线性归一化到 [0,1]。robotwin2 统计中第 7、14 维取值范围为 [0,1]（夹爪），其余 12 维为关节角范围——每臂「6 关节 + 1 夹爪」的拆分由该数据范围给出，代码未逐维命名。

---

## 6. GR-1

- 仓库：https://github.com/bytedance/GR-1
- 取证文件：
  - `logs/configs.json`
  - `models/gr1.py`
  - `evaluation/calvin_env_wrapper_raw.py`
  - `evaluation/calvin_evaluation.py`
  - `README.md`

【代码事实】训练配置（`logs/configs.json`）verbatim：

```json
{"embed_dim": 384, ..., "seq_len": 10, "action_mode": "ee_rel_pose", "act_dim": 7, "state_dim": 7, "use_hand_rgb": true, ..., "act_pred": true, "fwd_pred": true, "fwd_pred_hand": true, "fwd_pred_next_n": 3, ...}
```

模型动作头（`models/gr1.py`）verbatim：

```python
        self.embed_arm_state = torch.nn.Linear(self.state_dim - 1, hidden_size)
        self.embed_gripper_state = torch.nn.Linear(2, hidden_size) # one-hot gripper state
```

```python
        self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1) # arm action
        self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)
```

环境接口（`evaluation/calvin_env_wrapper_raw.py`）verbatim：

```python
        if self.relative_actions:
            action = action_tensor.squeeze().cpu().detach().numpy()
            assert len(action) == 7
```

```python
        action[-1] = 1 if action[-1] > 0 else -1
```

评测端输出处理（`evaluation/calvin_evaluation.py`）verbatim：

```python
        arm_state = state[:6]
        gripper_state = state[-1]
```

```python
        gripper_action_pred = torch.nn.Sigmoid()(gripper_action_pred)
        gripper_action_pred = gripper_action_pred > 0.5
        gripper_action_pred = gripper_action_pred.int().float()
        gripper_action_pred = gripper_action_pred * 2.0 - 1.0
        action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=0)  # (act_dim,)
```

**结论**：动作空间 = 7 维相对末端位姿（`action_mode: ee_rel_pose, act_dim: 7`）：前 6 维臂（`pred_arm_act = act_dim-1`）+ 1 维夹爪（`pred_gripper_act = 1`，sigmoid>0.5 二值化为 ±1）。
【未能获取】① 动作数值归一化：评测代码直接输出归一化空间的 7 维动作（无任何 denorm/mean-std 步骤），而训练数据归一化的 stats 表与数据处理代码未随仓库发布（仓库仅含 CALVIN 评测代码，README 引用外部 CALVIN 仓库数据）→ 归一化数值【未找到】。② 视频预测间隔 Δt：`fwd_pred_next_n: 3`（预测后续 3 帧）与 `seq_len: 10` 为代码事实，但帧间时间间隔（Δt）在仓库中无任何定义（数据采样代码未发布）→ Δt【未找到】。

---

## 7. （补充）Fast-WAM — RoboTwin 每臂 7 维拆分

- 仓库：https://github.com/Yongwei-Zhang/FastWAM（"Official codebase for Fast-WAM"）
- 取证文件：
  - `configs/data/robotwin.yaml`
  - `src/fastwam/datasets/lerobot/processors/fastwam_processor.py`
  - `src/fastwam/datasets/lerobot/transforms/relative_action.py`
  - `experiments/robotwin/fastwam_policy/deploy_policy.py`

【代码事实】RoboTwin 数据配置（`robotwin.yaml`）verbatim：

```yaml
    action:
      - key: default
        raw_shape: 14 # before transform
        shape: 14 # after transform
    state:
      - key: default
        raw_shape: 14 # before transform
        shape: 14 # after transform
  num_frames: 33
  global_sample_stride: 1
  action_video_freq_ratio: 4 # 32 action, 9 video frames
```

```yaml
    action_output_dim: 14
    proprio_output_dim: 14

    # action & state normalization
    action_state_transforms: null
```

处理器注释（`fastwam_processor.py`）verbatim：

```python
            # sample["action"][sample["action_is_pad"], :-1] = 0.0 # NOTE: we assume use delta_eef_pose + gripper， so pad action is 0
```

位姿变换头注释（`relative_action.py`）verbatim：

```python
# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r
```

RoboTwin 部署（`deploy_policy.py`）verbatim：

```python
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
```

```python
        task_env.take_action(action, action_type="qpos")
```

**结论**：Fast-WAM 在 RoboTwin 上动作 = **14 维原始关节位置（qpos）**（`raw_shape/shape: 14`、`action_output_dim: 14`、`action_state_transforms: null`、执行 `take_action(action, action_type="qpos")`），因此 RoboTwin 每臂 7 维**不经过 xyz+quat 或 xyz+rpy 的 EEF 位姿拆分**（无 EEF 转换）。仓库中 EEF 位姿相关代码（`RelativePoseTransform`，用于其他 embodiment）的姿态表示为**四元数 `x, y, z, i, j, k, r`（quaternion）**，而非 rpy；「7 维 ΔEE+夹爪」假设仅见于 processor 注释，属 libero 等 embodiment。

---

## 汇总表

| 论文 | 仓库 | 动作空间 | 维度 | chunk/序列 | 归一化 | 证据文件 |
|---|---|---|---|---|---|---|
| DiT4DiT | github.com/Mondo-Robotics/DiT4DiT | 绝对关节位置 `action_type: joint_position` | LIBERO 8；G1/RoboCasa GR-1 32（state 16/64） | LIBERO 8 步；G1/RoboCasa 16 步（`action_horizon`） | config 未列 stats | config/libero/dit4dit_libero.yaml; config/real_robot/dit4dit_g1.yaml; config/robocasa/dit4dit_robocasa_gr1.yaml |
| τ0-WM | github.com/sii-research/tau-0-wm | 预训练：相对 EEF 6d（eef6d，20 通道=双臂 xyz+6d+2 夹爪）；后训练：绝对关节 7（6+1） | Action DiT `action_dim:1024`、`action_in_dim:20/7`、`action_max_seq_len:60` | `action_chunk:33`、视频 `chunk:9`（H_a/H_v 符号未在代码出现） | meanstd（`norm_type: "meanstd"`） | README.md; configs/deployment/tau_pretrain_rela_eef6d.yaml; configs/tau_model/posttrain_taco_play_abs.yaml; configs/data/example_task/taco_abs_joint.yaml |
| UWM | github.com/weirdlabuw/unified-world-model | RoboMimic 7（pos+quat+gripper）；DROID 10（3pos+6rot+1gripper） | 由 shape_meta：7 / 10 | `action_len:16`；`num_frames:19, obs_num_frames:2` | `normalize_action: True`（DROID config） | configs/model/uwm.yaml; configs/dataset/template_robomimic.yaml; configs/dataset/droid.yaml; models/uwm/uwm.py |
| DreamZero | github.com/dreamzero0/dreamzero | 关节位置+夹爪（joint_position+gripper_position，DROID 相对动作） | 网络 32（max_action_dim pad）；本体 DROID 7 / AGIBOT 22 / YAM 14 | **48 步**（`action_horizon:48`、`num_frames:49`） | q99（per-key normalization_modes） | configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml; configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml; configs/model/dreamzero/transform/dreamzero_cotrain.yaml; configs/data/dreamzero/droid_relative.yaml |
| Motus | github.com/thu-ml/Motus | 潜动作 14；真实执行=关节位置（qpos, `take_action(action_type='qpos')`） | 潜动作 14；执行动作 14（AC-One/Aloha/RoboTwin2 均 14，每臂 6 关节+1 夹爪由 stat 范围给出） | `num_video_frames:8`；`global_downsample_rate`/`video_action_freq_ratio` 见各 config | min/max → [0,1]（stat.json） | configs/latent_action.yaml; configs/ac_one.yaml; inference/robotwin/Motus/deploy_policy.py; data/utils/stat.json; data/robotwin2/robotwin_data_convert/README.md |
| GR-1 | github.com/bytedance/GR-1 | 相对末端位姿（`ee_rel_pose`）7=6 臂+1 夹爪（±1 二值） | act_dim 7（state_dim 7） | `seq_len:10`；视频预测 `fwd_pred_next_n:3` | 归一化 stats 未发布【未能获取】；Δt【未能获取】 | logs/configs.json; models/gr1.py; evaluation/calvin_env_wrapper_raw.py; evaluation/calvin_evaluation.py |
| Fast-WAM（补充） | github.com/Yongwei-Zhang/FastWAM | RoboTwin：14 维 qpos（无 EEF 拆分）；EEF 位姿代码用四元数 ijkr（非 rpy） | 14（action_output_dim:14） | `num_frames:33`，`action_video_freq_ratio:4 # 32 action, 9 video frames` | z-score（norm_default_mode）+ dataset_stats.json | configs/data/robotwin.yaml; src/fastwam/datasets/lerobot/processors/fastwam_processor.py; src/fastwam/datasets/lerobot/transforms/relative_action.py; experiments/robotwin/fastwam_policy/deploy_policy.py |

更新时间: 2026-10-26 23:45
