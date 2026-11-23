# 12 · 实现手册：公式 ↔ 代码、模块与输入输出

> 这一章把**论文公式和实现代码绑在一起**：每条公式先原样列出，再给一张「公式项 ↔ 代码 ↔ 说明」的映射表，然后是完整伪代码、模块表和输入输出。
> 公式不是拿来推的，是拿来**对照代码**的：每个符号都能在下面的代码里找到对应行。
> 每个方法固定：**① 流程**（S1…Sn，纯动作）→ **② 模块表**（输入/输出/作用）→ **③ 公式 + 公式项↔代码 + 主伪代码**（`[S1]…[Sn]` 对齐流程）→ **④ 子模块** → **⑤ 训练与推理怎么调** → **⑥ 实现注意**。
> 这是**示意性伪代码，不是可运行脚本**；`✓` = 原文核实的数值/公式，`?` = 原文未给或此处为示意接口。
> 形状记号：`B` batch，`V` 视角，`H` chunk 长度，`D` 动作维，`K` 采样步数，`K_T` 教师步数，`C` 特征维，`N` token 数，`T` 文本长度。

---

## 1. Orion-Lite（2604.08266，驾驶）

### ① 流程

- **S1** 冻结教师（7B）跑一次，取出 latent planning token。
- **S2** 学生视觉塔（冻结）编码图像。
- **S3** 0.1B、6 层 decoder 用 learnable query 读视觉 token，输出学生 token。
- **S4** 学生 token 与教师 token 做逐元素 L1；同时 planner 出轨迹，算碰撞/边界/回归/VAE 四项 GT 损失。
- **S5** 只更新 decoder 与 planner；视觉塔、QT-Former、教师全冻结。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `teacher.planning_token` | `images [B,V,3,224,224]`, `text [B,T]` | `T_p [B,1,C_p]` | 冻结 7B，只产 token，不参与训练 |
| `student.vision` | `images` | `vis [B,Nv,1024]` | 冻结视觉塔 |
| `StudentDecoder` | `vis [B,Nv,1024]` | `T_s [B,1,C_p]` | 6 层 cross-attn，query 读视觉 |
| `student.planner` | `T_s`, `vis` | `traj [B,H,3]` | 出未来轨迹 |
| `traj_losses` | `pred [B,H,3]`, `gt [B,H,3]` | `scalar` | 碰撞/边界/回归/VAE |

### ③ 主伪代码

**公式（原文/重构）**

```text
L_mimic = (1/(B·C_p)) · Σ_b Σ_c | T_s[b,c] − T_p[b,c] |
L_GT    = L_col + λ_bd·L_bd + λ_reg·L_reg + λ_vae·L_vae,   λ_bd=λ_reg=λ_vae=3
L_total = L_GT + L_mimic
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `1/(B·C_p) Σ` | `.abs().mean()` | 对 batch 与通道同时平均 |
| `|T_s − T_p|` | `(T_s - T_p).abs()` | L1，不是平方 |
| `T_p` 不反传 | `T_p.detach()` / `no_grad` | 教师目标 |
| `λ=3` 三项 | `3.0 * L_bd` 等 | 原文权重 |
| `L_total` | `loss = L_GT + L_mimic` | 直接相加 |


```python
def orion_lite_step(batch, teacher, student, cfg):
    # 输入:
    #   batch.images [B,V,3,224,224] float32
    #   batch.text   [B,T]           int64
    #   batch.traj   [B,H,3]         float32   (x, y, heading)
    # 输出:
    #   loss  : scalar
    #   stats : dict  {"L_mimic": float, "L_gt": float}
    with torch.no_grad():                                      # [S1] 教师只前向
        T_p = teacher.planning_token(batch.images, batch.text)  # [B,1,C_p]

    vis = student.vision(batch.images)                         # [S2] 冻结视觉塔
    T_s = student.decoder(vis)                                 # [S3] [B,1,C_p]

    L_mimic = (T_s - T_p).abs().mean()                         # [S4] L1 对齐 ✓
    pred    = student.planner(T_s, vis)                        # [B,H,3]
    L_gt    = traj_losses(pred, batch.traj)                    # [S4] 四项 GT ✓

    loss = L_gt + L_mimic                                      # [S5] ✓
    return loss, {"L_mimic": L_mimic.item(), "L_gt": L_gt.item()}


def traj_losses(pred, gt):
    # 输入: pred [B,H,3], gt [B,H,3]
    # 输出: scalar = L_col + 3*L_bd + 3*L_reg + 3*L_vae   ✓ λ=3
    return (collision_loss(pred, gt)
            + 3.0 * boundary_loss(pred, gt)
            + 3.0 * regression_loss(pred, gt)
            + 3.0 * vae_loss(pred, gt))
```

### ④ 子模块：StudentDecoder

```python
class StudentDecoder(nn.Module):
    # 输入: vis [B,Nv,1024]
    # 输出: T_s [B,1,C_p]
    def __init__(self, c_in=1024, c_p=256, n_layer=6):         # ✓ 6 层
        super().__init__()
        self.inp    = nn.Linear(c_in, c_p)                     # input projection
        self.query  = nn.Parameter(torch.randn(1, 1, c_p))     # learnable query
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(c_p, 8, batch_first=True)
            for _ in range(n_layer)])
        self.out    = nn.Linear(c_p, c_p)                      # output projection

    def forward(self, vis):
        x = self.inp(vis)                                      # [B,Nv,C_p]
        q = self.query.expand(x.size(0), -1, -1)               # [B,1,C_p]
        for blk in self.blocks:
            q = blk(q, x)                                      # query 读视觉
        return self.out(q)                                     # [B,1,C_p]
```

### ⑤ 训练与推理

```python
optimizer = torch.optim.AdamW(
    list(student.decoder.parameters()) + list(student.planner.parameters()),
    lr=cfg.lr)                                   # 只放这两组参数 ✓

for batch in loader:
    loss, stats = orion_lite_step(batch, teacher, student, cfg)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

@torch.no_grad()
def infer(images):
    # 输入: images [B,V,3,224,224]；输出: traj [B,H,3]
    vis = student.vision(images)
    return student.planner(student.decoder(vis), vis)
```

### ⑥ 实现注意

1. `T_p` 必须是 planning query 那一个 token，取 `last_hidden.mean` 会丢信息。
2. 教师 token 缓存起来，别每个 step 都跑 7B；缓存 key = 样本 id。
3. 训练时学生仍可拿到 `text` 的 embedding 吗？**不行**——学生是纯视觉，text 只用于生成 `T_p`。

---

## 2. THAW-VLA（2609.24682，操作）

### ① 流程

- **S1** 冻结 Cosmos 3 理解塔，取 layer 24，按视图 mean-pool 成 4096 维特征。
- **S2** 对全训练集跑一次，特征写入 memmap，key=(traj_id, frame_idx)。
- **S3** 学生（QwenGR00T）前向，取图像 token 的池化特征。
- **S4** 投影后与教师特征做 cosine 对齐（λ=0.5），再加动作 flow-matching 损失。
- **S5** 部署时丢掉投影器与缓存，学生与未蒸馏基线完全一致。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `cosmos3.understanding` | `img_v [B,3,H,W]` | `f [B,4096]` | 冻结教师，layer 24 + mean-pool |
| `build_cache` | `loader`, `teacher` | `.npy` 文件 | 一次性离线提特征 |
| `CachedDataset` | `index` | `(item, f_T)` | 训练时读缓存，不加载教师 |
| `qwengr00t` | `obs`, `lang` | `h`（含 image tokens） | 0.8B 学生 |
| `p_phi` | `f_S [B,V,4096]` | 同形 | 可训投影，训完丢弃 |
| `flow_matching` | `h`, `actions` | `scalar` | 动作专家损失 |

### ③ 主伪代码

**公式（原文/重构）**

```text
L_align = 1 − cos( p_φ(f_c^S(o_t)),  sg[f_c^T(o_t)] )
L       = L_act + λ_align · L_align,   λ_align = 0.5
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `cos(·,·)` | `F.cosine_similarity(..., dim=-1)` | 最后一维是特征维 |
| `1 − cos` | `1 - cos` | 距离化 |
| `sg[f_T]` | `f_T`（来自缓存） | 缓存天然无梯度 |
| `λ=0.5` | `0.5 * L_align` | 原文 |
| `L_act` | `flow_matching(h, act)` | 主任务损失 |


```python
def thaw_step(batch, student, p_phi, cfg):
    # 输入:
    #   batch.obs   [B,V,3,H,W]
    #   batch.lang  [B,T]
    #   batch.act   [B,H,D]
    #   batch.keys  list[(traj_id, frame_idx)]
    # 输出:
    #   loss  : scalar = L_act + 0.5 * L_align
    #   stats : dict
    h   = student(batch.obs, batch.lang)                 # [S3] QwenGR00T 前向
    f_S = h.image_tokens.mean(dim=1)                     # [B,V,4096]

    f_T = load_cache(batch.keys)                          # [S2] [B,V,4096] fp16→fp32

    L_align = (1 - F.cosine_similarity(p_phi(f_S), f_T, dim=-1)).mean()   # [S4] ✓
    L_act   = flow_matching(h, batch.act)                # 动作专家
    loss    = L_act + 0.5 * L_align                      # ✓ λ=0.5
    return loss, {"L_align": L_align.item(), "L_act": L_act.item()}
```

### ④ 子模块：缓存构建、读取与部署校验

```python
@torch.no_grad()
def build_cache(loader, teacher, out_dir):
    # 输入: loader（按 key 可寻址）, teacher（冻结）, out_dir
    # 输出: 每个样本一个 .npy，形状 [V,4096]，fp16
    for batch in loader:
        per_view = []
        for v in range(batch.images.size(1)):            # 逐视角
            h = teacher.understanding(batch.images[:, v], layer=24)   # ✓ layer 24
            per_view.append(h.mean(dim=1))               # ✓ 每视图 mean-pool
        f_T = torch.stack(per_view, dim=1)               # [B,V,4096]
        for i, (traj_id, idx) in enumerate(batch.keys):
            np.save(f"{out_dir}/{traj_id}_{idx}.npy",
                    f_T[i].half().cpu().numpy())

def load_cache(keys, out_dir, device="cuda"):
    # 输入: keys list[(traj_id, frame_idx)], out_dir
    # 输出: f_T [B,V,4096] float32
    arr = [torch.from_numpy(np.load(f"{out_dir}/{k[0]}_{k[1]}.npy")).float()
           for k in keys]
    return torch.stack(arr).to(device)


def deploy_check(student, p_phi, obs):
    # 输入: 学生、投影器、一个 batch
    # 输出: bool（部署前后是否逐位一致）
    with torch.no_grad():
        a_full = student(obs)                            # 带投影器
        a_drop = student_without(p_phi, obs)             # 丢投影器
    return torch.equal(a_full, a_drop)                   # ✓ 构造上应完全一致
```

### ⑤ 训练与推理

```python
for batch in loader:
    loss, _ = thaw_step(batch, student, p_phi, cfg)
    loss.backward(); optimizer.step()

# 推理：一次 backbone prefill + 4 步 flow
@torch.no_grad()
def infer(obs, lang):
    h = student(obs, lang)
    return student.action_expert.sample(h, steps=4)
```

### ⑥ 实现注意

1. 缓存 key 用 `(traj_id, frame_idx)`，别用 batch 内下标——shuffle 后会错位。
2. `p_φ` 不保存进推理 checkpoint；保存时加白名单。
3. 学生偏离教师分布后缓存特征失效，所以 λ 只取 0.5，且只做视图级池化对齐。

---

## 3. CKT-WAM（2605.06247，WAM）

### ① 流程

- **S1** 教师 WAM（14B）**单次前向**，时间步固定 0，取第 20 层隐状态。
- **S2** 过一个 512 维 bottleneck。
- **S3** 两组 learnable query（各 32 个）做 cross-attention，压成 32+32 个 token。
- **S4** 常开 general adapter + 8 个 specialized adapter（top-2 路由）生成条件 token。
- **S5** 条件 token 拼到学生文本条件序列尾部（加 separator），cross-attn 注入学生 DiT。
- **S6** 教师、学生都冻结，只训 CKT 模块；目标是学生自己的视频+动作 flow-matching。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `teacher.encode` | `obs`, `text` | `H_t [B,N,D_t]` | 冻结 14B，t=0，layer 20 |
| `Bottleneck` | `H_t` | `[B,N,512]` | 降维 |
| `LQCA` | `queries [B,32,512]`, `h` | `C [B,32,512]` | 查询压缩 |
| `AdapterRouter` | `h.mean(1) [B,512]` | `logits [B,8]` | 选 2 个专家 |
| `ConditionBuilder` | `text_tokens`, `C_g`, `C_s` | `cond [B,T+1+64,D_s]` | 拼条件序列 |
| `student_dit` | `noisy_latent`, `t`, `cond` | `v [B,N*H,D]` | 冻结，只做前向 |

### ③ 主伪代码

**公式（原文/重构）**

```text
L_bal = λ_bal · Σ_e ( p_e · E − 1 )²,   λ_bal = 0.01, E = 8
L     = L_flow(video+action) + L_bal
```

> 注：`L_bal` 为 MoE 标准负载均衡形式的重构；论文只给 `λ_bal=0.01` 与 top-2 路由（原文未给逐项公式）。

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `p_e` | `logits.softmax(-1).mean(0)` | 专家平均使用率 |
| `(p_e·E − 1)²` | `((... * 8 - 1) ** 2)` | 均匀时=0 |
| `λ_bal` | `0.01 *` | 原文 |
| `L_flow` | `flow_matching(v, target)` | 学生自己的视频+动作目标 |


```python
def ckt_step(batch, teacher, ckt, student_dit, cfg):
    # 输入:
    #   batch.obs        [B,V,3,H,W]
    #   batch.text       [B,T]
    #   batch.latent     [B,Nz,Cl]   (视频+动作 latent)
    # 输出:
    #   loss  : scalar
    #   stats : dict {"L_fm", "L_bal"}
    with torch.no_grad():                                    # [S1]
        H_t = teacher.encode(batch.obs, batch.text, t=0, layer=20)  # [B,N,D_t]

    h  = ckt.bottleneck(H_t)                                 # [S2] [B,N,512]
    Cg = ckt.lqca_g(ckt.q_g.expand(B, -1, -1), h, h)         # [S3] [B,32,512]
    Cs = ckt.lqca_s(ckt.q_s.expand(B, -1, -1), h, h)         # [S3] [B,32,512]

    logits = ckt.router(h.mean(1))                           # [S4] [B,8]
    w, idx = logits.topk(2, dim=-1); w = w.softmax(-1)       # ✓ top-2
    Cs = sum(ckt.adapters[j](Cs) * w[:, k] for k, j in enumerate(idx.T))
    Cg = ckt.adapter_general(Cg)

    cond = torch.cat([batch.text_tokens, ckt.sep, Cg, Cs], dim=1)   # [S5]
    t    = torch.rand(B, device=obs.device)
    noisy = add_noise(batch.latent, t)
    v    = student_dit(noisy, t, cond)                       # 冻结，只前向
    L_fm = F.mse_loss(v, target_velocity(batch.latent))      # [S6]
    L_bal = 0.01 * ((logits.softmax(-1).mean(0) * 8 - 1) ** 2).mean()  # ✓
    loss = L_fm + L_bal
    return loss, {"L_fm": L_fm.item(), "L_bal": L_bal.item()}
```

### ④ 子模块：LQCA 与路由

```python
class LQCA(nn.Module):
    # 输入: q [B,32,512], kv [B,N,512]
    # 输出: C [B,32,512]
    def __init__(self, d=512, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
    def forward(self, q, kv):
        out, _ = self.attn(q, kv, kv)                        # query 读教师 token
        return self.norm(out + q)                            # 残差


def route_top2(logits, adapters, Cs):
    # 输入: logits [B,8], adapters: 8 个, Cs [B,32,512]
    # 输出: [B,32,512]（按 top-2 权重混合）
    w, idx = logits.topk(2, dim=-1); w = w.softmax(-1)
    outs = [adapters[j](Cs) for j in range(8)]
    return sum(outs[j] * w[:, k:k+1, None] for k, j in enumerate(idx.T))
```

### ⑤ 训练与推理

```python
ckt_params = [p for n, p in ckt.named_parameters()]
# 教师与学生 DiT 不进 optimizer ✓

for batch in loader:
    loss, _ = ckt_step(batch, teacher, ckt, student_dit, cfg)
    loss.backward(); optimizer.step()

@torch.no_grad()
def infer(obs, text):
    H_t = teacher.encode(obs, text, t=0, layer=20)   # 仍需要教师（训练期方案）
    cond = build_cond(ckt, H_t, text)
    return student_dit.sample(cond)                  # 部署时只保留学生
```

### ⑥ 实现注意

1. 教师 14B 训练时常驻，显存约 28GB(bf16)+学生+激活；用 `no_grad` + 分层 offload。
2. 条件 token 拼在**文本序列尾部**而不是 cross-attn 的 KV 单独一路——这是"不动去噪骨干"的关键。
3. `sep` 必须是可学习的，否则 token 边界不稳定。

---

## 4. S-VAM（2603.16195，WAM 自蒸馏）

### ① 流程

- **S1** 取去噪网络第一去噪步的上采样层多尺度特征，插值对齐后 concat。
- **S2** 用模型自己的多步去噪生成未来视频（不生成真值未来帧）。
- **S3** 冻结 VFM（DPAv3 几何 / DINOv2 语义）提特征，作为教师目标。
- **S4** 轻量 decoupler 把一步特征映射到 VFM 特征空间，平方误差。
- **S5** Uni-Perceiver 融合三路 token，条件扩散出动作。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `denoiser.upsample_features` | `obs`, `step=0` | 多尺度 `list[[B,N,C_i]]` | 一步特征 |
| `svd_sample` | `obs`, `K` | `V_hat [B,F,3,H,W]` | 自身多步生成 |
| `dpav3` / `dinov2` | `interp(V_hat)` | `Y_geo/Y_sem [B,N,C]` | 冻结 VFM 目标 |
| `Decoupler` | `[F, Y_ref]` | `F_tilde [B,N,C]` | 时空 Transformer |
| `uni_perceiver` | 三路 token | `tokens` | 压缩融合 |
| `diffusion_policy` | `tokens` | `actions [B,H,D]` | 出动作 |

### ③ 主伪代码

**公式（原文/重构）**

```text
L_i = || F̃_i^K − Y_i ||²,   i ∈ {geo, sem}
L   = L_geo + L_sem + L_action
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `F̃_i` | `decoup.geo(...)` / `decoup.sem(...)` | decoupler 输出 |
| `Y_i` | `vfm.dpav3(...)` / `vfm.dinov2(...)` | 冻结 VFM 目标 |
| `||·||²` | `((F - Y) ** 2).mean()` | 逐元素平方 |
| `L_action` | `flow_loss(policy(tokens), act)` | 动作专家 |


```python
def svam_step(batch, denoiser, decoup, vfm, policy):
    # 输入:
    #   batch.obs   [B,V,3,H,W]
    #   batch.act   [B,H,D]
    # 输出:
    #   loss  : scalar = L_geo + L_sem + L_action
    #   stats : dict
    F = concat_interp(denoiser.upsample_features(batch.obs, step=0))   # [S1] [B,N,C_F]
    with torch.no_grad():
        V_hat = svd_sample(denoiser, batch.obs, steps=K)                # [S2]
        Y_geo = vfm.dpav3(interp_to(V_hat, N))                          # [S3]
        Y_sem = vfm.dinov2(interp_to(V_hat, N))                          # [S3]

    F_geo = decoup.geo(torch.cat([F, Y_geo_ref], -1))                   # [S4]
    F_sem = decoup.sem(torch.cat([F, Y_sem_ref], -1))
    L_geo = ((F_geo - Y_geo) ** 2).mean()
    L_sem = ((F_sem - Y_sem) ** 2).mean()

    tokens = uni_perceiver([F, F_geo, F_sem])                           # [S5]
    L_act  = flow_loss(policy(tokens), batch.act)
    return L_geo + L_sem + L_act, {...}
```

### ④ 子模块：Decoupler

```python
class Decoupler(nn.Module):
    # 输入: F [B,N,C_F], Y_ref [B,N,C_Y] → concat 后 [B,N,C_F+C_Y]
    # 输出: F_tilde [B,N,C_Y]
    def __init__(self, c_in, c_out, heads=8):
        super().__init__()
        self.tf  = nn.TransformerEncoderLayer(c_in, heads, batch_first=True)
        self.proj = nn.Linear(c_in, c_out)
    def forward(self, F, Y_ref):
        x = torch.cat([F, Y_ref], dim=-1)
        return self.proj(self.tf(x))
```

### ⑤ 训练与推理

```python
# 自蒸馏阶段：只训 decoupler（50k steps, 单卡 H100 ✓）
for batch in loader:
    loss, _ = svam_step(batch, denoiser, decoup, vfm, policy)
    loss.backward(); optimizer.step()

# 推理：单次前向（RTX 3090 ✓）
tokens = uni_perceiver([F, decoup.geo(F, Y_geo_ref), decoup.sem(F, Y_sem_ref)])
actions = policy.sample(tokens)
```

### ⑥ 实现注意

1. 目标必须来自**同一条扩散轨迹**的多步生成；真值未来帧会引入轨迹错位（原文明确）。
2. `Y_ref` 是当前观测的 VFM 特征，不是目标的——别搞混。
3. decoupler 要两支分开，几何/语义不能共享投影。

---

## 5. SnapFlow（2604.05656，操作）

### ① 流程

- **S1** 标准 flow-matching 分支：随机 t，加噪，预测速度。
- **S2** shortcut 分支：用模型自己的两步 Euler 端点作为目标（stop-grad）。
- **S3** 加一个零初始化的目标时间投影，让同一个 expert 同时服务两个目标。
- **S4** 两个损失相加训练。
- **S5** 推理一次前向出动作。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `model` | `x_t [B,H,D]`, `t [B,1]`, `t_dst [B,1]`, `cond` | `v [B,H,D]` | 带目标时间条件的去噪网络 |
| `TimeProj` | `t_emb [B,1,d]` | 同形 | 零初始化，初始化时恒等 |
| `shortcut_target` | `x_t`, `t` | `v_tgt [B,H,D]` | 两段 Euler 端点折成速度 |
| `sample_one_step` | `x1`, `cond` | `a0 [B,H,D]` | 1 NFE 推理 |

### ③ 主伪代码

**公式（原文/重构）**

```text
L_FM = E_{t,x0,ε} || F_θ(x_t, t, t|c) − (ε − x0) ||²
v_sc = (x_dst − x_t) / (0 − t),   x_dst = Euler2(x_t, t)
L    = L_FM + λ · || F_θ(x_t, t, 0|c) − v_sc ||²
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `x_t` | `(1-t)*x0 + t*eps` | 线性插值 |
| `t_dst=t` | `model(..., t_dst=t)` | FM 分支目标时间=当前 |
| `t_dst=0` | `model(..., t_dst=0)` | shortcut 分支目标时间=终点 |
| `x_dst` | 两步 Euler，`no_grad` | 自身端点 |
| `no_grad` | `with torch.no_grad():` | stop-grad |


```python
def snapflow_step(model, x0, cond, lam=1.0):
    # 输入:
    #   x0   [B,H,D]  真值动作
    #   cond [B,N,C]  视觉语言条件
    # 输出:
    #   loss  : scalar
    #   stats : dict {"L_fm", "L_sc"}
    B  = x0.size(0)
    t  = torch.rand(B, 1, device=x0.device)
    eps = torch.randn_like(x0)
    x_t = (1 - t)[..., None] * x0 + t[..., None] * eps

    # [S1] 标准 FM：目标时间=当前 t
    v_pred = model(x_t, t, t_dst=t, cond=cond)
    L_fm   = F.mse_loss(v_pred, eps - x0)

    # [S2] shortcut：自己的两步 Euler 端点（整链 no_grad）
    with torch.no_grad():
        t_mid = 0.5 * t
        v1    = model(x_t, t, t_dst=t_mid, cond=cond)
        x_mid = x_t + (t_mid - t)[..., None] * v1
        v2    = model(x_mid, t_mid, t_dst=torch.zeros_like(t), cond=cond)
        x_dst = x_mid + (0 - t_mid)[..., None] * v2
        v_tgt = (x_dst - x_t) / (0 - t)[..., None]

    v_short = model(x_t, t, t_dst=torch.zeros_like(t), cond=cond)
    L_sc    = F.mse_loss(v_short, v_tgt)

    loss = L_fm + lam * L_sc                                   # [S4]
    return loss, {"L_fm": L_fm.item(), "L_sc": L_sc.item()}
```

### ④ 子模块：零初始化目标时间投影

```python
class TimeProj(nn.Module):
    # 输入: e [B,1,d] 时间 embedding
    # 输出: e' [B,1,d]；初始化时 e' == e
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d)
        nn.init.zeros_(self.w.weight)                          # ✓ 零初始化
        nn.init.zeros_(self.w.bias)
    def forward(self, e):
        return e + self.w(e)
```

### ⑤ 训练与推理

```python
for batch in loader:
    loss, _ = snapflow_step(model, batch.act, batch.cond)
    loss.backward(); optimizer.step()

@torch.no_grad()
def sample_one_step(model, cond):
    # 输入: cond；输出: a0 [B,H,D]
    x1 = torch.randn(B, H, D)
    return x1 - model(x1, t=torch.ones(B,1), t_dst=torch.zeros(B,1), cond=cond)
```

### ⑥ 实现注意

1. 两步链的 `no_grad` 要包住 `v1/v2/x_mid/x_dst` 全部，漏一个就会自我强化。
2. `t_dst` 与 `t` 必须走**两个不同的 embedding**，再相加；共用一个会丢"目标时间"。
3. 零初始化层在加载预训练权重时必须是新加的，不能覆盖原时间 MLP。

---

## 6. VLA-SSD（2611.20460，操作）

### ① 流程

- **S1** 把 flow map 扩成 Itô map：输入初始动作 + 布朗路径，输出单跳动作。
- **S2** 采样时注入布朗路径，得到随机但可算密度的单步策略。
- **S3** 用一次 JVP 算动作 log-density。
- **S4** 用 GRPO 的 clipped 目标更新策略。
- **S5** 从预训练速度场出发训练 Itô map，不需要外部教师。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `ItoMap` | `a0 [B,L]`, `W [B,L]` | `a1 [B,L]` | 单跳生成 |
| `sample_with_logprob` | `a0`, `s` | `a1`, `logp [B]` | 采样 + 密度 |
| `rollout_reward` | `a1` | `R [B]` | 环境奖励 |
| `grpo_loss` | `logp`, `logp_old`, `R` | `scalar` | 策略梯度 |

### ③ 主伪代码

**公式（原文/重构）**

```text
a_1 = a_0 + G_θ(a_0; W|s) + m,   W ~ 布朗路径,  m = σ·W_T
Â   = (R − R̄) / (std(R) + κ)
L   = −E[ min( ρ·Â, clip(ρ, 1−ε, 1+ε)·Â ) ],   ρ = π_θ/π_old
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `W` | `torch.randn(...).cumsum(-1)` | 离散布朗路径 |
| `G_θ` | `ito.net(cat([a0,W,state]))` | 学习的漂移 |
| `m=σW_T` | `sigma * W` | 路径末端漂移 |
| `Â` | `(R - mean)/(std + κ)` | 组内标准化 |
| `ρ` | `(logp - logp_old).exp()` | 概率比 |
| `clip` | `ratio.clamp(...)` | PPO 式截断 |


```python
def vla_ssd_train_step(ito, a0, state, reward_fn, logp_old, cfg):
    # 输入:
    #   ito       : Itô map 模块
    #   a0        [B,L]   初始动作（噪声）
    #   state     : 观测/语言条件（这里省略）
    #   reward_fn : callable(a1) -> R [B]
    #   logp_old  [B]     上一版策略的 log-prob
    # 输出:
    #   loss  : scalar
    #   stats : dict {"reward_mean", "zero_var_ratio"}
    W    = torch.randn(B, L, device=a0.device).cumsum(-1)      # [S1] 布朗路径
    a1   = ito(a0, W, state)                                   # [S2] 单跳
    logp, = jvp(lambda z: ito(z, W, state), (a0,), (torch.ones_like(a0),))  # [S3]

    R = reward_fn(a1)                                          # [S3]
    A = (R - R.mean()) / (R.std() + cfg.kappa)                 # [S4]
    ratio = (logp - logp_old).exp()
    loss = -torch.min(ratio * A,
                      ratio.clamp(1 - cfg.eps, 1 + cfg.eps) * A).mean()   # [S4]
    stats = {"reward_mean": R.mean().item(),
             "zero_var_ratio": (R.std(0) < 1e-6).float().mean().item()}
    return loss, stats
```

### ④ 子模块：Itô map

```python
class ItoMap(nn.Module):
    # 输入: a0 [B,L], W [B,L], state [B,C]
    # 输出: a1 [B,L]
    def __init__(self, L, c_state, sigma=0.1):
        super().__init__()
        self.net   = nn.Sequential(nn.Linear(2*L + c_state, 256), nn.GELU(),
                                   nn.Linear(256, L))
        self.sigma = sigma
    def forward(self, a0, W, state):
        g = self.net(torch.cat([a0, W, state], dim=-1))        # 漂移
        return a0 + g + self.sigma * W                          # [S2]
```

### ⑤ 训练与推理

```python
# 训练：GRPO rollout（每步都要算 logp，1 次 JVP）
for it in range(steps):
    a0 = torch.randn(B, L)
    loss, stats = vla_ssd_train_step(ito, a0, state, reward_fn, logp_old, cfg)
    loss.backward(); optimizer.step()

# 推理：单跳，不需要 JVP
a1 = ito(torch.randn(1, L), torch.randn(1, L).cumsum(-1), state)
```

### ⑥ 实现注意

1. `sigma` 要小（≈0.1），否则动作越界、真机不可执行。
2. `logp` 必须用 `torch.func.jvp` 算，数值差分误差会让 ratio 崩。
3. `logp_old` 来自冻结的旧策略，训练中不要更新。

---

## 7. FastOPD（2610.02832，操作）

### ① 流程

- **S1** 学生冻结 VLM 主干，只训 action expert；expert 同时输出速度与区间速度。
- **S2** 学生自己跳一步得到落点，教师只在这个点查询。
- **S3** 在落点上对齐教师速度场。
- **S4** 另采一个更早的 s，用"经中点的两段短跳"监督直跳。
- **S5** 两项相加训练；推理走任意步长单跳。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `student.jump` | `x1`, `s`, `t`, `cond` | `x_t [B,H,D]` | 学生单跳 |
| `student.velocity` | `x_t`, `t`, `cond` | `v [B,H,D]` | 速度场 |
| `student.interval_vel` | `x_s`, `s`, `t`, `cond` | `u [B,H,D]` | 区间平均速度 |
| `teacher.velocity` | `x_t`, `t`, `cond` | `v [B,H,D]` | 冻结教师 |
| `onpolicy_loss` | 学生/教师速度 | `scalar` | 落点蒸馏 |
| `consistency_loss` | 学生两段短跳 | `scalar` | 自一致 |

### ③ 主伪代码

**公式（原文/重构）**

```text
L_sc  = E || u_θ(x_s,s,t) − ½( sg[v_θ(x_s,s)] + sg[v_θ(x̂_m,m)] ) ||²
L_opd = E || v_θ(x̃_t,t) − v_φ(x̃_t,t) ||²
L(θ)  = L_sc + λ·L_opd,   x̃_t = sg[f_θ(x_1,1,t)],  m = (s+t)/2
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `x̃_t` | `student.jump(...).detach()` | 学生落点 |
| `v_φ` | `teacher.velocity(...)` | 冻结教师 |
| `u_θ` | `student.interval_vel(...)` | 区间速度 |
| `½(sg+sg)` | `0.5 * (v_s + v_m)` | 两段短跳平均 |
| `λ` | `cfg.lam` | 原文未给值 |


```python
def fastopd_step(student, teacher, x1, cond, cfg):
    # 输入:
    #   x1   [B,H,D]  噪声（动作生成的起点）
    #   cond [B,N,C]  视觉语言条件
    # 输出:
    #   loss  : scalar = L_sc + lam * L_opd
    #   stats : dict
    B = x1.size(0)

    # [S2] 学生自己的落点（教师只查这一个点）
    t   = torch.rand(B, 1) * 0.9 + 0.05
    x_t = student.jump(x1, s=1.0, t=t, cond=cond).detach()

    # [S3] 落点上的速度对齐
    L_opd = F.mse_loss(student.velocity(x_t, t, cond),
                       teacher.velocity(x_t, t, cond))

    # [S4] 有限区间自一致
    s   = t * torch.rand(B, 1)
    m   = 0.5 * (s + t)
    x_s = student.jump(x1, s=1.0, t=s, cond=cond).detach()
    with torch.no_grad():
        v_s  = student.velocity(x_s, s, cond)
        x_m  = x_s + (m - s) * v_s
        v_m  = student.velocity(x_m, m, cond)
        tgt  = 0.5 * (v_s + v_m)
    L_sc = F.mse_loss(student.interval_vel(x_s, s, t, cond), tgt)

    loss = L_sc + cfg.lam * L_opd                             # [S5]
    return loss, {"L_opd": L_opd.item(), "L_sc": L_sc.item()}
```

### ④ 子模块：学生的 jump 与区间速度

```python
class ActionExpert(nn.Module):
    # 输入: x [B,H,D], t [B,1], cond [B,N,C]
    # 输出: v/u [B,H,D]
    def velocity(self, x, t, cond):
        return self.net(torch.cat([x.flatten(1), t, cond.mean(1)], -1)).view_as(x)

    def interval_vel(self, x, s, t, cond):
        # 区间速度：多一个目标时间输入
        return self.interval_net(torch.cat([x.flatten(1), s, t, cond.mean(1)], -1)).view_as(x)

    @torch.no_grad()
    def jump(self, x1, s, t, cond, n=1):
        x = x1
        for _ in range(n):                                    # Euler 一跳
            x = x + (t - s) * self.velocity(x, s, cond)
            s = t
        return x
```

### ⑤ 训练与推理

```python
for p in student.vlm.parameters(): p.requires_grad_(False)    # ✓ 冻结主干
opt = torch.optim.AdamW(student.expert.parameters(), lr=cfg.lr)

for batch in loader:
    loss, _ = fastopd_step(student, teacher, batch.x1, batch.cond, cfg)
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def infer(student, cond, steps=1):
    x = torch.randn(B, H, D)
    return student.jump(x, s=1.0, t=0.0, cond=cond)            # 单跳
```

### ⑥ 实现注意

1. 教师查询点是 `x_t`（学生落点），不是数据集里的真值动作。
2. `s < t` 要覆盖整个区间，别只在中点附近采。
3. `jump` 里不要对 `x_t` 反传——落点是采样点，detach 掉。

---

## 8. VLA-MOPD（2611.12286，操作）

### ① 流程

- **S1** 共享学生自己跑，收集访问到的观测。
- **S2** 按指令选对应的冻结任务专家。
- **S3** 在每个观测和每个 ODE 中间状态上查教师 transition。
- **S4** 只对有效动作维度算 masked 匹配损失。
- **S5** 可加一个 reference 策略项防遗忘。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `rollout` | `student`, `task` | `[(obs, act)]` | 学生自采样 |
| `route` | `instruction` | `teacher` | 多专家路由 |
| `teacher.transition` | `obs`, `x`, `k` | `x_next [B,H,D]` | 单步 ODE 目标 |
| `masked_match` | `pred`, `tgt`, `mask` | `scalar` | 有效维匹配 |
| `ref_kl` | `student`, `ref` | `scalar` | 防遗忘 |

### ③ 主伪代码

**公式（原文/重构）**

```text
L = (1/|T|) Σ_{(o,x)∈T} Σ_k  mask ⊙ || u_θ(o,x_k) − sg[u_T(o,x_k)] ||²
    + β · KL( π_θ(·|o) || π_ref(·|o) )
```

> 注：该式为按论文描述的**重构**（摘要与正文未给逐项公式），mask/权重细节原文未公开。

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `T` | `traj` | 学生 rollout 的观测集合 |
| `x_k` | `x` 逐步推进 | 每个 ODE 中间状态 |
| `mask` | `valid_dims(act)` | 只算有效维 |
| `sg[u_T]` | `no_grad` + `transition(teacher,...)` | 教师目标 |
| `β·KL` | `beta * ref_kl(...)` | 防遗忘 |


```python
def vla_mopd_step(student, teachers, task, ref, cfg):
    # 输入:
    #   student  : 共享学生
    #   teachers : dict[task_id -> frozen teacher]
    #   task     : 含 instruction / obs
    #   ref      : reference policy（可为 None）
    # 输出:
    #   loss  : scalar
    #   stats : dict
    traj = rollout(student, task)                             # [S1]
    teacher = teachers[route(task.instruction)]               # [S2]

    loss = 0.0
    for obs, act in traj:
        x = torch.randn_like(act)                             # 起点噪声
        for k in range(cfg.K):                                # [S3] 逐 flow step
            with torch.no_grad():
                tgt = transition(teacher, obs, x, k, cfg.K)   # detach 目标
            mask = valid_dims(act)                            # [S4] 有效维
            loss = loss + masked_match(transition(student, obs, x, k, cfg.K), tgt, mask)
            x = x + tgt / cfg.K                               # 沿教师轨迹推进

    if ref is not None:
        loss = loss + cfg.beta * ref_kl(student, ref, task)   # [S5]
    return loss / max(len(traj), 1), {...}
```

### ④ 子模块：mask、路由与单步 transition

```python
def valid_dims(act):
    # 输入: act [B,H,D]
    # 输出: mask [B,H,D] float，1=有效
    return (act.abs().sum(-1, keepdim=True) > 1e-8).float().expand_as(act)

def masked_match(pred, tgt, mask):
    # 输入: pred/tgt [B,H,D], mask 同形
    # 输出: scalar
    return (mask * (pred - tgt) ** 2).sum() / mask.sum().clamp(min=1)

def route(instruction):
    # 输入: instruction str
    # 输出: task_id
    return TASK_MAP[normalize(instruction)]

def transition(expert, obs, x, k, K):
    # 输入: obs [B,N,C], x [B,H,D], k 当前步, K 总步数
    # 输出: x_next [B,H,D]（沿教师/学生速度场走一步）
    t = 1.0 - k / K
    return x + (1.0 / K) * expert.velocity(x, t, obs)
```

### ⑤ 训练与推理

```python
for batch in loader:
    loss, _ = vla_mopd_step(student, teachers, batch.task, ref, cfg)
    loss.backward(); optimizer.step()

# 推理：单个共享学生，不再需要教师
actions = student.sample(obs, instruction)
```

### ⑥ 实现注意

1. 每个 flow step 的目标都要 detach；不 detach 会通过教师轨迹反传到教师。
2. mask 用 float，布尔直接乘会类型报错或丢梯度。
3. reference 项权重很小（β≈0.01），主要作用是防止 Object suite 掉 13.2 点。

---

## 9. APM（2611.22592，WAM 双臂）

### ① 流程

- **S1** Stage I：AnyFlow 式有限区间蒸馏，把整网变成"任意区间一步走"的 map。
- **S2** Stage II：冻结 base，只训 video/action 两组 LoRA。
- **S3** 学生从噪声滚到 r，得到自达状态。
- **S4** 教师从该状态积分出端点锚和首步中间态。
- **S5** 两个损失相加；梯度冲突时用 anchor-priority 投影。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `rollout` | `student`, `K`, `to=r` | `z_r [B,N,D]` | 学生自达状态 |
| `teacher.integrate` | `z_r`, `r`, `target_t` | `z [B,N,D]` | 教师积分 |
| `psi` | `z`, `from`, `to` | `z' [B,N,D]` | transition map |
| `anchor_loss` | `psi`, `y0` | `scalar` | 端点锚 |
| `field_loss` | `psi`, `zs` | `scalar` | 首步中间态 |
| `project` | `g_a`, `g_f` | `g_total` | 梯度协调 |

### ③ 主伪代码

**公式（原文/重构）**

```text
L_anchor = || Ψ(z_r; r→0) − y_0 ||²,   y_0 = sg[Φ_T^{r→0}(z_r)]
L_field  = || Ψ(z_r; r→s) − z_s ||²,   z_s = sg[Φ_T^{r→s}(z_r)],  s = r − 1/K_T
g_f^⊥    = g_f − min(⟨g_f, g_a⟩, 0) / ||g_a||² · g_a
θ ← θ − η ( g_base + g_a + g_f^⊥ )
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `z_r` | `rollout(student, ..., to=r).detach()` | 学生自达状态 |
| `y_0` | `teacher.integrate(z_r, r, 0)` | 端点锚 |
| `z_s` | `teacher.integrate(z_r, r, s)` | 首步中间态 |
| `min(⟨g_f,g_a⟩,0)` | `(f*a).sum().clamp(max=0)` | 只去冲突分量 |
| `g_f^⊥` | `f - coef * a` | 投影 |


```python
def apm_step(student, teacher, x1, cfg):
    # 输入:
    #   x1 [B,N,D] 噪声（视频/动作 latent）
    # 输出:
    #   loss  : scalar
    #   stats : dict
    r = torch.rand(B, 1) * 0.9 + 0.05
    z_r = rollout(student, x1, K=cfg.K, to=r).detach()          # [S3]

    s   = r - 1.0 / cfg.K_T                                      # 教师首步
    with torch.no_grad():
        y_0 = teacher.integrate(z_r, r, 0.0)                     # [S4] 端点锚
        z_s = teacher.integrate(z_r, r, s)                       # [S4] 首步中间态

    L_anchor = F.mse_loss(psi(student, z_r, r, 0.0), y_0)        # [S5]
    L_field  = F.mse_loss(psi(student, z_r, r, s),   z_s)

    g_a, g_f = grad(L_anchor, student.lora), grad(L_field, student.lora)
    g_total  = project(g_a, g_f)                                 # [S5]
    return L_anchor + cfg.lam * L_field, {...}
```

### ④ 子模块：transition map 与梯度投影

```python
def psi(student, z, t_from, t_to):
    # 输入: z [B,N,D], t_from/t_to [B,1] 或标量
    # 输出: z' [B,N,D]（任意区间一步走）
    return student.transition_map(z, t_from, t_to)

def project(g_a, g_f):
    # 输入: 两个梯度张量列表（同形）
    # 输出: 合成梯度列表；只去掉 g_f 中与 g_a 冲突的分量
    out = []
    for a, f in zip(g_a, g_f):
        coef = (f * a).sum().clamp(max=0) / (a.norm() ** 2 + 1e-8)
        out.append(f - coef * a)
    return out
```

### ⑤ 训练与推理

```python
# Stage I 完成后进入 Stage II
for p in student.base.parameters(): p.requires_grad_(False)      # ✓ 冻结 base
opt = torch.optim.AdamW(student.lora.parameters(), lr=cfg.lr)

for batch in loader:
    loss, _ = apm_step(student, teacher, batch.x1, cfg)
    opt.zero_grad(); loss.backward(); opt.step()

# 推理：K ∈ {1,2,4} 单 ckpt 切换
actions = student.sample(x1, steps=K)
```

### ⑥ 实现注意

1. Stage II 必须冻结 base，否则 anchor 与 field 会把整网拉飞。
2. 投影只对冲突方向（内积为负）生效，同向分量保留。
3. `rollout` 的 K 与推理 K 可以不同，但要覆盖训练分布。

---

## 10. RealtimeWAM / TACD（2610.06617，WAM）

### ① 流程

- **S1** 冻结 Video Expert，一次前向产出 block-wise KV。
- **S2** 只训 action expert，另维护 EMA 目标。
- **S3** 教师多步 rollout 出端点，折成区间平均速度。
- **S4** 区间速度损失 + 一致性损失。
- **S5** action block 只用对应 video block 的 KV（CEWP）。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `video_expert.kv` | `obs` | `kv`（按 block） | 冻结，只前向一次 |
| `action_expert` | `a_t`, `t`, `kv` | `v [B,H,D]` | 学生 |
| `solver` | `a_t`, `t`, `0`, `teacher`, `kv` | `a0_T [B,H,D]` | 教师端点 |
| `interval_velocity` | `a_t`, `a0_T`, `t` | `u [B,H,D]` | 区间平均速度 |
| `consistency_loss` | 学生/EMA | `scalar` | 局部一致性 |

### ③ 主伪代码

**公式（原文/重构）**

```text
u_T   = (a_t − a_0^T) / t,   a_0^T = Solver(a_t, t, 0; θ_T)
L_TA  = E || v_S(a_t,t) − sg[u_T] ||²
L     = L_CD + λ·L_TA
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `a_0^T` | `solver(a_t, t, 0, teacher, kv)` | 教师多步端点 |
| `u_T` | `(a_t - a0_T) / t.clamp(min=1e-3)` | 区间平均速度 |
| `L_CD` | `consistency_loss(...)` | 局部一致性 |
| `kv` | `video_expert.kv(obs)` | 一次前向共享 |


```python
def tacd_step(action_expert, ema, teacher, obs, a0, cfg):
    # 输入:
    #   obs  [B,V,3,H,W]
    #   a0   [B,H,D]
    # 输出:
    #   loss  : scalar = L_CD + lam * L_TA
    #   stats : dict
    with torch.no_grad():
        kv = video_expert.kv(obs)                             # [S1] 只算一次

    t   = torch.rand(B, 1)
    eps = torch.randn_like(a0)
    a_t = (1 - t) * a0 + t * eps

    v_S = action_expert(a_t, t, kv)                           # [S2]
    with torch.no_grad():
        a0_T = solver(a_t, t, 0.0, teacher, kv)               # [S3] 教师端点
        u_T  = (a_t - a0_T) / t.clamp(min=1e-3)               # [S3] 区间平均速度

    L_TA = F.mse_loss(v_S, u_T)                               # [S4]
    L_CD = consistency_loss(action_expert, ema, a_t, t, kv)   # [S4]
    return L_CD + cfg.lam * L_TA, {...}
```

### ④ 子模块：CEWP 调度

```python
def cewp_forward(action_expert, a_t, t, kv):
    # 输入: kv 是 list[block]，每个 block [B,N,C]
    # 输出: v [B,H,D]；action block i 只看 video block i
    outs = []
    for i, blk in enumerate(action_expert.blocks):
        outs.append(blk(a_t, t, kv[i]))                       # [S5] 逐块对应
    return action_expert.head(torch.cat(outs, dim=1))
```

### ⑤ 训练与推理

```python
for batch in loader:
    loss, _ = tacd_step(action_expert, ema, teacher, batch.obs, batch.a0, cfg)
    loss.backward(); optimizer.step()
    ema.update(action_expert)

@torch.no_grad()
def infer(obs):
    kv = video_expert.kv(obs)
    return action_expert.sample(kv, steps=1)
```

### ⑥ 实现注意

1. video expert 必须真冻结；KV 缓存复用时别重复算。
2. `u_T` 的 `1/t` 在 t→0 时会爆，`clamp(min=1e-3)` 必须有。
3. CEWP 的 block 对齐关系要与训练时一致，否则推理错位。

---

## 11. WAM-Policy（2611.24011，WAM 四足）

### ① 流程

- **S1** 教师由 LingBot-VA 换四足接口微调后冻结。
- **S2** 学生 = 教师 + LoRA，采样器降到 1 步视频 / 2 步动作。
- **S3** 教师高 NFE 采样出视频/动作目标，可缓存。
- **S4** 两个端点分别回归，用 pseudo-Huber。
- **S5** 只更新 LoRA。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `teacher.sample_video` | `eps_Z`, `H` | `Z_T [B,N,Cz]` | 高 NFE 目标 |
| `teacher.sample_action` | `eps_A`, `Z_T`, `H` | `A_T [B,H,D]` | 高 NFE 目标 |
| `student.sample_video` | `eps_Z`, `H` | `Z_hat` | 1 步 |
| `student.sample_action` | `eps_A`, `sg(Z_hat)`, `H` | `A_hat` | 2 步 |
| `pseudo_huber` | `r`, `c` | `scalar` | 鲁棒回归 |

### ③ 主伪代码

**公式（原文/重构）**

```text
ρ_c(r) = √(r² + c²) − c
L_Z = ρ̄_c(Ẑ − Z_T),   L_A = ρ̄_c(Â − A_T)
L   = L_Z + λ_A·L_A,   Â 以 sg[Ẑ] 为条件
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `ρ_c` | `(r**2 + c**2).sqrt() - c` | pseudo-Huber |
| `Ẑ` | `student.sample_video(..., steps=1)` | 1 步视频 |
| `Â` | `student.sample_action(..., Z_hat.detach(), steps=2)` | 2 步动作 |
| `Z_T/A_T` | `teacher.sample_*(..., steps=K_T)` | 高 NFE 目标 |


```python
def wam_policy_step(student, teacher, cfg):
    # 输入:
    #   student : 教师 + LoRA（Δ 唯一可训）
    #   teacher : 冻结
    # 输出:
    #   loss  : scalar = L_Z + lam_A * L_A
    #   stats : dict
    B = cfg.batch
    eps_Z = torch.randn(B, cfg.N, cfg.Cz)
    eps_A = torch.randn(B, cfg.H, cfg.D)

    Z_hat = student.sample_video(eps_Z, H=cfg.H, steps=1)          # [S2]
    A_hat = student.sample_action(eps_A, Z_hat.detach(), H=cfg.H, steps=2)  # [S2]

    with torch.no_grad():
        Z_T = teacher.sample_video(eps_Z, H=cfg.H, steps=cfg.K_T)  # [S3]
        A_T = teacher.sample_action(eps_A, Z_T, H=cfg.H, steps=cfg.K_T)

    L_Z = pseudo_huber(Z_hat - Z_T).mean()                         # [S4]
    L_A = pseudo_huber(A_hat - A_T).mean()                         # [S4]
    return L_Z + cfg.lam_A * L_A, {...}                            # [S5]


def pseudo_huber(r, c=1e-3):
    # 输入: r 任意形状；输出: 同形
    return (r ** 2 + c ** 2).sqrt() - c
```

### ④ 子模块：采样器与 LoRA 注入

```python
@torch.no_grad()
def sample_video(model, eps, H, steps):
    # 输入: eps [B,N,Cz], H int, steps int
    # 输出: Z [B,N,Cz]（flow Euler 采样）
    z = eps
    for i in range(steps):
        t = 1.0 - i / steps
        z = z - (1.0 / steps) * model.video_velocity(z, t, H)
    return z

@torch.no_grad()
def sample_action(model, eps, Z, H, steps):
    # 输入: eps [B,H,D], Z [B,N,Cz], steps int
    # 输出: A [B,H,D]
    a = eps
    for i in range(steps):
        t = 1.0 - i / steps
        a = a - (1.0 / steps) * model.action_velocity(a, t, Z, H)
    return a

def add_lora(module, r=16, alpha=32):
    # 输入: 原 module（冻结）；输出: 包了 LoRA 的 module（仅 LoRA 可训）
    for p in module.parameters(): p.requires_grad_(False)
    for layer in module.transformer_layers:
        layer.attn.q_proj = LoRALinear(layer.attn.q_proj, r, alpha)
        layer.attn.v_proj = LoRALinear(layer.attn.v_proj, r, alpha)
    return module
```

### ⑤ 训练与推理

```python
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=cfg.lr)
for step in range(cfg.steps):
    loss, _ = wam_policy_step(student, teacher, cfg)
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def infer(student, obs, lang):
    Z = student.sample_video(torch.randn(...), steps=1)
    return student.sample_action(torch.randn(...), Z, steps=2)
```

### ⑥ 实现注意

1. `Z_hat` 进 action 时必须 detach——plan 只在视频分支被监督。
2. 教师目标可缓存（同一 eps 下），但 eps 每步随机，缓存收益有限。
3. LoRA 只加 attention 可能容量不够，MLP 也建议加（原文未细说）。

---

## 12. W2A（2611.02600，WAM）

### ① 流程

- **S1** 视频 DiT 用 LoRA 适配机器人视频，取中间层/中间步 latent。
- **S2** 对 latent 做 PCA，压到低维子空间。
- **S3** action DiT 从零训练；Q-Former 把动作侧中间层映射到同一空间。
- **S4** 按去噪时间加权，做 cosine 对齐。
- **S5** 总损失 = 动作 FM + λ·对齐；测试丢视频模型。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `video_dit.latent` | 未来帧, `layer`, `step` | `Z_v [B,N,Cv]` | 教师 latent |
| `pca_fit` | `Z_v` | `mu`, `U [Cv,K]` | 离线降维 |
| `action_dit.hidden` | `a_tau`, `tau` | `Z_a [B,N,Ca]` | 学生中间层 |
| `QFormer` | `query`, `Z_a` | `Z_hat [B,M,K]` | 对齐到 K 维 |
| `time_weight` | `tau` | `w [B]` | 高斯权重 |

### ③ 主伪代码

**公式（原文/重构）**

```text
L_distill = (1/B) Σ_b w(τ_b) · (1/M) Σ_i (1 − cos(ẑ^a_i, z̃^v_i))
w(τ)      = exp( −(τ − τ_c)² / (2σ_w²) )
L         = L_FM + λ_dist · L_distill
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `z̃^v` | `(Z_v - mu) @ U` | PCA 压缩 |
| `ẑ^a` | `qformer(query, Z_a)` | 动作侧投影 |
| `w(τ)` | `torch.exp(-(tau-tau_c)**2/(2*sigma_w**2))` | 时间权重 |
| `λ_dist` | `cfg.lam_dist` | 原文未给值 |


```python
def w2a_step(action_dit, qformer, video_dit, pca, batch, cfg):
    # 输入:
    #   batch.future [B,F,3,H,W]  未来帧（仅训练期教师可见）
    #   batch.act    [B,H,D]
    # 输出:
    #   loss  : scalar = L_fm + lam_dist * L_dist
    #   stats : dict
    with torch.no_grad():
        Z_v = video_dit.latent(batch.future, layer=cfg.layer, step=cfg.tau)  # [S1]
    mu, U = pca
    Z_v_t = (Z_v - mu) @ U                                   # [S2] [B,N,K]

    tau  = torch.rand(B, 1)
    Z_a  = action_dit.hidden(add_noise(batch.act, tau), tau, cond=batch.cond)  # [S3]
    Z_hat = qformer(cfg.query, Z_a)                          # [S3] [B,M,K]

    w = torch.exp(-(tau - cfg.tau_c) ** 2 / (2 * cfg.sigma_w ** 2))   # [S4]
    L_dist = (w * (1 - F.cosine_similarity(Z_hat, Z_v_t, dim=-1))).mean()

    L_fm = flow_matching(action_dit, batch.act, batch.cond)
    return L_fm + cfg.lam_dist * L_dist, {...}               # [S5]
```

### ④ 子模块：PCA 拟合

```python
def pca_fit(Z, K):
    # 输入: Z [B*N, Cv] float32
    # 输出: mu [Cv], U [Cv,K]
    mu = Z.mean(0, keepdim=True)
    Zc = Z - mu
    _, _, Vh = torch.linalg.svd(Zc, full_matrices=False)
    return mu.squeeze(0), Vh[:K].T
```

### ⑤ 训练与推理

```python
mu, U = pca_fit(collect_video_latents(video_dit, loader), K=cfg.K)   # 离线一次
for batch in loader:
    loss, _ = w2a_step(action_dit, qformer, video_dit, (mu, U), batch, cfg)
    loss.backward(); optimizer.step()

# 推理：只跑 action_dit
actions = action_dit.sample(cond, steps=1)
```

### ⑥ 实现注意

1. PCA 必须离线拟合并冻结，训练中更新会让目标漂移。
2. 余弦相似度在 K 维上算，别在原始 Cv 维上算。
3. 时间权重 `w(τ)` 的峰值 τ_c 要落在"教师信息量最大"的去噪阶段（原文有消融）。

---

## 13. VAD（2610.14885，操作）

### ① 流程

- **S1** 策略采 8 个候选，verifier 只评前 10 步。
- **S2** softmax 加权聚合成一条执行动作。
- **S3** 执行前 10 步，缓存观测与聚合动作。
- **S4** episode 成功才保留。
- **S5** 用标准 flow-matching loss 蒸进 LoRA。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `student.sample` | `obs` | `cand [B,H,D]` | 候选 |
| `verifier` | `obs`, `chunk[:10]` | `scores [N]` | 打分 |
| `aggregate` | `cands`, `scores` | `A_bar [B,H,D]` | 聚合 |
| `collector` | 成功 episode | `D` | 经验池 |
| `lora_flow_step` | `(o, A_bar)` | `scalar` | 蒸馏 |

### ③ 主伪代码

**公式（原文/重构）**

```text
Ā    = Σ_i softmax(s_i/λ) · A_i,   N = 8, λ = 0.3
A_τ  = (1 − τ)Ā + τε
L(φ) = E || v_{θ0,φ}(A_τ, τ | o) − (ε − Ā) ||²_F
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `softmax(s/λ)` | `torch.softmax(scores/0.3, 0)` | 候选权重 |
| `Ā` | `sum(w[i] * cands[i])` | 聚合 |
| `A_τ` | `(1-tau)*A_bar + tau*eps` | 插值 |
| `L` | `F.mse_loss(student.velocity(...), eps - A_bar)` | 只蒸成功 |


```python
def vae_collect_episode(student, verifier, env, cfg):
    # 输入: student, verifier, env
    # 输出: 成功则返回 [(obs, A_bar)]，否则 []
    buf = []
    for b in range(env.replan_steps):
        obs = env.observe()
        cands  = [student.sample(obs) for _ in range(cfg.N)]          # [S1]
        scores = verifier(obs, [c[:10] for c in cands])               # [S1]
        A_bar  = aggregate(cands, scores, cfg.lam)                    # [S2]
        env.execute([c[:10] for c in cands]); buf.append((obs, A_bar))# [S3]
    return buf if env.success else []                                 # [S4]


def aggregate(cands, scores, lam=0.3):
    # 输入: cands list[[B,H,D]], scores [N]
    # 输出: A_bar [B,H,D]
    w = torch.softmax(scores / lam, dim=0)
    return sum(w[i] * cands[i] for i in range(len(cands)))


def vae_distill_step(student, D):
    # 输入: D = [(obs, A_bar)]
    # 输出: loss scalar
    o, A_bar = sample_batch(D)
    tau = torch.rand(B, 1); eps = torch.randn_like(A_bar)
    A_tau = (1 - tau) * A_bar + tau * eps
    return F.mse_loss(student.velocity(A_tau, tau, o), eps - A_bar)   # [S5]
```

### ④ 子模块：verifier

```python
class Verifier(nn.Module):
    # 输入: obs [B,N,C], chunk [B,10,D]
    # 输出: score [B]
    def __init__(self, c_in, hidden=512, layers=6):
        super().__init__()
        self.net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, 8, batch_first=True), layers)  # ✓ 6 层
        self.head = nn.Linear(hidden, 1)
    def forward(self, obs, chunk):
        x = torch.cat([obs, chunk], dim=1)
        return self.head(self.net(x).mean(1)).squeeze(-1)
```

### ⑤ 训练与推理

```python
D = []
for ep in range(cfg.episodes):                    # 每任务 10 个 adaptation episode ✓
    D += vae_collect_episode(student, verifier, env, cfg)

opt = torch.optim.AdamW(lora_params(student), lr=1e-4)   # ✓ lr=1e-4
for step in range(200):                                   # ✓ 200 步
    loss = vae_distill_step(student, D)
    opt.zero_grad(); loss.backward(); opt.step()

# 推理：单候选，不再调 verifier
actions = student.sample(obs)
```

### ⑥ 实现注意

1. 只蒸成功 episode，失败样本会把 verifier 噪声学进去。
2. 聚合用的是"执行的前 10 步"，不是整段 chunk。
3. verifier 要冻结；它和策略一起训会自欺。

---

## 14. H2E（2611.16246，导航）

### ① 流程

- **S1** LLM planner + 冻结 VLA 组队跑任务，只保留成功 session。
- **S2** 把 planner 的子指令/工具移动/停止决策全部重标为 VLA 动作。
- **S3** 用"是否朝目标前进"生成 progress mask。
- **S4** 距目标 1m 内加停机标签。
- **S5** 与 DART/R2R 数据混合，masked L1 训练。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `run_agent` | `planner`, `vla` | `sessions` | 只留成功 |
| `relabel` | `step` | `action [D]` | 重标动作 |
| `progress_mask` | `p_t`, `p_{t+K}`, `goal` | `m_t ∈ {0,1}` | 进度掩码 |
| `stop_label` | `p_t`, `goal` | `y_t` 或 STOP | 停机 |
| `mix` | 三份数据 | `loader` | 0.60/0.25/0.15 |

### ③ 主伪代码

**公式（原文/重构）**

```text
m_t = 1[ d(p_{t+K}, g) ≤ d(p_t, g) + δ ],   δ = 0.25 m
y_t = STOP if d(p_t, g) ≤ r_arr,             r_arr = 1 m
L   = Σ_t m_t · |π(o_t) − y_t| / Σ_t m_t
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `m_t` | `progress_mask(...)` | 0/1 进度掩码 |
| `y_t=STOP` | `if dist <= 1.0: y = STOP_LABEL` | 停机 |
| `masked L1` | `(mask * (pred - label).abs()).mean()` | 只算前进步 |
| `混合` | `mix_datasets(..., (0.60,0.25,0.15))` | 原文 |


```python
def build_h2e_dataset(planner, vla, cfg):
    # 输入: planner（LLM）, vla（冻结 8B）
    # 输出: [{"obs", "label", "mask"}]，以及停机标签
    data = []
    for s in run_agent(planner, vla, cfg.tasks):          # [S1]
        if not s.reached_goal:
            continue
        for t, step in enumerate(s):                      # [S2]
            y = relabel(step)                             # 动作/子指令/停止
            m = progress_mask(s.p[t], s.p[t + cfg.K], s.goal)   # [S3]
            if dist(s.p[t], s.goal) <= cfg.r_arr:         # [S4] 1m
                y = STOP_LABEL
            data.append({"obs": step.obs, "label": y, "mask": m})
    return data


def progress_mask(p_t, p_tk, goal, delta=0.25):
    # 输入: 位置（2D/3D），输出: 0/1
    return float(dist(p_tk, goal) <= dist(p_t, goal) + delta)   # [S3]


def h2e_train_step(student, batch):
    # 输入: batch.obs [B,V,3,H,W], batch.label [B,D], batch.mask [B]
    # 输出: loss scalar
    pred = student(batch.obs)                             # LoRA + waypoint head
    return (batch.mask * (pred - batch.label).abs()).mean()   # [S5] masked L1
```

### ④ 子模块：数据混合

```python
def mix_datasets(agent_data, dart, r2r, ratios=(0.60, 0.25, 0.15)):
    # 输入: 三份数据；输出: 混合后的 Dataset
    n = len(agent_data)
    return ConcatDataset([
        agent_data,
        Subset(dart, range(int(n * ratios[1] / ratios[0]))),
        Subset(r2r,  range(int(n * ratios[2] / ratios[0])))])
```

### ⑤ 训练与推理

```python
data = build_h2e_dataset(planner, vla, cfg)
loader = DataLoader(mix_datasets(data, DART, R2R), batch_size=16)

for step in range(3000):                                  # ✓ 3000 updates
    batch = next(iter(loader))
    loss = h2e_train_step(student, batch)
    loss.backward(); optimizer.step()

# 推理：只跑 VLA，无 LLM、无工具
action = student(obs)
```

### ⑥ 实现注意

1. 只训 LoRA + waypoint head，视觉编码器与 LLM 冻结 ✓。
2. mask 去掉的是"远离目标"的步，不是随机丢步。
3. 停机标签要单独统计（30,181 个），别被动作样本淹没。

---

## 15. 剪枝 + 离线隐层（2609.19579，操作）

### ① 流程

- **S1** 对 7B 模型做结构化宽度剪枝（63% 或 72%）。
- **S2** 教师对所有训练帧提 hidden state，写缓存。
- **S3** 学生同层 hidden 与教师做 MSE（同形状，无 projector）。
- **S4** 加原任务损失。
- **S5** 约 8 GPU-hours 完成恢复。

### ② 模块

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `prune_width` | `model`, `ratio` | `pruned` | 同构剪枝 |
| `cache_hidden` | `loader`, `teacher` | `.npy` | 离线特征 |
| `align_step` | `batch`, `pruned`, `cache` | `scalar` | 同层 MSE |

### ③ 主伪代码

**公式（原文/重构）**

```text
L = L_task + λ · || H_s(x) − H_t(x) ||²
```

**公式项 ↔ 代码**

| 公式项 | 代码 | 说明 |
|---|---|---|
| `H_t` | `cache[key]` | 离线教师 hidden |
| `H_s` | `pruned(obs, return_hidden=True)` | 同形状 |
| `λ` | `cfg.lam` | 原文未给 |


```python
def prune_and_distill(teacher, loader, cfg):
    # 输入: teacher（7B）, loader, cfg.ratio
    # 输出: pruned 学生（已恢复）
    pruned = prune_width(teacher, ratio=0.63)                  # [S1]

    cache = {}
    with torch.no_grad():
        for batch in loader:                                   # [S2]
            cache[batch.key] = teacher(batch.obs, return_hidden=True).half()

    opt = torch.optim.AdamW(pruned.parameters(), lr=cfg.lr)
    for batch in loader:
        H_s = pruned(batch.obs, return_hidden=True)            # [S3] 同形状
        L_align = F.mse_loss(H_s, cache[batch.key].float())
        L_task  = task_loss(pruned(batch.obs), batch.label)
        loss    = L_task + cfg.lam * L_align                   # [S4]
        opt.zero_grad(); loss.backward(); opt.step()
    return pruned                                              # [S5]
```

### ④ 子模块：同构剪枝

```python
def prune_width(model, ratio):
    # 输入: model；输出: 同层数、同接口、宽度变小的模型
    for layer in model.layers:
        keep = int(layer.hidden_dim * (1 - ratio))
        layer.mlp  = shrink_linear(layer.mlp,  keep)
        layer.attn = shrink_heads(layer.attn,   keep)
    return model
```

### ⑤ 训练与推理

```python
pruned = prune_and_distill(teacher, loader, cfg)     # ~8 GPU-hours ✓
pruned.save("openvla_oft_pruned.pt")

# 推理：板载 2.23×、显存 −62% ✓
actions = pruned(obs)
```

### ⑥ 实现注意

1. 只能做**同构**剪枝（改宽度不改层数/接口），否则要加 projector，成本和增益都变差。
2. 缓存必须与剪枝后模型的层一一对应；换层号要重建缓存。
3. `lam` 原文未给，经验上从 0.5 起调。

---

## 16. 方法 ↔ 模块 ↔ 论文 对照

| 代码节 | 方法 | 核心模块 | 论文 |
|---|---|---|---|
| §1 | Orion-Lite | StudentDecoder / planner / traj_losses | 2604.08266 §3.2–3.4 |
| §2 | THAW-VLA | build_cache / p_phi / flow_matching | 2609.24682 §III |
| §3 | CKT-WAM | Bottleneck / LQCA / AdapterRouter | 2605.06247 §3 |
| §4 | S-VAM | upsample_features / svd_sample / Decoupler | 2603.16195 §3 |
| §5 | SnapFlow | TimeProj / shortcut target / sample_one_step | 2604.05656 §3 |
| §6 | VLA-SSD | ItoMap / sample_with_logprob / grpo | 2611.20460 §3–4 |
| §7 | FastOPD | jump / velocity / interval_vel | 2610.02832 §3 |
| §8 | VLA-MOPD | rollout / route / masked_match | 2611.12286 §3 |
| §9 | APM | rollout / integrate / project | 2611.22592 §3 |
| §10 | TACD | video_kv / solver / cewp_forward | 2610.06617 §3 |
| §11 | WAM-Policy | sample_video / sample_action / pseudo_huber | 2611.24011 §3 |
| §12 | W2A | video latent / pca_fit / QFormer | 2611.02600 §3 |
| §13 | VAD | verifier / aggregate / lora flow | 2610.14885 §3 |
| §14 | H2E | run_agent / progress_mask / stop label | 2611.16246 §3 |
| §15 | 剪枝+离线 | prune_width / cache_hidden / align | 2609.19579 §3 |

---

更新时间: 2026-11-21
