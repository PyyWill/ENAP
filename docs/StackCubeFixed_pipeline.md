# StackCubeFixed 流水线文档

> 本文档面向 AI Agent 阅读，详细描述 `ENAP_bs/StackCubeFixed` 仓库的完整工作流程。
> 任务场景：ManiSkill StackCube-v1（机械臂堆叠方块）。

---

## 1. 仓库结构总览

```
StackCubeFixed/
├── agent/                          # 核心模型与工具
│   ├── pmm_class.py                # PMM（Probabilistic Mealy Machine）类定义
│   ├── pmm_agent.py                # PMMResidualAgent + ResidualMLP 网络定义
│   └── utils.py                    # NatureCNN、特征归一化、特征索引构建、数据加载
├── data/
│   ├── preprocess.py               # 第一步：数据预处理（特征提取 + 聚类 + 状态标注）
│   └── episodes_with_states_first100_state_paths.jsonl  # 参考数据
├── results/
│   ├── pmm_residual_training.dot   # PMM 自动机可视化（训练后的 general policy）
│   ├── preference/                 # Büchi 自动机偏好约束（手工编写）
│   │   ├── not_c2.dot              # 偏好："避免聚类 c2"
│   │   └── not_c6.dot              # 偏好："避免聚类 c6"
│   └── product/                    # 乘积 PMM 结果
│       ├── product_pmm_not_c2.dot
│       └── product_pmm_not_c6.dot
└── scripts/
    ├── train/
    │   ├── rnn_train.py            # 第二步：RNN 预训练（序列编码器）
    │   ├── residual_train.py       # 第三步：PMM 构建 + Residual 网络训练
    │   └── production_pmm_ba.py    # 第四步：PMM × Büchi 乘积构造（偏好细化）
    └── eval/
        └── pmm_eval.py             # 第五步：策略评估
```

---

## 2. 完整流水线

整体流程分为 5 个阶段，产出两个策略：**p1**（general policy）和 **p2**（偏好细化后的 policy）。

```
原始轨迹 pkl
    │
    ▼
[阶段 1] preprocess.py ─────► episodes_with_states.pkl（含聚类标签 + 聚类中心）
    │
    ▼
[阶段 2] rnn_train.py ──────► rnn_pretrained.pt（RNN 序列编码器权重）
    │
    ▼
[阶段 3] residual_train.py ─► PMM 自动机 + residual_net.pt ──── 组合即为 p1（general policy）
    │
    ▼
[阶段 4] production_pmm_ba.py ► product PMM pkl ──── 与 residual_net 组合即为 p2（偏好 policy）
    │
    ▼
[阶段 5] pmm_eval.py ───────► 在 ManiSkill 环境中评估 p1 或 p2
```

---

## 3. 各阶段详解

### 3.1 阶段 1：数据预处理（`data/preprocess.py`）

**输入**：ManiSkill 采集的原始轨迹 pkl 文件（每条轨迹的每一步包含 `rgb_feature`、`pos_feature`、`action`、可选 `rgb` 图像）。

**处理步骤**：

1. **特征提取**：遍历所有 episode 的每一步，提取 `rgb_feature`（RGB 视觉特征）和 `pos_feature`（位姿/关节状态特征），展平为一维向量。
2. **Z-score 标准化**：对 RGB 和 POS 特征分别进行零均值单位方差归一化。
3. **PCA 降维**：对归一化后的 RGB 和 POS 特征分别做 PCA 降至 5 维，然后拼接为 10 维向量。
4. **HDBSCAN 聚类**：在 10 维 PCA 空间上运行 HDBSCAN（`min_cluster_size=max(10, 5%*N)`），将所有 timestep 分配到若干聚类（含 noise = -1）。
5. **状态标注**：根据聚类结果，为每个 timestep 添加 one-hot 编码的 `state` 向量（维度 = 聚类数 + noise，即所有 unique label 数量）。
6. **聚类中心计算**：在原始拼接特征空间（RGB_norm ∥ POS_norm）中计算每个聚类的质心。

**输出** (`episodes_with_states.pkl`)：

```python
{
    "episodes": [...],            # 每步增加了 step['state'] = one-hot 向量
    "num_clusters": int,          # 纯聚类数（不含 noise）
    "num_states": int,            # 总状态数（含 noise）
    "cluster_centers": np.array,  # shape (num_states, feat_dim)
    "cluster_labels": list,       # 各聚类的标签列表
    "metadata": ...               # 如存在则透传
}
```

**关键设计意图**：聚类结果将连续的高维观测空间离散化为有限个"状态符号"（如 c0, c1, ..., c8），使得后续可以用有限状态自动机（PMM）来建模策略。每个符号对应任务执行的一个阶段（例如"接近方块"、"抓取"、"抬起"、"放置"等）。

---

### 3.2 阶段 2：RNN 预训练（`scripts/train/rnn_train.py`）

**输入**：`episodes_with_states.pkl`

**目的**：训练一个 RNN 序列编码器，学习动作-状态序列的隐表征，供后续 PMM 构建时用于节点消歧（disambiguation）。

**模型架构** (`Pretrain`)：

| 组件 | 描述 |
|------|------|
| `state_embed` | `nn.Embedding(state_dim, 16)`，将离散状态索引映射为 16 维嵌入 |
| `enc` | `nn.RNN(action_dim + 16, 64)`，输入为 action 与 state embedding 的拼接 |
| `act_head` | `nn.Linear(64, action_dim)`，预测下一步动作 |
| `cls_head` | `nn.Linear(64, state_dim)`，预测下一步状态 |

**训练细节**：

- **多任务联合损失**：
  - `loss_act`：MSE 动作预测损失
  - `loss_cls`：CrossEntropy 状态分类损失
  - `loss_cont`：Phase-Aware Contrastive Loss（同状态步拉近隐状态，跨状态步推远隐状态）
  - 总损失 = `λ_a * loss_act + λ_s * loss_cls + λ_c * loss_cont`
- **优先经验回放**（PrioritizedReplayBuffer）：按损失大小分配采样优先级。
- 训练参数：`batch_size=512, epochs=32, steps_per_epoch=128, lr=3e-4`，CosineAnnealing 调度。

**输出** (`results/checkpoints/rnn_pretrained.pt`)：

```python
{
    'model_state': state_dict,           # RNN 权重
    'dims': {'a': A_dim, 's': S_dim, 'e': 16, 'h': 64}  # 维度元信息
}
```

**关键设计意图**：RNN 的隐状态编码了"到当前步为止的历史行为模式"。在 PMM 学习过程中，当自动机从一个节点出发有多条边可选时，通过比较当前 RNN 隐状态与各候选目标节点的代表性嵌入（representative embedding）的余弦相似度来选择正确的转移（消歧机制）。

---

### 3.3 阶段 3：PMM 构建 + Residual 网络训练（`scripts/train/residual_train.py`）

此阶段分为两个子过程。

#### 3.3.1 PMM 自动机学习（`agent/pmm_class.py` → `PMM.learn_pmm()`）

PMM（Probabilistic Mealy Machine）是一个概率性 Mealy 机，核心结构为：

```python
{
    'Q': [0, 1, ..., N-1],          # 节点集合（每个节点对应任务的一个子阶段）
    'delta': {(q, x): {q': prob}},  # 转移函数：在节点 q 读入符号 x 后以概率 prob 转移到 q'
    'reps': [tuple, ...],           # 每个节点的代表性前缀序列
    'rep_embeddings': [np.array],   # 每个节点的 RNN 代表性嵌入向量
}
```

**学习算法**：基于改进的 L* 算法（自动机学习），迭代过程为：

1. **初始化**：前缀集 `S = [()]`（空序列）。
2. **闭合性检查**（Closedness Check）：对 S 中每个前缀的 RNN 嵌入，检查其后继状态是否都能被现有节点覆盖（余弦相似度 ≥ `cos_tau_row=0.6`），如不能则添加新前缀。
3. **假设构建**（`_build_pmm`）：基于当前 S，用 RNN 嵌入数据库将所有 timestep 映射到节点，统计转移概率。
4. **等价性查询**（Equivalence Query）：NFA 风格检验——遍历所有 episode，检查假设自动机能否以合理误差再现实际动作序列。若某 episode 在某步失败，返回反例前缀。
5. **反例处理**：将反例的所有前缀加入 S，回到步骤 2。
6. **收敛**：连续 `stabil_required=2` 次等价查询无反例后，进行**节点合并**（Union-Find 合并"任务阶段相同"的冗余节点），最后剪枝不可达节点和空边。

**Replay Assign 机制**：构建完成后，所有 episode 按 PMM 的 delta 函数重新"回放"——每步确定当前节点 q、输入符号 x、目标节点 q'，并将对应动作存入边缓存 `_qx_actions[(q, x)]`。多边消歧使用 RNN 嵌入余弦相似度。

#### 3.3.2 Residual 网络训练

**目的**：PMM 的 `predict(q, x)` 只能给出一条边上所有动作的均值，精度不够。Residual 网络学习"从 PMM 动作原型到实际最优动作的残差修正"。

**模型架构** (`ResidualMLP`)：

| 组件 | 描述 |
|------|------|
| `query_proj` | `Linear(feat_dim, 32)`，将观测差异映射为注意力 query |
| `key_proj` | `Linear(a_dim, 32)`，将动作映射为注意力 key |
| `net` | `Linear(feat_dim + a_dim, 512) → ReLU → Linear(512, a_dim)`，残差网络主体 |
| `logstd` | 可学习的对数标准差参数 |

**前向传播**：

1. **计算 delta**：`delta = concat_feat - cluster_center`（当前观测特征与聚类中心的差异）
2. **Prototype 选择**（Straight-Through Attention）：从 PMM 缓存的动作列表 `action_list` 中，用注意力机制（query=delta, key=action）选出一个"原型动作"（prototype），同时计算 AABB 边界。
3. **残差修正**：`raw_action = MLP(delta ∥ prototype)`
4. **边界裁剪**：将输出 clamp 到 `[aabb_min - ε, aabb_max + ε]`，确保预测动作不偏离 PMM 观测到的动作范围太远。

**训练细节**：

- 损失函数：负对数似然 `-log_prob(true_action)`，使用高斯分布。
- 数据集：全量 episode（90% train / 10% eval），每步提取 `(current_q, state_idx, concat_feat, action_true)`。
- 训练参数：`epochs=400, batch_size=512, lr=3e-4`，CosineAnnealing + weight_decay。
- 训练时对 `concat_feat` 添加 0.01 标准差的高斯噪声作为数据增强。

**输出**：

| 文件 | 内容 |
|------|------|
| `results/checkpoints/residual_net.pt` | `{model: state_dict, norm_stats: {...}}` |
| `results/pmm_residual_training.pkl` | PMM 完整存档（拓扑 + 权重 + episodes） |
| `results/pmm_residual_training.dot` | PMM 自动机 Graphviz 可视化 |

**此时 p1（general policy）= PMM + ResidualMLP + RNN 组合**。

---

### 3.4 阶段 4：PMM × Büchi 乘积构造（`scripts/train/production_pmm_ba.py`）

**目的**：通过与 Büchi 自动机的乘积操作，将 general policy p1 细化为满足特定偏好约束的 policy p2。

#### 3.4.1 Büchi 自动机（偏好约束）

偏好以 Büchi 自动机（BA）的形式给出，存储在 `results/preference/` 下的 DOT 文件中。例如：

- **`not_c2.dot`**：接受除了 c2 以外的所有符号的自循环 → 含义：**轨迹中不能经过聚类 c2 对应的任务阶段**。
- **`not_c6.dot`**：接受除了 c6 以外的所有符号 → 含义：**轨迹中不能经过聚类 c6 对应的任务阶段**。

Büchi 自动机的语义是：一条轨迹的状态符号序列必须被 BA 接受（最终状态在接受集中），否则该轨迹不满足偏好。

#### 3.4.2 乘积构造（两阶段）

**Phase 1 — 高层骨架构建** (`build_product()`)：

1. 解析 BA DOT 文件，提取状态集、初始状态、接受状态集、转移函数。
2. 计算 **字母表交集**：PMM 的 sigma（观测到的符号集） ∩ BA 的符号集。
3. 构造乘积图：节点为 `(q_mealy, q_buchi)` 对；对于每条 PMM 边 `(q_m, x) → q_m'` 和 BA 边 `(q_b, x) → q_b'`，创建乘积边 `((q_m, q_b), x) → (q_m', q_b')`。
4. BFS 剪枝：从初始乘积状态出发，只保留可达的乘积节点。
5. 重映射为连续整数索引（初始状态保证为 0）。

**Phase 2 — Replay Assign** (`replay_on_product()`)：

1. 遍历 PMM 中存储的所有 episode，按乘积图的 delta 函数逐步行走。
2. **整条 episode 被拒绝的条件**：
   - 某步在乘积图中无可用转移（`no_transition`）
   - 走完整条 episode 后，最终乘积状态不在 BA 的接受集中（`not_accepting`）
3. 对通过验证的 episode，记录其动作到乘积边的缓存中。
4. 将乘积拓扑 + 缓存设置到 PMM 对象上，执行剪枝。

**输出**：

| 文件 | 内容 |
|------|------|
| `results/product/product_pmm_<ba_name>.pkl` | 乘积 PMM 完整存档 |
| `results/product/product_pmm_<ba_name>.dot` | 乘积 PMM Graphviz 可视化 |

**此时 p2（偏好 policy）= 乘积 PMM + 同一个 ResidualMLP + RNN**。乘积操作只改变了 PMM 的拓扑结构和动作缓存，Residual 网络权重不变。

---

### 3.5 阶段 5：策略评估（`scripts/eval/pmm_eval.py`）

**目的**：在 ManiSkill StackCube-v1 环境中运行 p1 或 p2，统计成功率和堆叠方向偏好。

**PMMResidualAgent 推理流程**（`agent/pmm_agent.py`）：

```
obs (RGB + state)
    │
    ▼
NatureCNN.extractors["rgb"] ──► rgb_raw (256-d)
NatureCNN.extractors["state"] ► pos_raw (256-d)
    │
    ▼  Z-score 归一化
rgb_features, pos_features
    │
    ▼  拼接 + 最近邻检索
concat_feat ──cdist──► state_idx（离散状态符号）
    │
    ▼  PMM 查询
pmm.predict_list(current_q, state_idx) ──► action_list
    │
    ▼  ResidualMLP
ResidualMLP(action_list, cluster_center, concat_feat) ──► action_mean
    │
    ▼  RNN step（更新隐状态）
rnn.forward_step(action, state_idx, h_prev) ──► h_new
    │
    ▼  PMM 节点转移（余弦消歧）
current_q = disambiguate(delta[(q, x)], h_new, rep_embeddings)
    │
    ▼
输出 action 给环境
```

**评估设置**：

| 参数 | 值 |
|------|-----|
| 环境 | `StackCube-v1`，固定方块位姿（CubeA: (0, -0.175), CubeB: (0, 0.2)） |
| episode 数 | 64（可配置） |
| 最大步数 | 80 |
| 状态噪声 | 初始 obs 加 σ=0.1 噪声，后续每步加 σ=0.01 噪声 |
| PPO encoder | 加载外部预训练的 PPO feature_net 权重，推理时冻结 |

**评估指标**：

- **成功率**（Success Rate）：episode 中 `success_once` 为真的比例。
- **堆叠方向**（Stacking Direction）：成功 episode 中红块在上（red_on_green）或绿块在上（green_on_red）的比例。

**切换 p1/p2**：只需修改 `--pmm-pkl` 参数：
- p1：`--pmm-pkl results/pmm_residual_training.pkl`
- p2：`--pmm-pkl results/product/product_pmm_not_c6.pkl`（或其他乘积 PMM）

---

## 4. 核心概念速查

### 4.1 状态符号（State Symbol）

聚类产生的离散标签 `c0, c1, ..., c8`，每个符号对应机械臂在执行堆叠任务时的一个特定阶段（如接近、抓取、抬起、移动、放置等）。符号序列构成了 PMM 的输入字母表。

### 4.2 PMM 节点（Node）

自动机的内部状态，表示"当前任务进度"。每个节点关联一个代表性前缀（rep）和一个 RNN 嵌入向量。节点数远少于符号数——多个连续相同符号的自循环会被合并为同一节点。

### 4.3 消歧机制（Disambiguation）

当 PMM 在节点 q 读入符号 x 后有多条出边时，使用 RNN 当前隐状态与各候选目标节点的 `rep_embedding` 的余弦相似度来选择最匹配的转移。

### 4.4 Residual 修正

PMM 的 `predict_list(q, x)` 返回该边上所有历史动作。ResidualMLP 通过注意力机制从中选取原型动作，并基于当前观测偏差（delta = 观测特征 - 聚类中心）生成精确的残差修正。

### 4.5 Büchi 乘积

将 PMM 的状态空间与 Büchi 自动机的状态空间做笛卡尔积，只保留同时满足 PMM 转移和 BA 转移的边。效果是**从 PMM 中"剔除"不满足偏好约束的行为路径**，剩余路径组成偏好策略 p2。

---

## 5. 命令行快速参考

```bash
# 阶段 1：数据预处理
python data/preprocess.py --pkl <轨迹pkl路径> --output-pkl data/episodes_with_states.pkl

# 阶段 2：RNN 预训练
python scripts/train/rnn_train.py --pkl data/episodes_with_states.pkl

# 阶段 3：PMM 构建 + Residual 训练（自动加载 RNN 权重）
python scripts/train/residual_train.py \
    --input-pkl data/episodes_with_states.pkl \
    --rnn-weights results/checkpoints/rnn_pretrained.pt

# 阶段 4：乘积构造（偏好细化）
python scripts/train/production_pmm_ba.py \
    --pmm-pkl results/pmm_residual_training.pkl \
    --ba-dot results/preference/not_c6.dot

# 阶段 5：评估（修改 --pmm-pkl 切换 p1/p2）
python scripts/eval/pmm_eval.py \
    --pmm-pkl results/product/product_pmm_not_c6.pkl \
    --residual-weights results/checkpoints/residual_net.pt \
    --rnn-weights results/checkpoints/rnn_pretrained.pt \
    --traj-pkl data/episodes_with_states.pkl
```

---

## 6. 文件依赖关系图

```
trajectories.pkl（原始数据）
    │
    ├──► data/preprocess.py
    │       └──► data/episodes_with_states.pkl
    │               │
    │               ├──► scripts/train/rnn_train.py
    │               │       └──► results/checkpoints/rnn_pretrained.pt
    │               │
    │               └──► scripts/train/residual_train.py
    │                       ├──► results/pmm_residual_training.pkl  ←── p1 的 PMM
    │                       ├──► results/pmm_residual_training.dot
    │                       └──► results/checkpoints/residual_net.pt
    │
    ├──► results/preference/not_c2.dot  （手工编写的偏好 BA）
    ├──► results/preference/not_c6.dot
    │
    └──► scripts/train/production_pmm_ba.py
            ├── 读取: pmm_residual_training.pkl + not_c*.dot
            └──► results/product/product_pmm_not_c*.pkl  ←── p2 的 PMM
                 results/product/product_pmm_not_c*.dot

外部依赖:
    PPO encoder 权重 (.pt) ──► scripts/eval/pmm_eval.py（仅评估时需要）
```

---

## 7. 关键超参数汇总

| 阶段 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 预处理 | PCA n_components | 5 | 每种模态的 PCA 目标维度 |
| 预处理 | HDBSCAN min_cluster_size | max(10, 5%×N) | 最小聚类大小 |
| RNN | h_dim | 64 | RNN 隐层维度 |
| RNN | state_embed_dim | 16 | 状态嵌入维度 |
| RNN | epochs / steps | 32 / 128 | 训练轮次 |
| PMM | cos_tau_row | 0.6 | 余弦相似度阈值（节点归属判断） |
| PMM | error_threshold | 0.3 | 等价查询的动作误差容限 |
| PMM | max_inner_iters | 1000 | L* 最大迭代次数 |
| PMM | stabil_required | 2 | 连续稳定次数后收敛 |
| Residual | hidden | 256 | MLP 隐层宽度（实际第一层 512） |
| Residual | d_attn | 32 | 注意力投影维度 |
| Residual | tau | 0.1 | Softmax 温度 |
| Residual | epochs | 400 | 训练轮次 |
| 评估 | max_steps | 80 | 每个 episode 最大步数 |
| 评估 | num_episodes | 64 | 评估 episode 总数 |
