# StackCubeFixed_Dual 流水线文档

> 本文档面向 AI Agent 阅读，详细描述 `ENAP_bs/StackCubeFixed_Dual` 仓库的完整工作流程。
> 任务场景：ManiSkill StackCube-v1（机械臂堆叠方块）— **Dual（对偶）** 公式化。
>
> **Dual 核心思想**：原版 ENAP 对**观测特征**做聚类，PMM 以聚类符号为输入、边上存储**动作**；
> Dual ENAP 反转这一关系——对**动作**做聚类，PMM 以动作聚类符号为输入、边上存储**观测特征**。

---

## 1. 仓库结构总览

```
StackCubeFixed_Dual/
├── agent/                          # 核心模型与工具
│   ├── pmm_class.py                # Dual PMM 类定义（动作聚类输入 → 特征输出）
│   ├── pmm_agent.py                # Dual ResidualMLP 网络定义
│   └── utils.py                    # NatureCNN、特征归一化、数据加载
├── data/
│   ├── preprocess.py               # 第一步：Dual 数据预处理（构建对偶 episode + 动作聚类）
│   └── cluster_sequences.jsonl     # 参考数据（每条 episode 的动作模式序列）
├── results/
│   └── pmm_residual_training.dot   # PMM 自动机可视化（训练后的 dual policy）
└── scripts/
    └── train/
        ├── rnn_train.py            # 第二步：RNN 预训练（Dual 序列编码器）
        └── residual_train.py       # 第三步-B：PMM 构建 + Residual 网络训练
```

> [!NOTE]
> **与原版 (StackCubeFixed) 的结构差异：**
>
> | 原版有但 Dual 没有 | 说明 |
> |---|---|
> | `scripts/eval/pmm_eval.py` | Dual 暂无评估脚本 |
> | `scripts/train/production_pmm_ba.py` | Dual 暂无 Büchi 乘积构造 |
> | `results/preference/` | Dual 暂无偏好 Büchi 自动机 |
> | `results/product/` | Dual 暂无乘积 PMM 结果 |


---

## 2. 完整流水线

整体流程分为 3 个阶段，产出一个 dual policy（PMM + ResidualMLP + RNN 的组合）。

```
原始轨迹 pkl
    │
    ▼
[阶段 1] preprocess.py ─────► episodes_with_states.pkl（含动作聚类标签 + 聚类中心 + 归一化统计量）
    │
    ▼
[阶段 2] rnn_train.py ──────► rnn_pretrained.pt（Dual RNN 序列编码器权重）
    │
    ▼
[阶段 3] residual_train.py ─► PMM 自动机 + residual_net.pt ── 组合即为 dual policy
```

> [!NOTE]
> **与原版流水线的差异：**
> - 原版有 5 个阶段（含偏好细化 + 评估），Dual 目前只有 3 个阶段。
> - 原版的阶段 3 将 PMM 构建和 Residual 训练耦合在同一脚本中。

---

## 3. 各阶段详解

### 3.1 阶段 1：Dual 数据预处理（`data/preprocess.py`）

**输入**：ManiSkill 采集的原始轨迹 pkl 文件（每条轨迹的每一步包含 `rgb_feature`、`pos_feature`、`action`、可选 `rgb` 图像）。

**处理步骤**：

1. **构建对偶 episode（Dual Episode）**：将原始 `(f_t, a_t)` 对转换为 `(a_t, f_{t+1})` 对。具体地，对于长度为 T 的原始 episode：
   - `dual_step_0`：**dummy action**（= `mean + 10×std`，远离正常动作分布） + `f_0`（第 0 步的真实特征）
   - `dual_step_t`（t = 1..T-1）：`a_{t-1}`（上一步的真实动作） + `f_t`（当前步的真实特征）
   - `dual_step_T`：`a_{T-1}`（最后一步的真实动作） + **dummy feature**（= `mean + 10×std`，远离正常特征分布）
2. **动作 Z-score 标准化**：仅对**真实动作**（非 dummy）进行零均值单位方差归一化。
3. **HDBSCAN 动作聚类**：在归一化后的 8 维动作空间上直接运行 HDBSCAN（`min_cluster_size=max(10, 6%×N)`, `min_samples=max(2, 30%×min_cluster_size)`），将所有 timestep 的动作分配到若干动作模式（action mode）。
4. **状态标注**：根据动作聚类结果 + dummy 标签（`-2`），为每个 dual step 添加 one-hot 编码的 `state` 向量（维度 = 聚类数 + noise + dummy）。
5. **聚类中心计算**：在归一化动作空间中计算每个动作聚类的质心；dummy 聚类中心设为全 10.0 向量。
6. **组合特征计算**：对每步的 `rgb_feature` 和 `pos_feature` 分别做 Z-score 归一化并 clamp 到 ±5.0，然后拼接为 `combined_feature`。

**输出** (`episodes_with_states.pkl`)：

```python
{
    "episodes": [...],                  # dual episodes，每步含 state (one-hot)、combined_feature、action 等
    "num_clusters": int,                # 纯动作聚类数（不含 noise 和 dummy）
    "num_states": int,                  # 总状态数（含 noise + dummy）
    "cluster_centers": np.array,        # shape (num_states, action_dim)，在归一化动作空间中
    "cluster_labels": list,             # 各聚类的标签列表（含 dummy_action_label = -2）
    "action_norm_stats": {"mean", "std"},  # 动作归一化统计量
    "feature_norm_stats": {"rgb_mean", "rgb_std", "pos_mean", "pos_std"},  # 特征归一化统计量
    "dummy_action_label": -2,           # dummy 动作的标签值
    "metadata": ...                     # 如存在则透传
}
```

**关键设计意图**：与原版对**观测特征**做聚类以离散化状态空间不同，Dual 对**动作**做聚类以离散化动作空间。每个动作模式符号（如 m0, m1, ..., m5）对应一种特定的动作模式（例如"向前推"、"抬起"、"旋转"等）。对偶 episode 的构建确保了每一步都有 `(动作模式符号, 下一步特征)` 这样的配对，使 PMM 能以动作模式为输入、特征为输出来建模。

> [!NOTE]
> **与原版 `preprocess.py` 的差异：**
>
> | 方面 | 原版 | Dual |
> |------|------|------|
> | **聚类对象** | 观测特征（RGB + POS） | 动作（8维） |
> | **聚类前降维** | PCA(5D) × 2 → 拼接为 10D | 无 PCA，直接在 8D 归一化动作上聚类 |
> | **HDBSCAN 参数** | `min_cluster_size = max(10, 5%×N)` | `min_cluster_size = max(10, 6%×N)`, `min_samples = max(2, 30%×min_cluster_size)` |
> | **Dummy 机制** | 无 dummy | 有 dummy action（episode 首步）和 dummy feature（episode 末步）|
> | **Episode 结构** | 原始 `(f_t, a_t)` 序列 | 对偶 `(a_t, f_{t+1})` 序列，长度比原始多 1 |
> | **额外输出** | 无 | `action_norm_stats`、`feature_norm_stats`、`dummy_action_label`、`combined_feature` |
> | **聚类中心空间** | 原始拼接特征空间（`RGB_norm ∥ POS_norm`） | 归一化动作空间 |
> | **符号命名** | `c0, c1, ...`（观测聚类） | `m0, m1, ...`（动作模式） |
> | **可视化** | PCA 散点图 + RGB 图像网格 | PCA 散点图 + 动作箭头图（每个聚类的动作方向） |

---

### 3.2 阶段 2：Dual RNN 预训练（`scripts/train/rnn_train.py`）

**输入**：`episodes_with_states.pkl`

**目的**：训练一个 RNN 序列编码器，学习特征-动作模式序列的隐表征，供后续 PMM 构建时用于节点消歧（disambiguation）。

**模型架构** (`Pretrain`)：

| 组件 | 描述 |
|------|------|
| `state_embed` | `nn.Embedding(state_dim, 16)`，将离散动作模式索引映射为 16 维嵌入 |
| `feat_proj` | `nn.Linear(feat_dim, 64)`，将高维组合特征投影到 64 维 |
| `enc` | `nn.RNN(64 + 16, 128)`，输入为 feature 投影与 state embedding 的拼接 |
| `feat_head` | `nn.Linear(128, feat_dim)`，预测下一步特征 |
| `cls_head` | `nn.Linear(128, state_dim)`，预测下一步动作模式 |

**训练细节**：

- **多任务联合损失**：
  - `loss_feat`：MSE 特征预测损失
  - `loss_cls`：CrossEntropy 动作模式分类损失
  - `loss_cont`：Phase-Aware Contrastive Loss（同动作模式步拉近隐状态，跨动作模式步推远隐状态）
  - 总损失 = `λ_f * loss_feat + λ_s * loss_cls + λ_c * loss_cont`
- **优先经验回放**（PrioritizedReplayBuffer）：按损失大小分配采样优先级。
- 训练参数：`batch_size=512, epochs=32, steps_per_epoch=128, lr=3e-4`，CosineAnnealing 调度。

**输出** (`results/checkpoints/rnn_pretrained.pt`)：

```python
{
    'model_state': state_dict,           # RNN 权重（含 feat_proj）
    'dims': {'f': F_dim, 's': S_dim, 'e': 16, 'h': 128}  # 维度元信息
}
```

**关键设计意图**：RNN 的隐状态编码了"到当前步为止的特征-动作模式历史"。在 PMM 学习过程中，通过比较当前 RNN 隐状态与各候选目标节点的代表性嵌入（representative embedding）的余弦相似度来选择正确的转移（消歧机制）。

> [!NOTE]
> **与原版 `rnn_train.py` 的差异：**
>
> | 方面 | 原版 | Dual |
> |------|------|------|
> | **RNN 输入** | `concat(action_t, embed(obs_cluster_t))` | `concat(feat_proj(feature_t), embed(action_cluster_t))` |
> | **额外投影层** | 无（action 直接与 embedding 拼接） | `feat_proj = Linear(feat_dim, 64)`，因为特征维度远高于动作维度，需要先降维 |
> | **隐层维度** | `h_dim = 64` | `h_dim = 128`（翻倍，以适应更高维的输入信息） |
> | **预测头** | `act_head` 预测动作 | `feat_head` 预测特征 |
> | **输入序列** | `(action, state)` 对的序列 | `(combined_feature, state)` 对的序列 |
> | **checkpoint dims** | `{'a': A_dim, 's': S_dim, 'e': 16, 'h': 64}` | `{'f': F_dim, 's': S_dim, 'e': 16, 'h': 128}` |

---

### 3.3 阶段 3：PMM 构建 + Residual 网络训练

直接运行 `residual_train.py`（不指定 `--pmm-pkl`），一次性完成 PMM 构建和 Residual 训练。

#### 3.3.1 PMM 自动机学习（`agent/pmm_class.py` → `PMM.learn_pmm()`）

Dual PMM 是一个概率性 Mealy 机，核心结构为：

```python
{
    'Q': [0, 1, ..., N-1],          # 节点集合（每个节点对应任务的一个子阶段）
    'delta': {(q, x): {q': prob}},  # 转移函数：在节点 q 读入动作模式 x 后以概率 prob 转移到 q'
    'reps': [tuple, ...],           # 每个节点的代表性前缀序列（动作模式符号组成）
    'rep_embeddings': [np.array],   # 每个节点的 RNN 代表性嵌入向量
}
```

**学习算法**：基于改进的 L* 算法（自动机学习），迭代过程与原版一致：

1. **初始化**：前缀集 `S = [()]`（空序列）。
2. **闭合性检查**（Closedness Check）：对 S 中每个前缀的 RNN 嵌入，检查其后继状态是否都能被现有节点覆盖（余弦相似度 ≥ `cos_tau_row=0.6`），如不能则添加新前缀。
3. **假设构建**（`_build_pmm`）：基于当前 S，用 RNN 嵌入数据库将所有 timestep 映射到节点，统计转移概率。
4. **等价性查询**（Equivalence Query）：NFA 风格检验——遍历所有 episode，检查假设自动机能否以合理误差再现实际特征序列。
5. **反例处理**：将反例的所有前缀加入 S，回到步骤 2。
6. **收敛**：连续 `stabil_required=2` 次等价查询无反例后，进行节点合并（Union-Find），最后剪枝不可达节点和空边。

**Replay Assign 机制**：构建完成后，所有 episode 按 PMM 的 delta 函数重新"回放"——每步确定当前节点 q、输入符号 x、目标节点 q'，并将对应**特征**存入边缓存 `_qx_features[(q, x)]`。多边消歧使用 RNN 嵌入余弦相似度。

> [!NOTE]
> **与原版 `pmm_class.py` 的差异：**
>
> | 方面 | 原版 | Dual |
> |------|------|------|
> | **输入字母表** | 观测聚类符号 `c0, c1, ...` | 动作模式符号 `m0, m1, ...` |
> | **边上存储的数据** | 动作 `A`（`_qx_actions`） | 特征 `F`（`_qx_features`） |
> | **Episode 内部表示** | `{'S': ..., 'A': ...}` | `{'S': ..., 'F': ...}` |
> | **RNN 架构** | `TrainableRNN(action_dim, state_dim, embed_dim, h_dim)`，无投影层 | `TrainableRNN(feat_dim, state_dim, embed_dim, h_dim, proj_dim=64)`，含 `feat_proj` 投影层 |
> | **RNN 权重加载** | 映射 `enc.*` → `rnn.*` | 额外映射 `feat_proj.*` |
> | **RNN 隐层维度** | 64 | 128 |
> | **error_threshold** | 0.3（8D 动作空间中的 L2 距离） | 20.0（512D 特征空间中的 L2 距离，量级远大于动作空间） |
> | **等价查询比较对象** | 比较 `true_action` 与边上存储的动作 | 比较 `true_feature` 与边上存储的特征 |
> | **DOT 符号前缀** | `c`（如 `c5`） | `m`（如 `m5`） |
> | **DOT 边标签** | 显示 avg action 的前 2 维值 | 显示 avg feature 的 L2 norm |
> | **持久化 payload key** | `'a_dim'` | `'f_dim'` |

#### 3.3.2 Residual 网络训练（`scripts/train/residual_train.py`）

**目的**：PMM 的 `predict(q, x)` 只能给出一条边上所有特征的均值，精度不够。Residual 网络学习"从 PMM 特征原型到实际真实特征的残差修正"。

**模型架构** (`ResidualMLP`，定义在 `agent/pmm_agent.py`)：

| 组件 | 描述 |
|------|------|
| `query_proj` | `Linear(a_dim, 32)`，将动作偏差映射为注意力 query |
| `key_proj` | `Linear(f_dim, 32)`，将特征映射为注意力 key |
| `net` | `Linear(f_dim + a_dim, 1024) → ReLU → Linear(1024, 1024) → ReLU → Linear(1024, f_dim)`，残差网络主体 |
| `logstd` | 可学习的对数标准差参数（维度 = `f_dim`） |

**前向传播**：

1. **计算 delta**：`delta = actual_action_norm - action_cluster_center`（当前归一化动作与动作聚类中心的差异）
2. **Prototype 选择**（Straight-Through Attention）：从 PMM 缓存的特征列表 `feature_list` 中，用注意力机制（query=delta, key=feature）选出一个"原型特征"（prototype），同时计算 AABB 边界。
3. **残差修正**：`raw_feature = MLP(prototype ∥ delta) + prototype`
4. **边界裁剪**：将输出 clamp 到 `[aabb_min - ε, aabb_max + ε]`，确保预测特征不偏离 PMM 观测到的特征范围太远。

**训练细节**：

- 损失函数：负对数似然 `-log_prob(true_feature)`，使用高斯分布。
- 数据集：全量 episode（90% train / 10% eval），每步提取 `(current_q, state_idx, combined_feature, action_norm)`。
- 训练参数：`epochs=400, batch_size=512, lr=3e-4`，CosineAnnealing + weight_decay。
- 训练时对 `action_norm` 添加 0.01 标准差的高斯噪声作为数据增强。
- 训练过程中同时输出量化可视化（L2 Error、Cosine Similarity 曲线）和定性可视化（True/Predicted/Diff 热力图）。

**输出**：

| 文件 | 内容 |
|------|------|
| `results/checkpoints/residual_net.pt` | `{model: state_dict, action_norm_stats: {...}}` |
| `results/pmm_residual_training.pkl` | PMM 完整存档（拓扑 + 权重 + episodes） |
| `results/pmm_residual_training.dot` | PMM 自动机 Graphviz 可视化 |
| `results/plot/loss_curves.png` | 训练/验证损失曲线 |
| `results/plot/ep*_quantitative.png` | 每条测试 episode 的 L2 Error + Cosine Similarity 曲线 |
| `results/plot/ep*_qualitative.png` | 每条测试 episode 的 True/Predicted/Diff 特征热力图 |

> [!NOTE]
> **与原版 Residual 训练的差异：**
>
> | 方面 | 原版 | Dual |
> |------|------|------|
> | **预测目标** | 动作 `action`（8D） | 特征 `combined_feature`（512D） |
> | **注意力方向** | query = 观测偏差（`concat_feat - cluster_center`），key = 动作 | query = 动作偏差（`action_norm - cluster_center`），key = 特征 |
> | **边上列表** | `action_list`（动作列表） | `feature_list`（特征列表） |
> | **MLP 结构** | 2 层：`Linear(feat_dim + a_dim, 512) → ReLU → Linear(512, a_dim)` | 3 层：`Linear(f_dim + a_dim, 1024) → ReLU → Linear(1024, 1024) → ReLU → Linear(1024, f_dim)`，更深更宽以适应高维输出 |
> | **默认 hidden** | 256（实际第一层 512） | 1024 |
> | **数据增强噪声加在** | `concat_feat`（观测特征） | `action_norm`（归一化动作） |
> | **state_idx 回退策略** | 基于观测特征最近邻（`cdist` on feature space） | 基于动作聚类中心最近邻（`norm` on action space） |
> | **归一化统计量来源** | 从训练 episode 实时计算 `norm_stats` | 从 pkl 读取 `action_norm_stats`（由 preprocess 阶段预计算） |
> | **checkpoint 内容** | `{model, norm_stats}` | `{model, action_norm_stats}` |
> | **可视化** | PCA-1D 动作对比（True/PMM/Residual） | L2 + Cosine 曲线 + 特征热力图 |
> | **PMM 构建可选分离** | PMM 构建内嵌在 `residual_train.py` 中 |

---

## 4. 核心概念速查

### 4.1 动作模式符号（Action Mode Symbol）

动作聚类产生的离散标签 `m0, m1, ..., m5`，每个符号对应机械臂在执行堆叠任务时的一种特定动作模式（例如"伸展"、"抓取闭合"、"抬起"等）。符号序列构成了 PMM 的输入字母表。

> [!NOTE]
> **与原版的差异：** 原版的字母表符号 `c0, c1, ...` 来自观测特征聚类，代表任务的"状态阶段"；Dual 的字母表符号 `m0, m1, ...` 来自动作聚类，代表任务的"动作模式"。

### 4.2 PMM 节点（Node）

自动机的内部状态，表示"当前任务进度"。每个节点关联一个代表性前缀（rep）和一个 RNN 嵌入向量。节点数远少于符号数——多个连续相同符号的自循环会被合并为同一节点。

### 4.3 消歧机制（Disambiguation）

当 PMM 在节点 q 读入符号 x 后有多条出边时，使用 RNN 当前隐状态与各候选目标节点的 `rep_embedding` 的余弦相似度来选择最匹配的转移。机制与原版完全一致。

### 4.4 Residual 修正

PMM 的 `predict_list(q, x)` 返回该边上所有历史特征。ResidualMLP 通过注意力机制从中选取原型特征，并基于当前动作偏差（delta = 归一化动作 - 动作聚类中心）生成精确的残差修正。

> [!NOTE]
> **与原版的差异：** 原版的 Residual 修正方向是 `观测偏差 → 动作修正`；Dual 的方向是 `动作偏差 → 特征修正`。所有"输入/输出"角色完全对调。

### 4.5 对偶 Episode（Dual Episode）

Dual 的核心数据结构。通过将原始 `(f_t, a_t)` 配对关系移位为 `(a_t, f_{t+1})`，并在首步添加 dummy action、末步添加 dummy feature，使得每一步都有 `(动作, 特征)` 的对偶配对。Dummy 机制确保 episode 边界信息不丢失。

---

## 5. 命令行快速参考

```bash
# 阶段 1：Dual 数据预处理
python data/preprocess.py --pkl <轨迹pkl路径> --output-pkl data/episodes_with_states.pkl

# 阶段 2：Dual RNN 预训练
python scripts/train/rnn_train.py --pkl data/episodes_with_states.pkl

# 阶段 3：PMM 构建 + Residual 训练（一体化）
python scripts/train/residual_train.py \
    --input-pkl data/episodes_with_states.pkl \
    --rnn-weights results/checkpoints/rnn_pretrained.pt


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
    │                       ├──► results/pmm_residual_training.pkl
    │                       ├──► results/pmm_residual_training.dot
    │                       ├──► results/checkpoints/residual_net.pt
    │                       └──► results/plot/*.png
```

---

## 7. 关键超参数汇总

| 阶段 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 预处理 | HDBSCAN min_cluster_size | max(10, 6%×N) | 最小聚类大小 |
| 预处理 | HDBSCAN min_samples | max(2, 30%×min_cluster_size) | 最小核心样本数 |
| 预处理 | FEATURE_CLAMP | 5.0 | combined_feature 的归一化后 clamp 范围 |
| RNN | h_dim | 128 | RNN 隐层维度 |
| RNN | state_embed_dim | 16 | 状态嵌入维度 |
| RNN | feat_proj_dim | 64 | 特征投影维度 |
| RNN | epochs / steps | 32 / 128 | 训练轮次 |
| PMM | cos_tau_row | 0.6 | 余弦相似度阈值（节点归属判断） |
| PMM | error_threshold | 20.0 | 等价查询的特征误差容限（远大于原版的 0.3） |
| PMM | max_inner_iters | 1000 | L* 最大迭代次数 |
| PMM | stabil_required | 2 | 连续稳定次数后收敛 |
| Residual | hidden | 1024 | MLP 隐层宽度 |
| Residual | d_attn | 32 | 注意力投影维度 |
| Residual | tau | 0.1 | Softmax 温度 |
| Residual | epochs | 400 | 训练轮次 |

> [!NOTE]
> **超参数与原版的关键差异：**
> - `h_dim`: 64 → 128（RNN 容量翻倍）
> - `error_threshold`: 0.3 → 20.0（因特征空间维度远高于动作空间，L2 距离量级不同）
> - `hidden` (Residual MLP): 256 → 1024（预测目标从 8D 动作变为 512D 特征，需要更大的网络容量）
> - 预处理阶段无 PCA 降维（原版有 PCA(5D) × 2 = 10D）
