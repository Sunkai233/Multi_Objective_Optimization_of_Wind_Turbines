# TimeXer 文件夹4  timeXer网络构建

## SelfAttention_Family.py

你贴的这段代码是关于 **多种注意力机制（Attention Mechanisms）和两阶段注意力层（Two-Stage Attention Layer, TSA）的实现**，它通常用于 **时序模型（Transformer类）**，尤其是长时间序列预测。下面我给你做一个 **模块逐层解析**：

------

#### 1️⃣ DSAttention（De-stationary Attention）

```
class DSAttention(nn.Module):
```

- **核心思想**：对注意力分数进行**非平稳性修正**

- 输入：

  - `queries, keys, values`：标准 QKV
  - `tau, delta`：可学习的非平稳缩放和偏移参数

- 公式：
  $$
  \text{scores} = QK^T \cdot \tau + \delta
  $$

- 支持：

  - `mask_flag` → 是否使用下三角 causal mask
  - `output_attention` → 是否返回 attention map

- 输出：

  - 注意力加权值 `V`，可选 attention map `A`

------

#### 2️⃣ FullAttention（标准全注意力）

```
class FullAttention(nn.Module):
```

- 标准 Transformer Attention

- 与 `DSAttention` 类似，但没有非平稳修正

- 公式：
  $$
  \text{scores} = QK^T
  $$

- 支持 dropout、mask、输出 attention map

------

#### 3️⃣ ProbAttention（稀疏注意力）

```
class ProbAttention(nn.Module):
```

- **核心思想**：通过随机采样 Top-k Queries 计算注意力（节省计算）
- 核心步骤：
  1. `_prob_QK` → 只计算 top u queries 的注意力分数
  2. `_get_initial_context` → 初始化 context
  3. `_update_context` → 将 top-k attention 更新到 context
- 优势：适合 **长序列注意力（L > 1024）**
- 输出：
  - 稀疏注意力加权值 `context`，可选 attention map `attn`

------

#### 4️⃣ AttentionLayer（多头注意力封装）

```
class AttentionLayer(nn.Module):
```

- 将任意注意力机制封装为 **多头注意力**
- 内部：
  - `query_projection` / `key_projection` / `value_projection`
  - 多头拆分 → 调用 `inner_attention`
  - 输出投影回原维度
- 支持 `tau, delta` → 可用于 DSAttention

------

#### 5️⃣ ReformerLayer

```
class ReformerLayer(nn.Module):
```

- 基于 Reformer 的 LSH Attention（局部敏感哈希）
- 优点：减少长序列注意力计算复杂度
- 方法：
  - `fit_length` → 补齐序列长度到 `bucket_size * 2` 的倍数
  - 调用 `LSHSelfAttention`
- 输出：加权序列，无 attention map

------

#### 6️⃣ TwoStageAttentionLayer（TSA）

```
class TwoStageAttentionLayer(nn.Module):
```

- **分两阶段注意力**：
  1. **时间维度注意力（Cross Time Stage）**
     - 对每个时间段进行标准多头注意力
     - 输入 `x` 维度 `[batch, D, L, d_model]` → `[batch*D, L, d_model]`
     - 包含残差 + LayerNorm + MLP
  2. **维度间注意力（Cross Dimension Stage）**
     - 学习少量 `router` 参数 → 汇聚维度间信息
     - `dim_sender` 和 `dim_receiver` 形成 D→D 消息传递
     - 同样包含残差 + LayerNorm + MLP
- 输出 `final_out` 维度 `[batch, ts_d, seg_num, d_model]`

------

#### 7️⃣ 总结

这段代码实际上是 **高级 Transformer 注意力模块**：

| 模块                   | 作用                                     |
| ---------------------- | ---------------------------------------- |
| DSAttention            | 对非平稳序列进行加权                     |
| FullAttention          | 标准全注意力                             |
| ProbAttention          | 长序列稀疏注意力                         |
| AttentionLayer         | 多头封装，可通用不同 attention           |
| ReformerLayer          | LSH 注意力，减少长序列计算               |
| TwoStageAttentionLayer | 两阶段注意力：时间 + 维度 → 强化特征交互 |

## Embed.py

你贴的这一段代码是各种 **时间序列数据嵌入（Embedding）模块** 的实现，它们是用于 Transformer/深度学习模型的输入编码。下面我帮你整理每个类的作用和原理：

------

#### 1️⃣ `PositionalEmbedding`

```
class PositionalEmbedding(nn.Module):
```

- **作用**：为序列中每个时间步增加位置信息
- **原理**：
  - 使用正弦和余弦函数编码位置（Transformer 原始做法）
  - 输入 `[batch, seq_len, d_model]`，返回 `[1, seq_len, d_model]` 的位置向量
- **用途**：
  - 告诉模型每个时间步的相对位置，避免序列顺序信息丢失

------

#### 2️⃣ `TokenEmbedding`

```
class TokenEmbedding(nn.Module):
```

- **作用**：对输入特征序列做卷积编码
- **原理**：
  - 一维卷积 `Conv1d` 将每个时间序列映射到 `d_model` 维
  - 保留局部时序信息（kernel_size=3）
- **用途**：
  - 将原始数值特征嵌入到向量空间，作为 Transformer 的输入

------

#### 3️⃣ `FixedEmbedding` & `TemporalEmbedding`

```
class FixedEmbedding(nn.Module):
class TemporalEmbedding(nn.Module):
```

- **FixedEmbedding**：
  - 使用固定正弦余弦编码，把类别特征映射到向量空间
- **TemporalEmbedding**：
  - 对时间特征（小时、星期、日期、月份等）做嵌入
  - `embed_type='fixed'` 用 `FixedEmbedding`，否则用可训练的 `nn.Embedding`
- **用途**：
  - 将时间信息融入模型，使模型捕捉周期性规律

------

#### 4️⃣ `TimeFeatureEmbedding`

```
class TimeFeatureEmbedding(nn.Module):
```

- **作用**：线性投影时间特征到 `d_model`
- **适用场景**：
  - 当输入的是数值型时间特征，而不是类别（hour, day, month…）
- **区别**：
  - TemporalEmbedding 处理离散时间特征
  - TimeFeatureEmbedding 处理连续时间特征

------

#### 5️⃣ `DataEmbedding` 系列

```
class DataEmbedding(nn.Module):
class DataEmbedding_inverted(nn.Module):
class DataEmbedding_wo_pos(nn.Module):
```

- **作用**：组合多种嵌入方式，得到最终输入特征向量
- **区别**：
  1. **DataEmbedding**：Token + Temporal + Positional
  2. **DataEmbedding_inverted**：直接用线性层投影原始序列，支持带时间标记的拼接
  3. **DataEmbedding_wo_pos**：不使用 PositionalEmbedding，只保留 Token + Temporal

------

#### 6️⃣ `PatchEmbedding`

```
class PatchEmbedding(nn.Module):
```

- **作用**：把序列切分成“patches”，然后做嵌入
- **原理**：
  1. 使用 `unfold` 切成长度为 `patch_len` 的片段
  2. 用线性层将每个 patch 映射到 `d_model`
  3. 添加位置编码和 dropout
- **用途**：
  - 类似 ViT 的 patch embedding，用于捕捉局部连续模式
  - 对长序列降维，减少计算量

------

#### 🔑 总结

| 类名                   | 作用                                             |
| ---------------------- | ------------------------------------------------ |
| PositionalEmbedding    | 提供序列位置编码                                 |
| TokenEmbedding         | 卷积编码原始特征序列                             |
| FixedEmbedding         | 固定类别编码                                     |
| TemporalEmbedding      | 时间维度特征编码（hour, day, month…）            |
| TimeFeatureEmbedding   | 连续时间特征线性映射                             |
| DataEmbedding          | 组合 Token + Temporal + Positional 嵌入          |
| DataEmbedding_inverted | 直接线性投影 + optional 时间特征                 |
| DataEmbedding_wo_pos   | Token + Temporal，不加位置编码                   |
| PatchEmbedding         | 序列切片 + 线性投影 + 位置编码（类似 ViT patch） |

## TimeXer.py

好的，我来给你做一个 **非常详细的分析**，把你的 `Model` 从数据输入到输出的每一步都拆开，并解释每个模块内部做了什么、形状如何变化、以及为什么这么设计。我们按流程逐步分析。

------

#### **1️⃣ 输入数据**

- 假设：
  - `x_enc`: `[B, seq_len, enc_in]` （输入序列）
  - `x_mark_enc`: `[B, seq_len, time_features]` （时间特征）
  - `x_dec`, `x_mark_dec`: 解码器输入，预测用（通常是历史真值或零填充）
- 目标：预测未来 `pred_len` 步的序列。

------

#### **2️⃣ Patch Embedding + 全局 Token（EnEmbedding）**

```
en_embed, n_vars = self.en_embedding(x_enc.permute(0,2,1))
```

### **作用**

1. **Patch 分割**：

   - 把序列按 `patch_len` 切成若干 patch

   - 形状变化：

     ```
     x_enc: [B, n_vars, seq_len]
     x.unfold(..., size=patch_len) -> [B, n_vars, patch_num, patch_len]
     ```

     - `patch_num = seq_len // patch_len`

2. **线性映射**：

   - 每个 patch 映射到 `d_model` 维度
   - `[B*n_vars, patch_num, patch_len] -> [B*n_vars, patch_num, d_model]`

3. **位置编码**：

   - 给每个 patch 加位置编码（类似 Transformer）
   - `[B*n_vars, patch_num, d_model] + positional_embedding`

4. **全局 token**：

   - 额外加一个 learnable token，用于捕捉 patch 间全局信息

   - 最终输出：

     ```
     en_embed: [B*n_vars, patch_num+1, d_model]
     ```

------

#### **3️⃣ 外部特征编码（DataEmbedding_inverted）**

```
ex_embed = self.ex_embedding(x_enc, x_mark_enc)
```

### **作用**

1. 对原始序列或附加时间特征进行线性映射

2. 输出：

   ```
   ex_embed: [B, seq_len, d_model]
   ```

   - 作为 **cross_attention** 的 `value` 和 `key`
   - 与全局 token 进行交互

------

#### **4️⃣ Encoder**

```
enc_out = self.encoder(en_embed, ex_embed)
```

**Encoder 层内部**

每一层 EncoderLayer：

1. **自注意力 (Self Attention)**：
   - 输入：`x`（patch embedding + global token）
   - 计算 patch 内依赖关系
   - `[B*n_vars, patch_num+1, d_model] -> [B*n_vars, patch_num+1, d_model]`
2. **全局 token 提取**：
   - `x_glb_ori = x[:, -1, :]` → `[B, 1, d_model]`
   - 用于跨序列注意力（cross_attention）
3. **Cross Attention**：
   - `x_glb` 与 `ex_embed` 交互
   - `[B, 1, d_model] -> [B, 1, d_model]`
   - 捕捉外部特征对全局信息的影响
4. **前馈卷积**：
   - 1D Conv + 非线性 + Dropout
   - `[B*n_vars, patch_num+1, d_model] -> [B*n_vars, patch_num+1, d_model]`
5. **残差 + LayerNorm**：
   - 每一步都用残差 + 归一化保持训练稳定

------

**Encoder 输出形状**

```
enc_out: [B*n_vars, patch_num+1, d_model]
```

- 重塑成 `[B, n_vars, patch_num+1, d_model]`
- `[batch, n_vars, d_model, patch_num+1]` 供 FlattenHead 使用

------

#### **5️⃣ FlattenHead**

```
dec_out = self.head(enc_out)
```

**作用**

1. 拉平每个变量的 patch 信息：

   ```
   x: [B, n_vars, d_model, patch_num+1] -> [B, n_vars, d_model*(patch_num+1)]
   ```

2. 线性映射到预测窗口：

   ```
   Linear(d_model*(patch_num+1) -> pred_len)
   ```

3. Dropout 防止过拟合

4. 输出：

   ```
   dec_out: [B, n_vars, pred_len]
   ```

5. 如果需要，把 `dec_out.permute(0, 2, 1)` → `[B, pred_len, n_vars]`，符合常规时间序列输出格式

------

#### **6️⃣ 非平稳序列归一化**

- **归一化（Normalization）**：

  ```
  means = x_enc.mean(1, keepdim=True)
  x_enc = x_enc - means
  stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True) + 1e-5)
  x_enc /= stdev
  ```

  - 保证输入序列在训练中稳定

- **反归一化（Denormalization）**：

  ```
  dec_out = dec_out * stdev + means
  ```

  - 预测结果回到原始尺度

------

#### **7️⃣ 多变量预测 (forecast_multi)**

- `forecast_multi` 支持多变量序列同时预测
- 对每个变量单独归一化、编码、然后一起预测
- 输出 `[B, pred_len, n_vars]`

------

#### **8️⃣ 总结数据流**

```
x_enc: [B, seq_len, n_vars] 
     │
     ├─> EnEmbedding (Patch + Global Token)
     │      [B*n_vars, patch_num+1, d_model]
     │
     ├─> DataEmbedding_inverted (外部特征编码)
     │      [B, seq_len, d_model]
     │
     ├─> Encoder (Self + Cross Attention)
     │      [B*n_vars, patch_num+1, d_model] -> reshape -> [B, n_vars, d_model, patch_num+1]
     │
     ├─> FlattenHead (拉平 + Linear)
     │      [B, n_vars, pred_len] -> permute -> [B, pred_len, n_vars]
     │
     └─> 可选 Denormalization
```

------

**关键特点**

1. **Patch + 全局 token** → 捕捉局部 + 全局依赖
2. **Cross Attention** → 引入外部特征信息
3. **非平稳处理** → 归一化 + 反归一化，稳定训练
4. **可多变量预测** → 支持多通道时间序列
5. **预测窗口灵活** → FlattenHead 可以自适应 patch_num