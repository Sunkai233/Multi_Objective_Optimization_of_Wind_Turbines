#  网络组件代码库讲解：

[DUET](https://github.com/decisionintelligence/DUET/tree/main)/[ts_benchmark](https://github.com/decisionintelligence/DUET/tree/main/ts_benchmark)/[baselines](https://github.com/decisionintelligence/DUET/tree/main/ts_benchmark/baselines)/[duet](https://github.com/decisionintelligence/DUET/tree/main/ts_benchmark/baselines/duet)/layers/

## 1 Autoformer_EncDec.py



这份代码是 **Autoformer 的编码器和解码器结构**，

核心思想是把时间序列拆成**趋势（trend）和季节性（seasonal）**，再结合注意力（attention）机制做预测。

📌 对应论文：

- **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**
   *AAAI 2022, Haixu Wu et al.*

这篇论文提出了：

- **序列分解模块（series decomposition）**：把序列拆成趋势 + 季节性；
- **渐进分解的 Encoder/Decoder**：在注意力和前馈网络之间不断去趋势；
- **Auto-Correlation 机制**（比普通 self-attention 更适合时间序列）。

------

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

引入 **PyTorch 模块**，用于构建神经网络。

------

### 1. 特殊的 LayerNorm

```python
class my_Layernorm(nn.Module):
    """
    特殊的 LayerNorm，用于季节性部分的归一化。
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)  # 普通的 LayerNorm
        # 在时间维度上取均值，作为 bias
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias
```

🔎 **解释**：
 LayerNorm 会对特征进行归一化，这里额外减去时间维度上的均值，确保**季节性成分围绕 0 波动**，突出周期性。

------

### 2. 移动平均（趋势提取）

```python
class moving_avg(nn.Module):
    """
    移动平均，用于提取时间序列的趋势部分。
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 一维平均池化，相当于滑动窗口平均
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 在序列两端补值，防止边界信息丢失
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # 池化操作（时间维度在中间，需要 permute 调整维度）
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
```

🔎 **解释**：
 输入 `x` 是形状 `[batch, time, channels]`。
 移动平均会得到**平滑趋势曲线**。

------

### 3. 序列分解（分成趋势 + 残差）

```python
class series_decomp(nn.Module):
    """
    时间序列分解：原序列 = 残差(季节性) + 趋势
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 提取趋势
        res = x - moving_mean             # 残差 = 季节性
        return res, moving_mean
```

------

### 4. 多尺度分解（FEDformer 提出的扩展）

```python
class series_decomp_multi(nn.Module):
    """
    多尺度分解：用不同窗口大小的移动平均来捕捉不同时间尺度的趋势。
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        # 多个分解结果取平均，得到更稳定的趋势和季节性
        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean
```

------

### 5. 编码器层（EncoderLayer）

```python
class EncoderLayer(nn.Module):
    """
    Autoformer 编码器层：
    1. 自注意力
    2. 残差 + 分解（去掉趋势，保留季节性）
    3. 前馈网络（两层卷积）
    4. 再分解
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # 前馈层用 1x1 卷积实现
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # 两次序列分解
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # (1) 自注意力
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        # (2) 第一次分解：去掉趋势
        x, _ = self.decomp1(x)
        # (3) 前馈网络
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # (4) 第二次分解
        res, _ = self.decomp2(x + y)
        return res, attn
```

------

### 6. 编码器（Encoder）

```python
class Encoder(nn.Module):
    """
    Autoformer 编码器：多个 EncoderLayer 堆叠。
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            # 最后一层只用注意力层
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
```

------

### 7. 解码器层（DecoderLayer）

```python
class DecoderLayer(nn.Module):
    """
    Autoformer 解码器层：
    1. 自注意力
    2. 交叉注意力（编码器信息）
    3. 前馈网络
    4. 多次分解：提取残差 + 累加趋势
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # 三次分解
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        # 投影层，用于输出趋势部分
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, 
                                    kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # (1) 自注意力
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        # (2) 交叉注意力（利用 Encoder 的输出 cross）
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        # (3) 前馈网络
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        # (4) 趋势累加，并用卷积投影到输出维度
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
```

------

### 8. 解码器（Decoder）

```python
class Decoder(nn.Module):
    """
    Autoformer 解码器：多个 DecoderLayer 堆叠。
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
```

- **趋势提取 (moving_avg, series_decomp)**：对应状态方程中的缓慢变化（低频部分），类似长期风速/负载趋势；
- **残差/季节性**：对应快速波动（高频部分），如功率振荡；
- **Encoder/Decoder 注意力机制**：建模不同时间、变量间的依赖关系；
- **Decoder 趋势预测**：通过残差+趋势叠加，得到未来的预测。



非常好的问题 👍 我来结合你给的输入形状 **[batch=32, seq_len=96, feature_dim=7]**，走一遍 **Autoformer 的处理流程**，并结合编码器、解码器的逻辑解释它是如何预测未来 24 步的。

------

### 🚀 输入输出设定

- **输入序列**：

  X∈R32×96×7X \in \mathbb{R}^{32 \times 96 \times 7}

  代表 32 个样本，每个样本长度 96，每个时间步有 7 个特征（例如：风速、功率、密度等）。

- **预测输出**：

  Y^∈R32×24×1\hat{Y} \in \mathbb{R}^{32 \times 24 \times 1}

  预测未来 24 步的目标变量（通常是功率/负载，维度可以是 1）。

------

### 🔎 编码器（Encoder）处理过程

1. **输入线性投影**（论文里有 embedding，这里省略细节）：
    把输入的 **7 维特征**投影到 `d_model` 维，比如 `d_model=512`，变成

   X∈R32×96×512X \in \mathbb{R}^{32 \times 96 \times 512}

2. **第一层 EncoderLayer**：

   - **自注意力**：建模序列内部 96 个时间点之间的依赖关系；
   - **分解**：得到 `残差（季节性部分） + 趋势`；
   - **前馈卷积**：进一步提取非线性特征；
   - **再次分解**：去掉趋势，只留下更新后的残差。

   输出仍然是：

   X∈R32×96×512X \in \mathbb{R}^{32 \times 96 \times 512}

3. **多层堆叠**（例如 2~3 层）：
    每一层都会不断**去趋势**，保留主要的**季节性波动**，趋势部分则被传递到解码器。

------

### 🔎 解码器（Decoder）处理过程

输入给解码器的有两部分：

1. **Decoder 输入序列**：
    通常是由 `label_len`（历史一部分，比如 48 步）+ 未来的 24 个 **占位符 0** 组成，形状：

   DecInput∈R32×(48+24)×512\text{DecInput} \in \mathbb{R}^{32 \times (48+24) \times 512}

   其中未来 24 步没有真实值，所以用 0 或者 mask 代替。

2. **Encoder 输出**：

   EncOutput∈R32×96×512\text{EncOutput} \in \mathbb{R}^{32 \times 96 \times 512}

------

### 解码器逐层计算

1. **自注意力**（Decoder 内部）：
    只在 `48+24` 序列内部建模，保证预测第 t 步时不会看到未来信息。

2. **交叉注意力**：
    将 `DecInput` 与 `EncOutput` 对齐，让解码器能“查询”编码器里的历史模式。

3. **分解三次**：

   - 第一次：得到季节性部分 + 趋势1；
   - 第二次：结合交叉注意力结果，再分解，得到趋势2；
   - 第三次：结合前馈网络，再分解，得到趋势3。

   最终趋势：

   Trend=Trend1+Trend2+Trend3\text{Trend} = \text{Trend}_1 + \text{Trend}_2 + \text{Trend}_3

   季节性残差会继续传下去。

------

### 📌 输出拼接

1. **残差（季节性部分）**：从 Decoder 最后一层得到

   SeasonalPart∈R32×24×1\text{SeasonalPart} \in \mathbb{R}^{32 \times 24 \times 1}

2. **趋势（Trend 部分）**：由 `projection` 卷积输出，形状相同

   TrendPart∈R32×24×1\text{TrendPart} \in \mathbb{R}^{32 \times 24 \times 1}

3. **最终预测**：

   Y^=SeasonalPart+TrendPart\hat{Y} = \text{SeasonalPart} + \text{TrendPart}

------

### 🎯 举个直观例子

假设我有 **风速、风向、气温、气压、空气密度、桨距角、转速** 这 7 个特征，输入 **96 步（4 天，每天 24 小时）**，我要预测未来 **24 步（1 天）**的功率：

1. **Encoder**：
   - 把历史 96 步分解成**长期趋势（风速逐渐增强）** + **周期性波动（白天/夜晚温度周期、转速周期）**；
   - 保留周期性残差，丢掉趋势部分（但趋势会送到解码器）。
2. **Decoder**：
   - 输入 = 最近 48 步真实值 + 未来 24 步占位符；
   - 自注意力学到“过去 48 步的周期结构”；
   - 交叉注意力查询编码器，获得完整 96 步历史的趋势；
   - 分解并逐步加上趋势，预测未来 24 步功率。
3. **输出**：
   - 未来 24 步的功率预测曲线，会包含：
     - **趋势**：比如未来一天风速持续升高，功率逐渐增大；
     - **季节性**：例如 24 小时的昼夜波动、周期性负载变化。

------

要不要我帮你画一张 **数据流向示意图**（Encoder/Decoder + Trend/Seasonal 分解箭头），这样你可以在汇报里直观展示【[32,96,7] → [32,24,1]】的处理过程？



### 1. 趋势 vs 季节性

- **趋势（trend）**：
   是低频、平滑的成分（长期缓慢上升/下降）。比如风速逐渐增强、电价慢慢上涨。
   👉 通常 **变化慢，结构简单**，甚至可以用线性/卷积拟合出来。
- **季节性（seasonal）/残差**：
   是高频、快速波动的部分（周期振荡、昼夜周期、突发波动）。
   👉 **复杂、非线性、依赖性强**，需要强模型（如注意力机制）来捕捉。

------

### 2. Transformer 的“短板”

普通 Transformer（比如 Informer、Transformer-TS）有两个问题：

1. **注意力很容易去拟合趋势**
    趋势是一种低频信号，模型很容易就把注意力放在“序列整体形状”上，而不是细节。这样会导致模型预测时**输出一条平滑曲线**，但缺少周期性波动。
2. **难以建模长序列的高频依赖**
    季节性/周期性才是 Transformer 应该重点学的东西，但如果趋势不去掉，它会掩盖掉高频依赖。

------

### 3. Autoformer 的做法：不断去趋势

Autoformer 的核心思想是：

- **趋势**：简单 → 单独提取出来，用卷积 Projection 来预测。
- **季节性**：复杂 → 留在残差里交给注意力机制去建模。

这就是为什么每一层 **Encoder/Decoder** 之间都要“去一次趋势”：

1. **Encoder 层**：
   - 输入 = 残差 + 趋势
   - 去掉趋势，只留下残差，送进下一层。
   - 趋势部分则被丢弃（因为简单）。
2. **Decoder 层**：
   - 每次 attention 和前馈后都会分解 → 得到趋势分量。
   - 所有趋势分量在最后**累加起来**，再加上残差预测，形成最终结果。

------

### 4. 好处

- **建模更聚焦**：注意力层只需要关注季节性（复杂关系），不会被趋势干扰。
- **提高稳定性**：趋势被分解出来后，模型不容易“漂移”。
- **提升长期预测效果**：长期预测里，趋势部分容易外推，而季节性部分需要捕捉依赖关系 → Autoformer 正好把两者分开建模。





## 2 linear_pattern_extractor.py

好的 👍 我来逐段讲解这份 **`Linear_extractor`** 代码，结合 Autoformer 的思想。

这个模块主要功能是：
 👉 把输入的时间序列 **分解成趋势 (trend) + 季节性 (seasonal)**，然后用 **线性层** 对二者分别建模，最后再合成预测结果。

------

### 代码解析

```python
import torch
import torch.nn as nn
from ..layers.Autoformer_EncDec import series_decomp
```

- 引入 PyTorch 基础模块。
- `series_decomp`：是 Autoformer 定义的 **序列分解函数**，把输入序列拆分为 **季节性部分** 和 **趋势部分**。

------

```python
class Linear_extractor(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    这个模块源自 FEDformer/Autoformer，主要思想是：
    - 分解序列 = 季节性 + 趋势
    - 各自用线性映射做预测
    """
```

------

#### 初始化

```python
def __init__(self, configs, individual=False):
    super(Linear_extractor, self).__init__()
    self.seq_len = configs.seq_len     # 输入序列长度
    self.pred_len = configs.d_model    # 输出序列长度 (预测步数，用 d_model 配置)
    self.decompsition = series_decomp(configs.moving_avg)  # 趋势 + 季节性分解器
    self.individual = individual       # 是否对每个变量单独建模
    self.channels = configs.enc_in     # 输入通道数（变量数）
    self.enc_in = 1 if configs.CI else configs.enc_in  # 是否每个变量独立处理
```

- **`configs.seq_len`**：历史窗口长度，例如输入 96 步。
- **`configs.d_model`**：预测长度，例如输出未来 24 步。
- **`configs.enc_in`**：输入变量数量（多元时间序列）。
- **`configs.CI`**：通道独立模式，若开启则每个通道单独处理。

------

#### 独立 / 共享模式

```python
if self.individual:
    # 每个变量单独一套线性层
    self.Linear_Seasonal = nn.ModuleList()
    self.Linear_Trend = nn.ModuleList()

    for i in range(self.channels):
        self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
        self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

        # 初始化：权重平均分布 (均匀加权)
        self.Linear_Seasonal[i].weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend[i].weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
else:
    # 所有变量共享一套线性层
    self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
    self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    self.Linear_Seasonal.weight = nn.Parameter(
        (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
    self.Linear_Trend.weight = nn.Parameter(
        (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
```

🔎 **解释：**

- **Individual 模式**：对每个变量独立建模，互不干扰。
- **共享模式**：所有变量共享线性层，假设它们服从相同的规律。
- **初始化**：权重设置为均匀分布，意味着预测一开始是“平均值外推”，避免模型偏置太大。

------

#### 编码器

```python
def encoder(self, x):
    seasonal_init, trend_init = self.decompsition(x)  # 分解成季节性+趋势
    seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
    # 现在维度是 [B, D, L]，方便线性层操作

    if self.individual:
        seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                      dtype=seasonal_init.dtype).to(seasonal_init.device)
        trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                   dtype=trend_init.dtype).to(trend_init.device)

        for i in range(self.channels):
            seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
            trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
    else:
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

    x = seasonal_output + trend_output  # 融合趋势+季节性
    return x.permute(0, 2, 1)           # 转回 [B, L, D]
```

🔎 **解释：**

1. 输入 `x` 形状 `[B, L, D]`。
2. `series_decomp(x)` → 分解为 **趋势** 和 **季节性**。
3. 每部分用 **线性层** 从输入长度 → 预测长度做映射。
4. 最后把趋势 + 季节性加起来，得到预测序列。

------

#### 预测函数

```python
def forecast(self, x_enc):
    return self.encoder(x_enc)
```

- 简单封装，方便调用。

------

#### 前向传播

```python
def forward(self, x_enc):
    if x_enc.shape[0] == 0:
        return torch.empty((0, self.pred_len, self.enc_in)).to(x_enc.device)
    dec_out = self.forecast(x_enc)
    return dec_out[:, -self.pred_len:, :]  # [B, L, D]
```

🔎 **解释：**

- `x_enc`：输入序列 `[B, seq_len, D]`。
- `forecast(x_enc)`：得到预测输出 `[B, pred_len, D]`。
- `dec_out[:, -self.pred_len:, :]`：取最后 `pred_len` 个时间步作为最终预测。



这个 `Linear_extractor` 的作用是：

1. **分解**输入时间序列 = 趋势 + 季节性；
2. **分别建模**趋势和季节性，用线性层做投影；
3. **重构输出**作为预测序列。

它本质上是一个 **线性预测器**，可以看作 Autoformer/FEDformer 的「低阶模块」。
 复杂模型（比如 DUET）会把它作为子模块，再叠加注意力机制、聚类机制来增强。





好，我来结合你给的 **`Linear_extractor`**，用一个具体例子（比如输入 **[32,96,7]**，预测未来 **24 步**）走一遍它的处理过程。这样会更直观。

------

### 🎯 假设输入

- **输入序列**
  $$
  x \in \mathbb{R}^{32 \times 96 \times 7}
  $$

  - batch = 32
  - 历史长度 = 96（过去 96 个时间步，比如 4 天，每天 24 小时）
  - 特征数 = 7（风速、气温、气压、功率、桨距角、密度、转速）

- **目标**
   预测未来 24 步（1 天）的 7 个变量：
  $$
  \hat{y} \in \mathbb{R}^{32 \times 24 \times 7}
  $$

------

### 🔎 Linear_extractor 的处理流程

1. **序列分解**

```
seasonal_init, trend_init = self.decompsition(x)
```

- `series_decomp` 用滑动平均分解输入序列：
  $$
  x = \text{seasonal\_init} + \text{trend\_init}
  $$

举例：

- **趋势**：风速整体逐渐升高，功率逐渐增大；
- **季节性**：昼夜周期波动（24 小时周期），或者短期波动。

------

2. **转置维度**

```
seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
```

把 `[32,96,7]` 变成 `[32,7,96]`，这样 **每个变量单独是一个时间序列**，方便送进 `nn.Linear`。

------

3. **线性映射预测**

- 如果是 **individual=True**：每个变量都有独立的 `nn.Linear`（从 96 → 24）。
- 如果是 **共享模式**：所有变量共享一套 `nn.Linear`。

```
seasonal_output = self.Linear_Seasonal(seasonal_init)  # [32,7,24]
trend_output    = self.Linear_Trend(trend_init)        # [32,7,24]
```

- `nn.Linear(96,24)` 等价于学到一个「投影矩阵」，把过去 96 步线性组合，得到未来 24 步。
- 初始权重是均匀分布 → 刚开始预测相当于做“平均值外推”。

------

4. **融合趋势 + 季节性**

```
x = seasonal_output + trend_output
```

得到：
$$
\hat{y} \in \mathbb{R}^{32 \times 7 \times 24}
$$
再转回 `[32,24,7]` 方便后续使用。

------

5. **输出预测**

```
return dec_out[:, -self.pred_len:, :] 
```

- `dec_out` 形状 `[32,24,7]`
- 最终输出就是未来 24 步的 7 个变量。

------

### 📌 举个直观例子

假设输入变量是：

- **风速**（逐渐升高 + 昼夜波动）
- **气温**（下降趋势 + 日间高夜间低）
- **功率**（随风速变化 + 振荡）

那么 `Linear_extractor` 的处理：

1. **分解风速序列**：
   - 趋势：整体升高的直线；
   - 季节性：24 小时周期性振荡。
2. **线性预测趋势部分**：
   - 趋势线被外推（线性延长），未来 24 小时继续升高。
3. **线性预测季节性部分**：
   - 把过去的周期模式“复制”到未来 → 得到昼夜波动。
4. **融合**：
   - 未来风速预测 = 升高的趋势 + 周期性的昼夜波动；
   - 功率、气温等变量也类似。

最终输出：
 未来 24 小时的 **风速、气温、功率** 等预测曲线。



## 3 distributional_router_encoder.py

👌 我来帮你详细讲解这个 **`encoder`** 模块。

这个模块看起来是 **DUET** 里 **distributional router encoder** 的一个实现，用于 **时间维度聚类 (Temporal Clustering)**，把输入序列映射到“专家 (experts)”的分布上。

------

### 代码逐段解析

```python
class encoder(nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        input_size = config.seq_len          # 输入序列长度
        num_experts = config.num_experts     # 专家个数 (聚类数)
        encoder_hidden_size = config.hidden_size  # 隐藏层维度

        # 一个 2 层 MLP，用来拟合“分布”
        self.distribution_fit = nn.Sequential(
            nn.Linear(input_size, encoder_hidden_size, bias=False),  # [B, L] -> [B, H]
            nn.ReLU(),                                              # 非线性激活
            nn.Linear(encoder_hidden_size, num_experts, bias=False) # [B, H] -> [B, num_experts]
        )
```

🔎 **解释：**

- 输入序列长度为 `seq_len`，即每个时间序列的历史窗口大小。
- 模块内部是一个 **两层全连接网络 (MLP)**，把序列特征压缩到 `hidden_size`，再映射到 `num_experts`。
- 输出的维度等于 `num_experts`，可以理解为“输入序列属于每个专家的概率/权重”。

------

```python
def forward(self, x):
    mean = torch.mean(x, dim=-1)   # 在最后一个维度上取平均
    out = self.distribution_fit(mean)
    return out
```

- **输入 `x`**：通常形状 `[B, L, D]`
  - `B` = batch size
  - `L` = 序列长度 (`seq_len`)
  - `D` = 特征数（变量数，enc_in）
- `torch.mean(x, dim=-1)`：对变量维度取均值，得到 `[B, L]`，相当于把多个变量压缩成单通道。
- `distribution_fit(mean)`：把 `[B, L]` 映射到 `[B, num_experts]`，输出表示该序列在 **各个专家/聚类** 下的“分布”。

------

### 🔧 举例说明

假设：

- `seq_len=96`（输入 96 步时间序列）
- `num_experts=4`（聚类成 4 类时间模式）
- `hidden_size=128`

流程：

1. 输入 `x: [32, 96, 7]`（batch=32，有 7 个变量）。
2. 对变量取平均：`mean: [32, 96]`。
3. 经过 MLP：`out: [32, 4]`。
   - 每个样本都会得到一个 `[4]` 向量，表示在 4 个专家上的权重。

------

### 🧩 作用

这个 `encoder` 的作用是 **把原始时间序列分配到不同的专家 (clusters)**，具体来说：

- 类似于 **soft clustering**：不是硬分类，而是得到一个分布。
- 每个专家可能代表 **不同时间模式**（比如：平稳区间、突变区间、周期性模式、趋势模式）。
- DUET 就是用这种方式来 **分解复杂的时间维度关系**





## 4 RevIN.py

很好，你贴的是 **RevIN (Reversible Instance Normalization)** 模块的实现。这个在很多时间序列预测模型里（包括 DUET、TimeXer 等）都会出现，用来**消除不同时间序列之间的分布差异**。我给你分块讲一下它的逻辑和意义。

------

### 1. 模块作用

RevIN 的主要思想是：

- 在 **输入时 (norm)**：对每个样本、每个通道（feature）进行 **归一化**，消除数值范围和分布的差异，使模型更容易学习。
- 在 **输出时 (denorm)**：把预测结果再还原到原始的数值分布，保证结果可解释。

它的“Reversible”就是指这种 **归一化 ↔ 反归一化** 的过程。

------

### 2. 代码解析

#### 初始化

```python
def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
    self.num_features = num_features   # 通道数 D
    self.eps = eps                     # 数值稳定项
    self.affine = affine               # 是否引入可学习仿射参数
    self.subtract_last = subtract_last # 是否减去最后一个值
    if self.affine:
        self._init_params()            # 初始化 scale 和 bias
```

- `affine=True` 时，会学一个 `weight` 和 `bias`，类似于 BatchNorm 的可学习缩放和偏移。
- `subtract_last=True` 主要用于 **非平稳序列**（比如风速、价格），可以让模型更聚焦于变化量而不是绝对值。

------

#### 前向传播

```python
def forward(self, x, mode: str):
    if mode == 'norm':
        self._get_statistics(x)
        x = self._normalize(x)
    elif mode == 'denorm':
        x = self._denormalize(x)
    return x
```

分两种模式：

- `"norm"`：标准化输入（训练或推理前使用）。
- `"denorm"`：把预测结果还原到原始尺度。

------

#### 统计量计算

```python
def _get_statistics(self, x):
    dim2reduce = tuple(range(1, x.ndim - 1))  # 沿时间维度求均值/方差
    if self.subtract_last:
        self.last = x[:, -1, :].unsqueeze(1) # 记住最后时刻
    else:
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
    self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
```

- 对输入 `x: [B, L, D]`，在 `L` 维度上求均值和标准差。
- `subtract_last=True` 时，用最后一个值代替均值（对应“去趋势”操作）。
- `.detach()` 的作用是防止统计量进入反向传播。

------

#### 归一化

```python
def _normalize(self, x):
    if self.subtract_last:
        x = x - self.last
    else:
        x = x - self.mean
    x = x / self.stdev
    if self.affine:
        x = x * self.affine_weight
        x = x + self.affine_bias
    return x
```

就是标准化公式：

$x' = \frac{x - \mu}{\sigma}$

加上可学习的 `scale` 和 `bias`。

------

#### 反归一化

```python
def _denormalize(self, x):
    if self.affine:
        x = x - self.affine_bias
        x = x / (self.affine_weight + self.eps * self.eps)
    x = x * self.stdev
    if self.subtract_last:
        x = x + self.last
    else:
        x = x + self.mean
    return x
```

就是反向操作：

$x = (x' - b) / w \times \sigma + \mu$

------

### 3. 举例说明

假设输入 `x: [32, 96, 7]`（32 个样本，序列长度 96，变量数 7）。

1. **norm 阶段**：
   - 先按时间维度计算每个通道的均值、标准差。
   - 得到归一化后的序列 `x_norm`，均值 0，方差 1。
2. **训练模型**：
   - 模型输入 `x_norm`，输出预测结果 `y_norm`。
3. **denorm 阶段**：
   - 把 `y_norm` 还原回原始尺度，得到预测值 `y_pred`。

这样模型就能在标准化空间学习，又能输出物理上有意义的结果。

------

⚡总结：

- `encoder` 你之前贴的主要是 **聚类 (TCM/CCM)**。
- `RevIN` 主要是 **预处理 & 后处理模块**，让神经网络在稳定分布下学习。
- 它们两个是 **不同功能的组件**：一个在 **建模核心**，一个在 **输入/输出规范化**。

要不要我帮你整理一个 **DUET 的模块表**（比如 `encoder` = TCM，`RevIN` = 数据归一化工具，`linear_extractor_cluster` = 模式提取…），这样你能快速知道每个文件的功能？



## 5 linear_extractor_cluster.py

你贴的这份代码实现了 **“稀疏门控的专家网络（Sparse Gated Mixture of Experts, MoE）”**，结合了 Autoformer/Linear_extractor 做时间序列特征提取，并且支持 RevIN 归一化。下面我帮你逐块讲解，保证你快速理解核心逻辑。

------

明白了，我给你把你贴的整个 `Linear_extractor_cluster` 以及 `SparseDispatcher` 代码模块**每一段都详细解释**，不仅说明类名，还会解释里面每行或每块的作用。下面是完整解析：

------

### 1️⃣ SparseDispatcher

```python
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts."""
```

- **作用**：辅助实现稀疏门控 MoE（Mixture of Experts）
- **功能**：
  - 将输入 batch 分发给各个专家（dispatch）
  - 将各专家输出按 gate 权重合并（combine）

------

#### 初始化

```python
def __init__(self, num_experts, gates):
    self._gates = gates
    self._num_experts = num_experts
```

- `num_experts`：专家数量
- `gates`：每个样本对应每个专家的权重，shape `[batch_size, num_experts]`

```python
sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
_, self._expert_index = sorted_experts.split(1, dim=1)
self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
self._part_sizes = (gates > 0).sum(0).tolist()
```

- 找到非零 gate 的专家索引和 batch 索引
- `_part_sizes`：每个专家实际接收的样本数量

```python
gates_exp = gates[self._batch_index.flatten()]
self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
```

- `_nonzero_gates` 保存每个专家对应样本的 gate 权重，用于合并输出

------

#### dispatch

```python
def dispatch(self, inp):
    inp_exp = inp[self._batch_index].squeeze(1)
    return torch.split(inp_exp, self._part_sizes, dim=0)
```

- **作用**：把输入 batch 按 gate 分配给各个专家
- 返回 `[num_experts]` 个小 batch，每个专家只接收对应样本

------

#### combine

```python
def combine(self, expert_out, multiply_by_gates=True):
    stitched = torch.cat(expert_out, 0)
    if multiply_by_gates:
        stitched = torch.einsum("i...,ij->i...", stitched, self._nonzero_gates)
```

- **作用**：合并专家输出
- `einsum` 对每个样本乘以 gate 权重

```python
shape = list(expert_out[-1].shape)
shape[0] = self._gates.size(0)
zeros = torch.zeros(*shape, requires_grad=True, device=stitched.device)
combined = zeros.index_add(0, self._batch_index, stitched.float())
return combined
```

- 使用 `_batch_index` 把 stitched 输出累加到对应样本位置，得到最终 batch 输出

------

#### expert_to_gates

```python
def expert_to_gates(self):
    return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
```

- 返回每个专家的 gate 权重，用于分发计算

------

### 2️⃣ Linear_extractor_cluster

```python
class Linear_extractor_cluster(nn.Module):
    """Sparsely gated mixture of experts layer with Linear_extractor experts."""
```

- **功能**：实现一个 MoE 层，每个专家是 `Linear_extractor`（提取趋势+季节性）
- **特点**：
  - 每个样本只选择 top-k 个专家
  - 支持训练时加噪声的 gate（noisy gating）
  - 支持 RevIN 输入归一化

------

#### 初始化

```python
self.noisy_gating = config.noisy_gating
self.num_experts = config.num_experts
self.input_size = config.seq_len
self.k = config.k
self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])
self.W_h = nn.Parameter(torch.eye(self.num_experts))
self.gate = encoder(config)
self.noise = encoder(config)
self.n_vars = config.enc_in
self.revin = RevIN(self.n_vars)
self.CI = config.CI
self.softplus = nn.Softplus()
self.softmax = nn.Softmax(1)
self.register_buffer("mean", torch.tensor([0.0]))
self.register_buffer("std", torch.tensor([1.0]))
assert self.k <= self.num_experts
```

- 初始化所有参数和组件：
  - `experts`：每个专家都是 `Linear_extractor`
  - `gate` & `noise`：计算 gate 权重和噪声
  - `RevIN`：归一化输入
  - `W_h`：专家输出线性组合矩阵
  - `softplus` & `softmax`：用于 gate 计算

------

#### cv_squared

```python
def cv_squared(self, x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)
```

- **作用**：计算方差/均值²，用作负载均衡 loss
- 防止某些专家过载或闲置

------

#### _gates_to_load

```python
def _gates_to_load(self, gates):
    return (gates > 0).sum(0)
```

- **作用**：计算每个专家实际接收的样本数

------

#### _prob_in_top_k

```python
def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
    ...
```

- **作用**：计算噪声下每个专家进入 top-k 的概率，用于梯度回传
- 来源论文：[Noisy Top-k Gating](https://arxiv.org/abs/1701.06538)

------

#### noisy_top_k_gating

```python
def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
    clean_logits = self.gate(x)
    if self.noisy_gating and train:
        raw_noise_stddev = self.noise(x)
        noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
        noise = torch.randn_like(clean_logits)
        noisy_logits = clean_logits + (noise * noise_stddev)
        logits = noisy_logits @ self.W_h
    else:
        logits = clean_logits

    logits = self.softmax(logits)
    top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
    top_k_logits = top_logits[:, : self.k]
    top_k_indices = top_indices[:, : self.k]
    top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

    zeros = torch.zeros_like(logits, requires_grad=True)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)

    if self.noisy_gating and self.k < self.num_experts and train:
        load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    else:
        load = self._gates_to_load(gates)

    return gates, load
```

- **功能**：
  1. 计算每个样本 top-k 专家
  2. 归一化 gate 权重
  3. 可选训练阶段加噪声
  4. 输出 load（每个专家的样本数量）

------

#### forward

```python
def forward(self, x, loss_coef=1):
    gates, load = self.noisy_top_k_gating(x, self.training)
    importance = gates.sum(0)
    loss = self.cv_squared(importance) + self.cv_squared(load)
    loss *= loss_coef

    dispatcher = SparseDispatcher(self.num_experts, gates)

    if self.CI:
        x_norm = rearrange(x, "(x y) l c -> x l (y c)", y=self.n_vars)
        x_norm = self.revin(x_norm, "norm")
        x_norm = rearrange(x_norm, "x l (y c) -> (x y) l c", y=self.n_vars)
    else:
        x_norm = self.revin(x, "norm")

    expert_inputs = dispatcher.dispatch(x_norm)
    gates = dispatcher.expert_to_gates()
    expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
    y = dispatcher.combine(expert_outputs)

    return y, loss
```

- **流程**：
  1. 计算 gate + load
  2. 计算负载均衡 loss
  3. 对输入做 RevIN 归一化
  4. 分发给专家
  5. 每个专家输出
  6. 合并专家输出
  7. 返回预测结果 + loss

------

✅ 总结：

- `SparseDispatcher` 负责输入分发 & 输出合并
- `Linear_extractor_cluster` 实现 **稀疏门控 MoE 层**
- 每个专家是 `Linear_extractor` → 拆分趋势 + 季节性
- 加上 **RevIN + Noisy Top-k Gating + 负载均衡 loss** → 提高模型泛化和训练稳定性

------



### 处理过程

明白了，我帮你梳理一下 **整个 Linear_extractor_cluster + SparseDispatcher + Linear_extractor 的流程**，假设你输入是 `[batch, seq_len, channels] = [32, 96, 7]`，预测 24 步（pred_len=24）。我会按数据流、模块分解，讲清每一步做了什么。

------

#### 1️⃣ 输入数据

- **形状**：`x_enc = [32, 96, 7]`
  - batch = 32
  - seq_len = 96（历史长度）
  - channels = 7（特征维度，例如风速、功率等）
- **目标**：预测未来 24 个时间步的序列，输出 `[32, 24, 7]`。

------

#### 2️⃣ RevIN 归一化

- **模块**：`RevIN`
- **功能**：
  - 对每个样本每个通道进行均值-方差归一化
  - 如果 `CI=True`，先 reshape `[batch*channels, seq_len, 1]` 做归一化，再恢复原 shape
- **输出形状**：仍然 `[32, 96, 7]`，但是值被归一化到 0 均值、单位方差附近

------

#### 3️⃣ Gate 计算（Noisy Top-k Gating）

- **模块**：`encoder` → `noisy_top_k_gating`
- **输入**：归一化后的 `[32, 96, 7]`
- **处理**：
  1. 先计算每个样本对应每个专家的 `logits`，形状 `[32, num_experts]`
  2. 如果训练阶段 `noisy_gating=True`，在 logits 上加噪声
  3. softmax 得到概率
  4. 选择 **top-k** 专家作为当前样本的预测路径
- **输出**：
  - `gates`：[32, num_experts]，每行只有 k 个非零值
  - `load`：[num_experts]，每个专家实际被选中的样本数

------

#### 4️⃣ SparseDispatcher 分发输入

- **模块**：`SparseDispatcher(dispatch)`
- **输入**：归一化后的 `[32, 96, 7]`
- **处理**：
  - 根据 gate，把每个样本分发到对应的 top-k 专家
  - 形成每个专家的小 batch
- **输出**：
  - `[num_experts]` 个小 tensor，例如每个 `[num_samples_for_expert, 96, 7]`

------

#### 5️⃣ 每个专家处理（Linear_extractor）

- **模块**：`Linear_extractor`
- **输入**：每个专家的小 batch `[batch_expert, 96, 7]`
- **步骤**：
  1. **序列分解**：`series_decomp`
     - 输入 `[batch_expert, 96, 7]` → 拆成：
       - `seasonal_init`：[batch_expert, 96, 7]（残差/季节性）
       - `trend_init`：[batch_expert, 96, 7]（趋势）
  2. **Linear 映射**：
     - 将每个时间序列（长度 96）通过线性层映射到预测长度 24
     - `seasonal_output` + `trend_output` → `[batch_expert, 24, 7]`
- **输出**：每个专家的小 batch `[batch_expert, 24, 7]`

------

#### 6️⃣ SparseDispatcher 合并专家输出

- **模块**：`SparseDispatcher(combine)`
- **输入**：所有专家的小 batch 输出 + gates 权重
- **处理**：
  - 对每个样本，根据 gate 权重加权求和 top-k 专家输出
- **输出**：
  - `[32, 24, 7]`，和输入 batch 对齐，得到最终预测

------

#### 7️⃣ 负载均衡 loss

- **计算**：
  - importance = 每个专家 gate 总和
  - load = 每个专家实际接收样本数
  - `loss = cv_squared(importance) + cv_squared(load)`
- **用途**：
  - 训练时鼓励每个专家使用均匀，不让某些专家闲置或过载

------

#### 8️⃣ 总体数据流示意

```
Input x: [32, 96, 7]
      │
      ▼
RevIN (归一化) -> [32, 96, 7]
      │
      ▼
Top-k Gate (encoder + noisy gating) -> gates [32, num_experts]
      │
      ▼
SparseDispatcher.dispatch -> 每个专家小 batch
      │
      ▼
Linear_extractor (trend+seasonal) -> [batch_expert, 24, 7]
      │
      ▼
SparseDispatcher.combine -> [32, 24, 7] (预测输出)
```

------

✅ 总结：

- **输入 `[32,96,7]` → 归一化 → top-k gate → 分发给专家 → 每个专家提取 trend+seasonal → 输出 24 步预测 → 合并 top-k 输出 → `[32,24,7]`**
- 这个流程**把历史序列分解 + 专家稀疏组合**，兼顾局部模式学习和全局多样性

------

我可以帮你画一张 **流程图**，把 `[32,96,7]` 的输入到 `[32,24,7]` 的整个 MoE + Linear_extractor 处理可视化，带每个模块的形状，非常直观。

你希望我画吗？



该模型的设计灵感来源于以下论文：

- **Outrageously Large Neural Networks: Sparsely-Gated Mixture of Experts Layers**：提出了稀疏门控专家层的概念，通过选择性地激活部分专家来提高模型的计算效率。
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**：介绍了 Switch Transformer 模型，采用了类似的稀疏专家机制。