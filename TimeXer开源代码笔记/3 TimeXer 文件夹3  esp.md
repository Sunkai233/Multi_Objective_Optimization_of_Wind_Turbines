# TimeXer 文件夹3  esp

我帮你梳理每个文件的作用和用途：

```lua
exp/
├── __init__.py
├── exp_anomaly_detection.py
├── exp_basic.py
├── exp_classification.py
├── exp_imputation.py
├── exp_long_term_forecasting.py
├── exp_short_term_forecasting.py
```

**文件说明**

1. **`__init__.py`**
2. **`exp_anomaly_detection.py`**
   - 功能：进行异常检测实验，例如检测电力负荷或风电预测中的异常值。
3. **`exp_basic.py`**
   - 功能：基础实验模板或示例脚本。
4. **`exp_classification.py`**
   - 功能：分类任务实验。
5. **`exp_imputation.py`**
   - 功能：缺失值填补实验。
6. **`exp_long_term_forecasting.py`**
   - 功能：长周期预测实验。
7. **`exp_short_term_forecasting.py`**
   - 功能：短周期预测实验。



## **`exp_basic.py`**

你贴的这段代码是 `Exp_Basic` 类，是 **整个实验框架的基础类**，为各种时间序列实验（预测、异常检测等）提供 **通用接口和基础功能**。我给你详细分析每一部分：

------

### 1️⃣ 类定义与初始化

```
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
```

- `Exp_Basic` 是实验的 **父类**
- `args` 包含训练参数、模型选择、设备配置、超参数等

------

### 2️⃣ 模型字典

```
self.model_dict = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    ...
    'TimeXer': TimeXer
}
```

- `model_dict` 将 **字符串模型名** 映射到 **模型类**
- 支持多种时序预测模型，包括：
  - Transformer 系列：`Transformer`, `Informer`, `Autoformer`, `FEDformer`
  - 基于线性模型：`DLinear`
  - Patch/卷积时序模型：`PatchTST`, `MICN`, `Crossformer`
  - 自回归或混合模型：`SCINet`, `TimeXer`, `TSMixer`
- 对 `Mamba` 模型做了单独处理，提示用户安装依赖

------

### 3️⃣ 设备选择

```
self.device = self._acquire_device()
```

- `_acquire_device` 方法根据 `args.use_gpu` 决定使用 GPU 或 CPU
- 多 GPU 支持 `args.use_multi_gpu` 和 `args.devices`
- 返回 `torch.device` 对象

------

### 4️⃣ 模型构建接口

```
self.model = self._build_model().to(self.device)
```

- `_build_model()` 是 **抽象方法**，父类未实现，只定义接口
- 子类必须实现 `_build_model`，比如 `Exp_Anomaly_Detection` 中实现具体模型构建
- 这样父类负责 **设备分配**，子类负责 **模型具体构建**

------

### 5️⃣ 核心方法接口

```
def _get_data(self):
    pass

def vali(self):
    pass

def train(self):
    pass

def test(self):
    pass
```

- 这些方法是 **通用接口**：
  - `_get_data()` → 加载训练/验证/测试数据集
  - `vali()` → 验证模型效果
  - `train()` → 模型训练流程
  - `test()` → 模型测试与评估
- 父类只提供接口，具体逻辑由子类实现（如异常检测或长短期预测）

------

### 🔑 总结

`Exp_Basic` 主要作用是：

1. **统一管理模型和训练设备**
   - 通过 `model_dict` 支持多种模型
   - 自动分配 GPU/CPU
2. **提供实验框架接口**
   - 数据加载、训练、验证、测试方法抽象
   - 子类可以继承并实现不同任务逻辑（预测、异常检测等）
3. **增强代码复用性**
   - 所有实验类共享统一接口
   - 避免重复写训练/测试框架

## exp_long_term_forecasting.py

你这段代码是 `Exp_Long_Term_Forecast` 类的完整实现，它继承自 `Exp_Basic`，用于 **长周期时间序列预测任务**。相比于异常检测，逻辑更加复杂，因为涉及 **Encoder-Decoder 模型结构、解码器输入构造、多变量预测、自动混合精度训练（AMP）、预测结果反归一化和可视化**。下面我给你做一个 **详细模块解析**：

------

### 1️⃣ 类初始化

```
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
```

- 调用父类 `Exp_Basic` 完成模型字典初始化、设备分配
- `args` 包含模型类型、训练参数、预测长度、GPU 配置等

------

### 2️⃣ 模型构建

```
def _build_model(self):
    model = self.model_dict[self.args.model].Model(self.args).float()
    if self.args.use_multi_gpu and self.args.use_gpu:
        model = nn.DataParallel(model, device_ids=self.args.device_ids)
    return model
```

- 从 `model_dict` 选择模型构建实例
- 支持多 GPU 并行训练

------

### 3️⃣ 数据加载

```
def _get_data(self, flag):
    data_set, data_loader = data_provider(self.args, flag)
    return data_set, data_loader
```

- 根据 `flag` 获取训练、验证或测试数据
- 数据包含：
  - `batch_x` → 输入序列
  - `batch_y` → 输出/预测目标序列
  - `batch_x_mark`/`batch_y_mark` → 时间特征（如日期、时间戳、节假日等）

------

### 4️⃣ 优化器与损失函数

```
def _select_optimizer(self):
    return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

def _select_criterion(self):
    return nn.MSELoss()
```

- 使用 **Adam** 优化器
- 使用 **MSE** 作为回归损失函数

------

### 5️⃣ 验证函数（Vali）

```
def vali(self, vali_data, vali_loader, criterion):
```

- 功能：计算验证集平均损失

- 构建 **decoder input**：

  ```
  dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
  dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
  ```

  - 前 `label_len` 作为历史已知序列
  - 后续 `pred_len` 为预测占位

- 支持 **自动混合精度训练（AMP）**

- 输出只关注目标维度 `f_dim`

------

### 6️⃣ 训练函数（Train）

```
def train(self, setting):
```

- **核心流程**：
  1. 获取训练/验证/测试数据
  2. 初始化 EarlyStopping、优化器、损失函数
  3. 循环训练每个 epoch：
     - 构建 decoder 输入
     - 前向传播 → 计算损失
     - 反向传播 → 更新参数（支持 AMP）
     - 每 100 步打印训练速度和剩余时间
  4. 验证集和测试集计算损失
  5. 使用 EarlyStopping 判断是否提前停止
  6. 调整学习率
- 最终加载 **最佳模型** checkpoint

------

### 7️⃣ 测试函数（Test）

```
def test(self, setting, test=0):
```

核心流程

1. **加载测试数据和模型**（可选 `test` 参数）

2. **批量预测**：

   - 构建 decoder 输入
   - 前向传播
   - 取最后 `pred_len` 的输出
   - 支持 AMP

3. **预测反归一化**（inverse transform）：

   ```
   if test_data.scale and self.args.inverse:
       outputs = test_data.inverse_transform(outputs)
       batch_y = test_data.inverse_transform(batch_y)
   ```

4. **可视化**：

   - 每 20 批次绘制预测 vs 真实曲线
   - 保存 PDF 文件

5. **结果保存**：

   - `metrics.npy` → MAE、MSE、RMSE、MAPE、MSPE
   - `pred.npy` → 预测序列
   - `true.npy` → 真实序列
   - 文本文件 `result_long_term_forecast.txt` 保存 MSE/MAE

------

### 8️⃣ 特点总结

- **支持多模型**：TimesNet、Transformer、Autoformer 等
- **多变量/单变量**预测（`features` 参数）
- **Encoder-Decoder 预测**：适合长时间步预测
- **自动混合精度**：节省显存，加速训练
- **EarlyStopping**：防止过拟合
- **可视化与结果保存**：方便对比预测效果

------

✅ 总结一句话：

`Exp_Long_Term_Forecast` 是一个 **面向长周期时间序列预测的训练-验证-测试完整实验框架**，兼容多种 Transformer/Autoformer 类模型，并集成了 **AMP、EarlyStopping、反归一化、可视化和指标保存**，非常适合风电/负荷预测任务。