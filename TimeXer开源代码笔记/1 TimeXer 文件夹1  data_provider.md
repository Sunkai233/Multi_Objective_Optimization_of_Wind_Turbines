# TimeXer 文件夹1  data_provider

## 文件夹结构：

```
data_provider
├── __init__.py
├── data_factory.py
├── data_loader.py
├── m4.py
└── uea.py
```

## m4.py

### 代码加注释:

好的 👍
 我会在你提供的 **m4.py** 文件代码中逐段加上中文注释，帮助你理解每个函数、类和逻辑。

```python
# 本代码来自 Element AI Inc.，实现了 N-BEATS 模型所使用的 M4 数据集加载部分。
# 版权声明和许可信息略（非商业使用许可）。

"""
M4 数据集处理模块
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm
import logging
import os
import pathlib
import sys
from urllib import request


def url_file_name(url: str) -> str:
    """
    从 URL 提取文件名

    :param url: 输入的 URL 地址
    :return: 提取出的文件名
    """
    return url.split('/')[-1] if len(url) > 0 else ''


def download(url: str, file_path: str) -> None:
    """
    从指定 URL 下载文件到本地路径

    :param url: 下载链接
    :param file_path: 保存文件路径
    """

    def progress(count, block_size, total_size):
        # 下载进度条
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        sys.stdout.flush()

    if not os.path.isfile(file_path):  # 如果文件不存在，则下载
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]  # 设置请求头
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)  # 确保文件夹存在
        f, _ = request.urlretrieve(url, file_path, progress)  # 执行下载，带进度条
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:  # 文件已存在
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')


@dataclass()
class M4Dataset:
    """
    M4 数据集结构定义
    """
    ids: np.ndarray         # 时间序列的 ID
    groups: np.ndarray      # 时间序列所属的类别 (Yearly, Monthly 等)
    frequencies: np.ndarray # 时间序列的采样频率
    horizons: np.ndarray    # 预测步长（horizon）
    values: np.ndarray      # 序列的实际值

    @staticmethod
    def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4Dataset':
        """
        加载缓存的数据集

        :param training: True 表示加载训练集，False 表示加载测试集
        :param dataset_file: 数据集文件夹路径
        """
        info_file = os.path.join(dataset_file, 'M4-info.csv')       # 存放元信息
        train_cache_file = os.path.join(dataset_file, 'training.npz') # 训练集缓存文件
        test_cache_file = os.path.join(dataset_file, 'test.npz')      # 测试集缓存文件
        m4_info = pd.read_csv(info_file)  # 读取 M4-info.csv 文件
        return M4Dataset(
            ids=m4_info.M4id.values,
            groups=m4_info.SP.values,
            frequencies=m4_info.Frequency.values,
            horizons=m4_info.Horizon.values,
            values=np.load(train_cache_file if training else test_cache_file, allow_pickle=True)
        )


@dataclass()
class M4Meta:
    """
    M4 数据集的元信息定义
    包括不同季节性模式对应的 horizon、frequency 等
    """
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]     # 不同类别对应的预测长度
    frequencies = [1, 4, 12, 1, 1, 24]    # 不同类别对应的频率
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # 不同季节性模式对应的预测长度
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }  # 不同季节性模式对应的采样频率
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }  # 历史序列长度比例（来源于 interpretable.gin 配置）


def load_m4_info() -> pd.DataFrame:
    """
    读取 M4-info 文件

    :return: Pandas DataFrame，包含 M4 的元信息
    """
    return pd.read_csv(INFO_FILE_PATH)  # ⚠️ 注意：INFO_FILE_PATH 在本文件中未定义，需要在别处定义
```

------

要点总结：

1. `M4Dataset` 类主要负责 **读取训练/测试数据**。
2. `M4Meta` 类存放 **不同时间序列类别的元信息**（预测长度、频率、历史长度比例）。
3. `download` 函数实现了 **带进度条的文件下载**。
4. `load_m4_info()` 最后用到的 `INFO_FILE_PATH` 在这里没有定义，需要额外配置。



### 代码作用

这个 **`m4.py`** 文件的主要作用可以总结为：

它是 **N-BEATS 模型代码里专门用于处理 M4 数据集的工具模块**，功能大概分三类：

------

🔹 1. 数据下载与管理

- `download(url, file_path)`：
   负责从互联网上下载 M4 数据集的压缩包或缓存文件（带进度条），并保存到本地。

------

🔹 2. 数据集封装与加载

- `M4Dataset` 类：
  - 定义了 M4 数据集的基本结构（ID、分组、频率、预测步长、数值序列）。
  - `M4Dataset.load()`：可以从本地缓存文件里 **直接读取训练集或测试集**，并返回一个 `M4Dataset` 对象。
  - 这样做的好处是模型训练时不需要每次都去解析原始 CSV，而是用 `.npz` 缓存加快速度。

------

🔹 3. 元信息配置

- `M4Meta` 类：
  - 定义了 M4 不同时间序列类型（Yearly, Quarterly, Monthly, Weekly, Daily, Hourly）的
    - 预测步长（horizon）
    - 采样频率（frequency）
    - 历史长度比例（history size）
  - 这些信息在训练模型时会用到，比如决定输入窗口长度、预测区间长度。

### m4数据集

M4 数据集（M4 Competition Dataset）是 **一个时间序列预测竞赛数据集**，里面包含了 **10万条以上** 的不同时间序列数据（按 Yearly, Quarterly, Monthly, Weekly, Daily, Hourly 六类划分）。

每一条时间序列都有：

- **ID**（序列编号）
- **类别**（季节性模式，比如 Yearly）
- **频率**（1 表示年，12 表示月，24 表示小时等）
- **历史数据值**（训练数据）
- **预测步长 Horizon**（未来要预测多少点，比如月度序列要预测 18 个月）

------

📊 举个具体例子（Monthly 序列）

假设有一个月度的序列：

| M4id | Category | Frequency | Horizon | Values (历史序列)            |
| ---- | -------- | --------- | ------- | ---------------------------- |
| Y1   | Monthly  | 12        | 18      | [266, 145, 183, 119, 180, …] |

- **M4id**: `Y1` （序列的唯一标识符）
- **Category**: `Monthly` （月度序列）
- **Frequency**: `12` （一年 12 个点）
- **Horizon**: `18` （需要预测未来 18 个月的值）
- **Values**: `[266, 145, 183, 119, 180, …]` （从过去某年到现在的月度观测值）

------

📊 再举个 Yearly 序列的例子

| M4id | Category | Frequency | Horizon | Values (历史序列)            |
| ---- | -------- | --------- | ------- | ---------------------------- |
| Y10  | Yearly   | 1         | 6       | [123, 150, 160, 180, 200, …] |

- **Frequency = 1**，表示每年一个点
- **Horizon = 6**，表示要预测未来 6 年
- **Values** 是这个序列过去几十年的数值（比如经济指标、销量、能源消耗等）

------

📌 总结一下：
 M4 数据集就是一个 **大量时间序列的集合**，每个序列有 **历史值** + **类别元信息** + **未来预测步长**。
 N-BEATS 或其它模型的任务就是 **学会从历史值预测 Horizon 个未来值**。

## uea.py

### 代码加注释

好的 👍
 我在你给的代码基础上，逐行加上中文注释，帮你梳理每个函数和类的作用：

```python
import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """
    构建一个 batch 的张量（主要用于 DataLoader 的 collate_fn 函数）

    Args:
        data: 一个 batch 的样本列表，长度为 batch_size
              每个样本是 (X, y) 二元组
                - X: torch.tensor，形状 (seq_length, feat_dim)，序列长度可以不同
                - y: torch.tensor，形状 (num_labels,)，标签（分类/回归任务）
        max_len: 全局固定的序列长度
                 - 如果模型需要固定长度输入，就用这个参数
                 - 如果序列太长则截断，太短则补零

    Returns:
        X: (batch_size, padded_length, feat_dim) 输入特征张量（填充后）
        targets: (batch_size, num_labels) 标签张量
        padding_masks: (batch_size, padded_length) 布尔张量
            - 1 表示真实值，0 表示补零
    """

    batch_size = len(data)
    features, labels = zip(*data)  # 解压得到特征列表和标签列表

    # 每个样本原始序列的长度
    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)  # 如果没指定 max_len，则用当前 batch 的最大长度

    # 初始化特征张量，默认填 0
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])
    for i in range(batch_size):
        end = min(lengths[i], max_len)  # 避免超出 max_len
        X[i, :end, :] = features[i][:end, :]  # 拷贝数据到 batch 张量中

    # 将标签堆叠成 (batch_size, num_labels) 形状
    targets = torch.stack(labels, dim=0)

    # 构造 padding 掩码，标记哪些位置是有效的
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    根据序列长度生成 padding 掩码

    Args:
        lengths: 张量，形状 (batch_size,) ，表示每个样本的真实长度
        max_len: 批次中序列的最大长度

    Returns:
        mask: (batch_size, max_len) 的布尔掩码张量
              - 1 表示保留（真实数据）
              - 0 表示 padding（补零）
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()  # 如果没传入，就用最大序列长度
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)                 # 保持 dtype 一致
            .repeat(batch_size, 1)            # 扩展为 (batch_size, max_len)
            .lt(lengths.unsqueeze(1)))        # 小于真实长度的为 1，否则为 0


class Normalizer(object):
    """
    数据归一化工具类
    - 可以对整个 DataFrame 或每个样本单独做标准化 / min-max 归一化
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: 归一化方式
                "standardization"   -> 全局标准化
                "minmax"            -> 全局 min-max 归一化
                "per_sample_std"    -> 每个样本单独标准化
                "per_sample_minmax" -> 每个样本单独 min-max 归一化
            mean, std, min_val, max_val: 可选，预先计算好的参数
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        对 DataFrame 做归一化

        Args:
            df: 输入的 pandas.DataFrame
        Returns:
            df: 归一化后的 DataFrame
        """
        if self.norm_type == "standardization":
            # 全局标准化 (x - mean) / std
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            # 全局 min-max 归一化
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            # 每个样本单独标准化
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            # 每个样本单独 min-max 归一化
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    线性插值填补缺失值

    Args:
        y: pd.Series，输入序列
    Returns:
        y: 填补缺失值后的序列
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    子采样函数
    - 如果序列长度大于 limit，就按步长 factor 采样
    - 否则原样返回

    Args:
        y: pd.Series，输入序列
        limit: 限制长度
        factor: 子采样步长
    Returns:
        y: 子采样后的序列
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
```

------

🔑 功能总结

1. **`collate_fn` + `padding_mask`**
   - 用于构造一个 batch 的输入，支持不同长度序列，统一成固定形状（padding）。
2. **`Normalizer`**
   - 提供不同的归一化方法（全局/逐样本的标准化和 min-max）。
3. **`interpolate_missing`**
   - 用插值方法填补缺失值。
4. **`subsample`**
   - 如果序列太长，可以子采样以减少计算量。

------

### 代码作用

明白了 ✅

这一段代码的**主要作用**可以总结为：



这是一个 **时间序列数据预处理工具模块**，它的职责是把原始的时间序列整理成能直接送进神经网络模型的张量，同时提供一些常用的数据清洗与归一化方法。

1. **批处理拼接 (`collate_fn` + `padding_mask`)**

   - 解决时间序列 **长短不一** 的问题
   - 把一批 `(X, y)` 数据对整理成统一形状的三维张量 `(batch_size, padded_len, feat_dim)`
   - 对短序列补零（padding），对长序列截断，并生成掩码标记哪些是真实值、哪些是补零

   👉 主要用于 **PyTorch DataLoader** 的 `collate_fn` 参数

------

1. **数据归一化 (`Normalizer`)**
   - 提供 4 种归一化方式：
     - 全局标准化
     - 全局 min-max
     - 按样本标准化
     - 按样本 min-max
   - 作用是消除不同特征量级差异，让模型更容易收敛

------

1. **缺失值处理 (`interpolate_missing`)**
   - 用 **线性插值** 自动填补时间序列里的 NaN 缺失点

------

1. **序列压缩 (`subsample`)**
   - 如果时间序列太长（> limit），就按照 `factor` 进行子采样
   - 这样可以减少计算量，提高训练速度

------



## data_loader.py



### 代码加注释

（代码太长了：链接在：“[TimeXer/data_provider/data_loader.py at main · thuml/TimeXer · GitHub](https://github.com/thuml/TimeXer/blob/main/data_provider/data_loader.py)”）



### 代码作用

**代码的主代码的主要作用**

1. **统一数据读取接口**
   - 针对不同来源的数据集（ETT、M4、PSM、MSL、SMAP、SMD、SWAT、UEA等），定义了对应的 Dataset 类。
   - 不同数据集的文件格式、时间粒度、切分方式不同，这些类屏蔽了差异，统一输出训练所需的 `(seq_x, seq_y, seq_x_mark, seq_y_mark)` 或 `(窗口数据, 标签)`。
2. **数据预处理**
   - 使用 `StandardScaler` 进行归一化（防止特征尺度差异影响训练）。
   - 缺失值填充 (`np.nan_to_num`)。
   - 提取时间特征（年月日、小时、分钟、weekday 等，或通过 `time_features` 提取频率编码）。
3. **时间序列切片**
   - 输入（历史窗口 `seq_len`）和输出（标签窗口 `label_len` + 预测窗口 `pred_len`）。
   - 保证在训练、验证、测试阶段使用不同的时间段。
4. **支持数据增强**
   - 如果 `args.augmentation_ratio > 0`，在训练集上运行 `run_augmentation_single`，扩充数据。
5. **兼容不同任务**
   - **预测任务**（ETTh1/ETTm1/Custom/M4）：输出带时间标记的输入和预测窗口。
   - **异常检测/分割任务**（PSM、MSL、SMAP、SMD、SWAT）：按固定滑动窗口划分序列，返回数据与对应标签。
   - **分类任务**（UEA）：加载 UEA Archive 数据集，返回序列与标签。

------

**各类 Dataset 的说明**

1. **`Dataset_ETT_hour` / `Dataset_ETT_minute`**
   - 用于 ETT 数据集（电力变压器负荷数据），小时级和分钟级。
   - 按照论文默认比例划分 **训练：验证：测试 = 12:4:4 月**。
2. **`Dataset_Custom`**
   - 用于自定义 CSV 数据集（第一列为 date，最后一列为 target）。
   - 按比例切分：70% 训练，20% 测试，10% 验证。
3. **`Dataset_M4`**
   - M4 时间序列预测竞赛数据集。
   - 按频率（Yearly、Quarterly、Monthly等）抽取序列，生成 in-sample 和 out-sample 窗口。
4. **`PSMSegLoader`、`MSLSegLoader`、`SMAPSegLoader`、`SMDSegLoader`、`SWATSegLoader`**
   - 针对工业过程/传感器的 **异常检测数据集**。
   - 按滑动窗口生成输入序列，并返回对应的异常标签。
5. **`UEAloader`**
   - 加载 UEA 时间序列分类数据集。
   - 支持选择部分文件、限制样本数、归一化等。



## data_factory.py



### 🔑代码加注释

这段代码其实就是一个 **数据工厂函数 (data_provider)**，根据配置参数自动选择并构造不同的数据集（Dataset）和对应的 DataLoader。
 我给你逐步加上中文注释和作用说明。

------

```
from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4,
    PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader,
    UEAloader, Dataset_Meteorology
)
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader


# =============================================
# 数据集字典 - 任务名 -> 数据集类
# =============================================
data_dict = {
    'ETTh1': Dataset_ETT_hour,      # 电力负荷 (小时级)
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,    # 电力负荷 (分钟级)
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,       # 自定义 CSV
    'm4': Dataset_M4,               # M4 竞赛数据
    'PSM': PSMSegLoader,            # 异常检测 (PSM)
    'MSL': MSLSegLoader,            # 异常检测 (MSL)
    'SMAP': SMAPSegLoader,          # 异常检测 (SMAP)
    'SMD': SMDSegLoader,            # 异常检测 (SMD)
    'SWAT': SWATSegLoader,          # 异常检测 (SWAT)
    'UEA': UEAloader,               # UEA 时间序列分类
    'Meteorology': Dataset_Meteorology  # 气象数据
}


# =============================================
# 数据工厂函数
# 根据任务类型和数据集选择合适的 Dataset & DataLoader
# =============================================
def data_provider(args, flag):
    # 根据参数 args.data 选择数据集类
    Data = data_dict[args.data]

    # 时间特征编码方式
    # 如果 embedding 方式是 "timeF"，则使用嵌入式编码 (1)，否则用传统时间分解 (0)
    timeenc = 0 if args.embed != 'timeF' else 1

    # DataLoader 配置
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True  # 测试集不打乱
    drop_last = False                                                    # 默认不丢弃最后一个 batch
    batch_size = args.batch_size
    freq = args.freq  # 采样频率 (h/m/d)

    # ========== 1. 异常检测任务 ==========
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,   # 窗口大小 (异常检测用固定窗口)
            flag=flag,
        )
        print(flag, len(data_set))   # 打印数据集规模
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    # ========== 2. 分类任务 ==========
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  
            # 分类任务需要特殊的 collate_fn（变长序列对齐 & mask）
        )
        return data_set, data_loader

    # ========== 3. 时间序列预测任务 ==========
    else:
        if args.data == 'm4':  # M4 数据集不丢弃最后 batch
            drop_last = False

        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],  # 窗口长度配置
            features=args.features,         # 'M' (多变量) / 'S' (单变量)
            target=args.target,             # 预测目标列
            timeenc=timeenc,                # 时间特征编码方式
            freq=freq,                      # 数据频率
            seasonal_patterns=args.seasonal_patterns  # 季节模式 (M4 特有)
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
```

------

### 🔑代码作用

这段代码就是一个 **统一数据入口**：

1. **根据数据集名称 (`args.data`)** 从 `data_dict` 选择对应的 `Dataset` 类。
2. **根据任务类型 (`args.task_name`)** 构建数据集：
   - **预测 (forecasting)** → 返回预测用 Dataset + DataLoader
   - **分类 (classification)** → 使用 `collate_fn` 处理变长序列
   - **异常检测 (anomaly_detection)** → 使用固定窗口 Dataset
3. **自动配置 DataLoader**（batch_size, shuffle, drop_last, num_workers）。

这样做的好处是：
 👉 在训练脚本里，只需要写 `train_data, train_loader = data_provider(args, 'train')`，不用关心具体的数据集差异。









