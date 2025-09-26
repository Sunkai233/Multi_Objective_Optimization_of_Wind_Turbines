# timeXer  训练运行run.py实现

## 代码实现

```python
import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    # 设置随机种子，确保实验结果的可重现性
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='TimesNet')

    # ================================== 基础配置 ==================================
    # 任务类型：长期预测、短期预测、数据填充、分类、异常检测
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    # 是否为训练模式 (1: 训练, 0: 测试)
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    # 模型标识符，用于区分不同的实验
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    # 模型名称：Autoformer、Transformer、TimesNet等
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # ================================== 数据加载配置 ==================================
    # 数据集类型
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    # 数据文件的根目录路径
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    # 具体的数据文件名
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    # 预测任务类型：M(多变量预测多变量), S(单变量预测单变量), MS(多变量预测单变量)
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    # 在S或MS任务中的目标特征
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    # 时间特征编码的频率：s(秒), t(分钟), h(小时), d(天), b(工作日), w(周), m(月)
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # 模型检查点保存位置
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # ================================== 预测任务配置 ==================================
    # 输入序列长度（历史数据长度）
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # 标签序列长度（开始标记长度，用于decoder）
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    # 预测序列长度（未来预测的时间步数）
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # M4数据集的季节性模式
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # 是否对输出数据进行逆变换
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # ================================== 数据填充任务配置 ==================================
    # 数据掩码比例（用于填充任务）
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # ================================== 异常检测任务配置 ==================================
    # 先验异常比例
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # ================================== 模型架构配置 ==================================
    # Mamba模型的扩展因子
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    # Mamba模型的卷积核大小
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    # TimesBlock的top-k参数
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    # Inception模块的核数量
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    # 编码器输入维度
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    # 解码器输入维度
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    # 输出维度
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    # 模型的隐藏维度
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    # 注意力头数
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    # 编码器层数
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    # 解码器层数
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    # 前馈网络维度
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    # 移动平均窗口大小
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    # 注意力因子
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    # 是否在编码器中使用蒸馏
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    # Dropout比例
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # 时间特征编码方式：timeF(时间特征), fixed(固定), learned(学习)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    # 激活函数类型
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    # 是否在编码器中输出注意力权重
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    # FreTS模型的通道独立性：0(通道依赖), 1(通道独立)
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    # 序列分解方法：moving_avg(移动平均), dft_decomp(DFT分解)
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    # 是否使用标准化：True(1), False(0)
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    # 下采样层数
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    # 下采样窗口大小
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    # 下采样方法：avg(平均), max(最大值), conv(卷积)
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    # SegRNN的分段长度
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # ================================== 优化配置 ==================================
    # 数据加载器的工作进程数
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # 实验重复次数
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    # 训练轮数
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    # 批次大小
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    # 早停耐心值（验证集性能不提升的最大轮数）
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # 学习率
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # 实验描述
    parser.add_argument('--des', type=str, default='test', help='exp description')
    # 损失函数
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # 学习率调整策略
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # 是否使用自动混合精度训练
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # ================================== GPU配置 ==================================
    # 是否使用GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    # GPU设备号
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # 是否使用多GPU
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # 多GPU设备ID列表
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # ================================== 去平稳投影器参数 ==================================
    # 投影器隐藏层维度列表
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    # 投影器隐藏层数量
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # ================================== 评估指标配置 ==================================
    # 是否使用DTW指标（动态时间规整，计算耗时，非必要不建议使用）
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # ================================== 数据增强配置 ==================================
    # 数据增强倍数
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    # 随机种子
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    # 以下是各种数据增强技术的开关
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")  # 抖动增强
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")  # 缩放增强
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")  # 等长排列增强
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")  # 随机长度排列增强
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")  # 幅度扭曲增强
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")  # 时间扭曲增强
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")  # 窗口切片增强
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")  # 窗口扭曲增强
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")  # 旋转增强
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")  # SPAWNER增强
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")  # DTW扭曲增强
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")  # 形状DTW扭曲增强
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")  # 加权DBA增强
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")  # 判别性DTW扭曲增强
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")  # 判别性形状DTW扭曲增强
    # 额外标签，用于实验标记
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # ================================== TimeXer模型配置 ==================================
    # TimeXer模型的补丁长度
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查CUDA可用性并设置GPU使用
    args.use_gpu = True if torch.cuda.is_available() else False
    print(torch.cuda.is_available())

    # 多GPU设置
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')  # 移除空格
        device_ids = args.devices.split(',')  # 分割设备ID
        args.device_ids = [int(id_) for id_ in device_ids]  # 转换为整数列表
        args.gpu = args.device_ids[0]  # 主GPU设备

    # 打印实验参数
    print('Args in experiment:')
    print_args(args)

    # ================================== 根据任务类型选择实验类 ==================================
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast  # 长期预测实验类
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast  # 短期预测实验类
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation  # 数据填充实验类
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection  # 异常检测实验类
    elif args.task_name == 'classification':
        Exp = Exp_Classification  # 分类实验类
    else:
        Exp = Exp_Long_Term_Forecast  # 默认为长期预测

    # ================================== 训练或测试模式 ==================================
    if args.is_training:
        # 训练模式：可以进行多次实验
        for ii in range(args.itr):
            # 创建实验实例
            exp = Exp(args)
            
            # 生成实验设置字符串，用于保存模型和结果
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,      # 任务名称
                args.model_id,       # 模型ID
                args.model,          # 模型名称
                args.data,           # 数据集名称
                args.features,       # 特征类型
                args.seq_len,        # 序列长度
                args.label_len,      # 标签长度
                args.pred_len,       # 预测长度
                args.d_model,        # 模型维度
                args.n_heads,        # 注意力头数
                args.e_layers,       # 编码器层数
                args.d_layers,       # 解码器层数
                args.d_ff,           # 前馈网络维度
                args.expand,         # 扩展因子
                args.d_conv,         # 卷积核大小
                args.factor,         # 注意力因子
                args.embed,          # 嵌入类型
                args.distil,         # 蒸馏标志
                args.des,            # 描述
                ii)                  # 实验次数

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)  # 开始训练

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)   # 训练后测试
            
            # 清空CUDA缓存，避免内存溢出
            torch.cuda.empty_cache()
    else:
        # 测试模式：仅进行一次测试
        ii = 0
        # 生成测试设置字符串
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # 创建实验实例
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)  # 仅测试模式
        torch.cuda.empty_cache()   # 清空CUDA缓存
```

## 实现1

```shell
#!/bin/bash

# ================================== 环境配置 ==================================
# 设置CUDA可见设备为GPU 1，指定使用第1号GPU进行训练
export CUDA_VISIBLE_DEVICES=1

# ================================== 变量定义 ==================================
# 定义模型名称为TimeXer（一种先进的时间序列预测模型）
model_name=TimeXer
# 定义实验描述，用于区分不同的实验设置
des='Timexer-MS'

# ================================== 实验1：预测未来96个时间步 ==================================
# 电力数据集长期预测实验 - 预测长度96
python3 -u run.py \
  --task_name long_term_forecast \      # 任务类型：长期预测
  --is_training 1 \                     # 训练模式：1表示训练，0表示测试
  --root_path ./dataset/electricity/ \  # 数据集根目录路径
  --data_path electricity.csv \         # 具体的数据文件名
  --model_id ECL_96_96 \               # 模型标识符：ECL(电力)_输入长度_预测长度
  --model $model_name \                # 使用的模型名称（TimeXer）
  --data custom \                      # 数据类型：自定义数据加载器
  --features MS \                      # 特征模式：MS = 多变量输入预测单变量输出
  --seq_len 96 \                       # 输入序列长度：使用96个历史时间步作为输入
  --label_len 48 \                     # 标签长度：decoder的起始token长度
  --pred_len 96 \                      # 预测长度：预测未来96个时间步
  --e_layers 1 \                       # 编码器层数：1层
  --factor 3 \                         # 注意力因子：控制注意力机制的稀疏程度
  --enc_in 321 \                       # 编码器输入维度：321个变量（320个客户+1个目标）
  --dec_in 321 \                       # 解码器输入维度：321个变量
  --c_out 321 \                        # 模型输出维度：321个变量
  --des $des \                         # 实验描述：用于结果文件命名
  --batch_size 4 \                     # 批次大小：4（较小，可能因为内存限制）
  --itr 1                              # 实验重复次数：1次

# ================================== 实验2：预测未来192个时间步 ==================================
# 电力数据集长期预测实验 - 预测长度192（约8天）
python3 -u run.py \
  --task_name long_term_forecast \      # 任务类型：长期预测
  --is_training 1 \                     # 训练模式
  --root_path ./dataset/electricity/ \  # 数据集根目录路径
  --data_path electricity.csv \         # 数据文件
  --model_id ECL_96_192 \              # 模型ID：输入96步，预测192步
  --model $model_name \                # 模型：TimeXer
  --data custom \                      # 自定义数据加载器
  --features MS \                      # 多变量预测单变量
  --seq_len 96 \                       # 输入序列长度：96步
  --label_len 48 \                     # 解码器起始长度：48步
  --pred_len 192 \                     # 预测长度：192步（比实验1更长期的预测）
  --e_layers 1 \                       # 编码器层数：1层
  --factor 3 \                         # 注意力因子：3
  --enc_in 321 \                       # 编码器输入维度：321
  --dec_in 321 \                       # 解码器输入维度：321
  --c_out 321 \                        # 输出维度：321
  --des $des \                         # 实验描述
  --batch_size 32 \                    # 批次大小：32（比实验1大，预测更长但模型复杂度相同）
  --itr 1                              # 实验次数：1

# ================================== 实验3：预测未来336个时间步 ==================================  
# 电力数据集长期预测实验 - 预测长度336（约2周）
 python3 -u run.py \
  --task_name long_term_forecast \      # 任务类型：长期预测
  --is_training 1 \                     # 训练模式
  --root_path ./dataset/electricity/ \  # 数据集路径
  --data_path electricity.csv \         # 数据文件
  --model_id ECL_96_336 \              # 模型ID：输入96步，预测336步
  --model $model_name \                # 模型：TimeXer
  --data custom \                      # 自定义数据类型
  --features MS \                      # 多变量预测单变量
  --seq_len 96 \                       # 输入序列长度：96
  --label_len 48 \                     # 标签长度：48
  --pred_len 336 \                     # 预测长度：336步（更长期预测，约2周）
  --e_layers 1 \                       # 编码器层数：1
  --factor 3 \                         # 注意力因子：3
  --enc_in 321 \                       # 编码器输入：321维
  --dec_in 321 \                       # 解码器输入：321维
  --c_out 321 \                        # 输出：321维
  --des $des \                         # 实验描述
  --batch_size 32 \                    # 批次大小：32
  --itr 1                              # 实验次数：1

# ================================== 实验4：预测未来720个时间步 ==================================
# 电力数据集长期预测实验 - 预测长度720（约1个月）
python3 -u run.py \
  --task_name long_term_forecast \      # 任务类型：长期预测
  --is_training 1 \                     # 训练模式
  --root_path ./dataset/electricity/ \  # 数据集路径
  --data_path electricity.csv \         # 数据文件
  --model_id ECL_96_720 \              # 模型ID：输入96步，预测720步
  --model $model_name \                # 模型：TimeXer
  --data custom \                      # 自定义数据类型
  --features MS \                      # 多变量预测单变量
  --seq_len 96 \                       # 输入序列长度：96
  --label_len 48 \                     # 标签长度：48
  --pred_len 720 \                     # 预测长度：720步（最长期预测，约30天）
  --e_layers 3 \                       # 编码器层数：3层（比前面实验更深，处理更复杂的长期依赖）
  --factor 3 \                         # 注意力因子：3
  --enc_in 321 \                       # 编码器输入：321维
  --dec_in 321 \                       # 解码器输入：321维
  --c_out 321 \                        # 输出：321维
  --des $des \                         # 实验描述
  --d_model 512 \                      # 模型隐藏维度：512（增加模型容量以处理长期预测）
  --itr 1                              # 实验次数：1

# ================================== 实验设计说明 ==================================
# 1. 预测长度递增：96 -> 192 -> 336 -> 720，测试不同预测范围的效果
# 2. 批次大小变化：实验1使用batch_size=4，其他为32，可能因为内存或计算复杂度考虑
# 3. 模型复杂度：最长预测(720步)使用3层编码器和512维隐藏层，增强模型能力
# 4. 固定输入长度：所有实验都使用96步历史数据作为输入
# 5. 电力数据特点：321维特征，适合测试多变量时间序列预测模型的性能
```

## 实现2

```shell
#!/bin/bash

# ================================== 环境配置 ==================================
# 设置使用GPU 0进行训练
export CUDA_VISIBLE_DEVICES=0

# ================================== 全局变量定义 ==================================
# 模型名称：TimeXer（专门用于时间序列预测的Transformer变体）
model_name=TimeXer
# 实验描述：TimeXer模型的多变量预测单变量(MS)任务
des='Timexer-MS'
# 补丁长度：24，TimeXer模型将时间序列分割成长度为24的补丁进行处理
patch_len=24

# ================================== 实验1：挪威电力市场(NP) ==================================
# 挪威(Nord Pool)电力价格预测：使用一周数据预测一天价格
python3 -u run.py \
  --is_training 1 \                     # 训练模式
  --task_name long_term_forecast \      # 长期预测任务
  --root_path ./dataset/EPF/ \          # EPF(电力价格预测)数据集根目录
  --data_path NP.csv \                  # 挪威电力价格数据文件
  --model_id NP_168_24 \               # 模型标识：NP(挪威)_输入168小时_预测24小时
  --model $model_name \                # 使用TimeXer模型
  --data custom \                      # 自定义数据加载器
  --features MS \                      # 多变量输入预测单变量输出（价格预测）
  --seq_len 168 \                      # 输入序列长度：168小时（1周的小时数据）
  --pred_len 24 \                      # 预测长度：24小时（1天）
  --e_layers 3 \                       # 编码器层数：3层（较复杂的模型结构）
  --enc_in 3 \                         # 编码器输入维度：3个特征（价格+2个辅助特征）
  --dec_in 3 \                         # 解码器输入维度：3个特征
  --c_out 1 \                          # 输出维度：1（只预测电力价格）
  --des $des \                         # 实验描述
  --patch_len $patch_len \             # 补丁长度：24
  --d_model 512 \                      # 模型隐藏维度：512
  --d_ff 512 \                         # 前馈网络维度：512
  --batch_size 4 \                     # 批次大小：4（小批次，可能因为数据复杂度高）
  --itr 1                              # 实验重复次数：1

# ================================== 实验2：美国PJM电力市场 ==================================
# PJM电力价格预测：美国东部最大的电力市场
python3 -u run.py \
  --is_training 1 \                     # 训练模式
  --task_name long_term_forecast \      # 长期预测任务
  --root_path ./dataset/EPF/ \          # EPF数据集路径
  --data_path PJM.csv \                 # PJM电力市场数据
  --model_id PJM_168_24 \              # 模型ID：PJM_一周输入_一天预测
  --model $model_name \                # TimeXer模型
  --data custom \                      # 自定义数据处理
  --features MS \                      # 多变量预测单变量
  --seq_len 168 \                      # 输入：168小时历史数据
  --pred_len 24 \                      # 预测：未来24小时
  --e_layers 3 \                       # 编码器：3层
  --enc_in 3 \                         # 输入特征：3维
  --dec_in 3 \                         # 解码器输入：3维
  --c_out 1 \                          # 输出：1维（价格）
  --des $des \                         # 实验描述
  --patch_len $patch_len \             # 补丁长度：24
  --d_model 512 \                      # 模型维度：512
  --batch_size 16 \                    # 批次大小：16（比NP更大，数据可能更规律）
  --itr 1                              # 实验次数：1

# ================================== 实验3：比利时电力市场(BE) ==================================
# 比利时电力价格预测
python3 -u run.py \
  --is_training 1 \                     # 训练模式
  --task_name long_term_forecast \      # 长期预测
  --root_path ./dataset/EPF/ \          # 数据集路径
  --data_path BE.csv \                  # 比利时电力数据
  --model_id BE_168_24 \               # 模型ID：比利时_168小时输入_24小时预测
  --model $model_name \                # TimeXer模型
  --data custom \                      # 自定义数据类型
  --features MS \                      # 多变量预测单变量
  --seq_len 168 \                      # 输入序列：168小时
  --pred_len 24 \                      # 预测长度：24小时
  --e_layers 2 \                       # 编码器层数：2层（比NP/PJM少1层，可能数据复杂度较低）
  --enc_in 3 \                         # 编码器输入：3维
  --dec_in 3 \                         # 解码器输入：3维
  --c_out 1 \                          # 输出：1维
  --des $des \                         # 实验描述
  --patch_len $patch_len \             # 补丁长度：24
  --d_model 512 \                      # 模型维度：512
  --d_ff 512 \                         # 前馈网络维度：512
  --batch_size 16 \                    # 批次大小：16
  --itr 1                              # 实验次数：1

# ================================== 实验4：法国电力市场(FR) ==================================
# 法国电力价格预测
python3 -u run.py \
  --is_training 1 \                     # 训练模式
  --task_name long_term_forecast \      # 长期预测任务
  --root_path ./dataset/EPF/ \          # EPF数据集
  --data_path FR.csv \                  # 法国电力数据
  --model_id FR_168_24 \               # 法国电力_168输入_24预测
  --model $model_name \                # TimeXer模型
  --data custom \                      # 自定义数据
  --features MS \                      # 多变量预测单变量
  --seq_len 168 \                      # 输入：一周数据
  --pred_len 24 \                      # 预测：一天数据
  --e_layers 2 \                       # 编码器：2层
  --enc_in 3 \                         # 输入维度：3
  --dec_in 3 \                         # 解码器输入：3
  --c_out 1 \                          # 输出：1（电力价格）
  --des $des \                         # 实验描述
  --patch_len $patch_len \             # 补丁长度：24
  --batch_size 16 \                    # 批次大小：16
  --d_model 512 \                      # 模型维度：512
  --itr 1                              # 实验次数：1

# ================================== 实验5：德国电力市场(DE) ==================================
# 德国电力价格预测
python3 -u run.py \
  --is_training 1 \                     # 训练模式
  --task_name long_term_forecast \      # 长期预测
  --root_path ./dataset/EPF/ \          # 数据集路径
  --data_path DE.csv \                  # 德国电力数据
  --model_id DE_168_24 \               # 德国_168小时输入_24小时预测
  --model $model_name \                # TimeXer模型
  --data custom \                      # 自定义数据处理
  --features MS \                      # 多变量预测单变量
  --seq_len 168 \                      # 输入序列：168小时（1周）
  --pred_len 24 \                      # 预测序列：24小时（1天）
  --e_layers 1 \                       # 编码器层数：1层（最简单的配置，可能德国数据最规律）
  --enc_in 3 \                         # 编码器输入：3维特征
  --dec_in 3 \                         # 解码器输入：3维
  --c_out 1 \                          # 输出：1维（电力价格）
  --des $des \                         # 实验描述
  --patch_len $patch_len \             # 补丁长度：24小时
  --batch_size 4 \                     # 批次大小：4（最小批次，可能因为内存限制）
  --d_model 512 \                      # 模型隐藏维度：512
  --itr 1                              # 实验次数：1

# ================================== 实验设计分析 ==================================
# 1. **统一的预测任务**：所有实验都是用1周历史数据预测未来1天的电力价格
# 2. **数据特征**：每个市场都有3个输入特征，预测1个目标（电力价格）
# 3. **模型复杂度递减**：
#    - NP, PJM: 3层编码器（复杂市场，需要更深的模型）
#    - BE, FR: 2层编码器（中等复杂度）
#    - DE: 1层编码器（相对简单，可能数据模式较规律）
# 4. **批次大小调整**：
#    - NP, DE: batch_size=4（可能数据更复杂或内存限制）
#    - PJM, BE, FR: batch_size=16（标准批次大小）
# 5. **patch_len=24**：将时间序列分成24小时的补丁，符合电力市场的日周期特性
# 6. **跨国比较**：可以分析不同国家/地区电力市场的预测难度和模式差异
```

