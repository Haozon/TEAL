import transformers
import torch
import torch.nn as nn
from transformers import AutoConfig
from collections import defaultdict

import os

class SparsifyFn(nn.Module):
    """
    稀疏化函数模块，用于对神经网络的激活值进行稀疏化处理
    
    该模块通过设置阈值来过滤激活值，只保留绝对值大于阈值的激活值，
    其余激活值置零，从而实现激活值的稀疏化，减少计算量。
    """

    def __init__(self, distr, init_sparsity=None, init_threshold=None, apply_prefill=True):
        """
        初始化稀疏化函数
        
        Args:
            distr: 激活值的分布对象，用于计算稀疏化阈值，即 Distribution 对象
            init_sparsity: 初始稀疏度 (0-1之间)，与 init_threshold 互斥
            init_threshold: 直接指定的初始阈值，与 init_sparsity 互斥
            apply_prefill: 是否在预填充阶段应用稀疏化，默认为 True
        """
        super(SparsifyFn, self).__init__()

        # 确保稀疏度和阈值参数不能同时指定
        assert init_sparsity is None or init_threshold is None, "init_sparsity and init_threshold cannot both be specified"

        # 根据输入参数计算初始阈值
        if init_sparsity is not None:
            # 通过分布的逆累积分布函数计算阈值
            # 0.5 + init_sparsity/2 确保中心对称的稀疏化
            thresh = distr.icdf(0.5 + init_sparsity/2)
        elif init_threshold is not None:
            # 直接使用指定的阈值
            thresh = init_threshold
        else:
            # 默认情况：无稀疏化
            init_sparsity = 0
            thresh = 0

        # 将阈值注册为缓冲区，不参与梯度计算但会随模型保存/加载
        self.register_buffer("a", torch.tensor([thresh]).to(torch.float16))

        # 保存分布对象和预填充设置
        self.distr = distr
        self.apply_prefill = apply_prefill

    def set_threshold(self, sparsity):
        """
        动态设置稀疏化阈值
        
        Args:
            sparsity: 目标稀疏度 (0-1之间)
        """
        # 根据稀疏度计算新的阈值
        self.threshold = self.distr.icdf(0.5 + sparsity/2).item() if sparsity != 0.0 else 0.0
        # 记录当前稀疏度级别
        self.sparsity_level = sparsity

    def forward(self, x):
        """
        前向传播，对输入进行稀疏化处理
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
            
        Returns:
            稀疏化后的张量
        """
        
        # 预填充阶段的稀疏化处理
        # NOTE: 作者注释提到应该稀疏化99%的token而不是50%
        # 但论文评估时使用的是50%，发现预填充稀疏化现象后才意识到这个问题
        if x.size(1) > 1 and self.apply_prefill:
            # 计算序列长度的一半
            half_seq_len = x.size(1) // 2
            # 可选：稀疏化99%的token（被注释掉）
            # half_seq_len = int(0.99 * x.size(1))
            
            # 只对序列的后半部分进行稀疏化
            last_context = x[:, -half_seq_len:, :]
            modified_context = self.apply(last_context)
            
            # 将未稀疏化的前半部分与稀疏化的后半部分拼接
            x = torch.cat((x[:, :-half_seq_len, :], modified_context), dim=1)
            return x

        # 预填充阶段但不应用稀疏化的情况
        if x.size(1) > 1 and not self.apply_prefill:
            # 直接返回原始输入，不做任何处理
            return x

        # 解码阶段：序列长度为1，对单个token进行稀疏化
        assert x.size(1) == 1, "supposedly x is decode only"
        return self.apply(x)

    def apply(self, x):
        """
        实际的稀疏化操作
        
        Args:
            x: 输入张量
            
        Returns:
            稀疏化后的张量，绝对值小于阈值的元素被置零
        """
        # 核心稀疏化逻辑：
        # 1. x.abs() 计算绝对值
        # 2. .gt(self.threshold) 生成布尔掩码，标记大于阈值的元素
        # 3. 与原始张量相乘，保留大于阈值的值，其余置零
        return x.abs().gt(self.threshold) * x

    def get_threshold(self):
        """
        获取当前的稀疏化阈值
        
        Returns:
            当前阈值
        """
        return self.threshold


def interp(x, xp, fp):
    """Custom interpolation function for PyTorch tensors."""
    i = torch.searchsorted(xp, x)
    i = torch.clamp(i, 1, len(xp) - 1)
    
    xp_left = xp[i - 1]
    xp_right = xp[i]
    fp_left = fp[i - 1]
    fp_right = fp[i]
    
    t = (x - xp_left) / (xp_right - xp_left)
    return fp_left + t * (fp_right - fp_left)



'''
核心设计思想
基于直方图的分布建模：使用预计算的激活值直方图来建模实际的激活值分布
核密度估计：通过高斯核平滑离散直方图，得到连续的概率密度函数
高效的分位数计算：使用二分搜索和线性插值快速计算逆CDF
稀疏化支持：为稀疏化算法提供基于分布的阈值计算功能
这个类是稀疏化系统中的关键组件，它将激活值的统计特性转化为可用于阈值计算的数学工具。
'''
class Distribution:
    """
    分布类，用于处理激活值的统计分布
    
    该类基于预先计算的直方图数据，提供概率密度函数(PDF)、
    累积分布函数(CDF)和逆累积分布函数(ICDF)的计算功能。
    主要用于稀疏化过程中根据激活值分布计算合适的阈值。
    """

    def __init__(self, file_path, hidden_type):
        """
        初始化分布对象
        
        Args:
            file_path: 直方图文件的路径
            hidden_type: 隐藏层类型标识符，通常为 'h1' 或 'h2'
                        对应不同的激活值类型（如MLP的不同层）
        """
        self.file_path = file_path
        self.hidden_type = hidden_type  # h1 or h2

        # 从预先保存的直方图文件中加载数据
        histogram = torch.load(f"{self.file_path}/histograms.pt")

        # 提取对应类型的直方图数据
        # bin_centers: 直方图每个区间的中心值
        # counts: 每个区间的计数值
        self.bin_centers, self.counts = histogram[f"{self.hidden_type}_centers"], histogram[self.hidden_type]

        # 计算总计数，用于归一化
        self.total_count = self.counts.sum()
        # 计算累积计数，用于CDF和ICDF计算
        self.cumulative_counts = torch.cumsum(self.counts, dim=0)

    def pdf(self, x, bandwidth=None):
        """
        计算概率密度函数(Probability Density Function)
        使用核密度估计(Kernel Density Estimation)进行平滑处理
        
        Args:
            x: 输入值或值的张量
            bandwidth: 核函数的带宽参数，如果为None则自动计算
            
        Returns:
            对应输入值的概率密度
        """
        # 使用Silverman规则自动计算带宽
        if bandwidth is None:
            # 1.06 * std * n^(-1/5) 是经典的带宽选择公式
            bandwidth = 1.06 * torch.std(self.bin_centers[1:-1]) * (self.total_count-2)**(-1/5)
        
        # 为广播操作准备维度
        bin_centers = self.bin_centers.unsqueeze(1)
        
        # 处理输入格式，确保为张量
        if isinstance(x, float) or isinstance(x, int):
            x = torch.tensor([x])
        else:
            x = x.unsqueeze(0)
        
        # 计算高斯核函数
        # 使用标准正态分布核：exp(-0.5 * ((x-μ)/σ)^2) / (σ * sqrt(2π))
        kernel = torch.exp(-0.5 * ((x - bin_centers) / bandwidth)**2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))
        
        # 加权求和得到概率密度估计
        # 每个核函数乘以对应区间的计数，然后归一化
        pdf = torch.sum(kernel * self.counts.unsqueeze(1), dim=0) / self.total_count
        
        return pdf
    
    def cdf(self, x):
        """
        计算累积分布函数(Cumulative Distribution Function)
        
        Args:
            x: 输入值
            
        Returns:
            P(X <= x) 的概率值
        """
        # 使用线性插值在离散的累积计数中估计连续的CDF值
        return interp(x, self.bin_centers, self.cumulative_counts / self.total_count)
    
    def icdf(self, q):
        """
        计算逆累积分布函数(Inverse Cumulative Distribution Function)
        也称为分位数函数(Quantile Function)
        
        注意：假设分布是零均值单峰分布
        
        Args:
            q: 概率值 (0-1之间)，表示累积概率
            
        Returns:
            对应概率q的分位数值，即满足 P(X <= x) = q 的 x 值
        """
        # 被注释的警告：极端分位数会被截断到最极端的区间
        # if q < 0.01 or q > 0.99:
        #     print(f"WARNING: All outliers clip to the most extreme bin")

        # 将概率转换为目标计数
        target_count = q * self.total_count
        
        # 使用二分搜索找到目标计数在累积计数中的位置
        idx = torch.searchsorted(self.cumulative_counts, target_count)
        
        # 边界情况处理
        if idx == 0:
            # 如果目标计数小于第一个累积计数，返回最小值
            return self.bin_centers[0]
        elif idx == len(self.bin_centers):
            # 如果目标计数大于最后一个累积计数，返回最大值
            return self.bin_centers[-1]
        else:
            # 在两个相邻区间之间进行线性插值
            lower_count = self.cumulative_counts[idx - 1]    # 下界累积计数
            upper_count = self.cumulative_counts[idx]        # 上界累积计数
            lower_value = self.bin_centers[idx - 1]          # 下界值
            upper_value = self.bin_centers[idx]              # 上界值
            
            # 计算插值比例
            fraction = (target_count - lower_count) / (upper_count - lower_count)
            
            # 线性插值得到精确的分位数值
            return lower_value + fraction * (upper_value - lower_value)


class ActivationModule:
    def __init__(self, file_path):
        self.file_path = file_path
        self.activations = defaultdict(list)
        self.histograms = None
        
        # store is to store stuff like position_ids in attn (for convinience, is bad code)
        self.store = {}

    def grab_activations(self, x, key):
        if x.size(1) > 1:  # Check if seq_len > 1
            self.activations[key].append(x.detach().squeeze(0).cpu().float())
    def save_activations(self):
        self.activations = self.combine_activations()
        torch.save(self.activations, f"{self.file_path}/activations.pt")

    def load_activations(self):
        self.activations = torch.load(f"{self.file_path}/activations.pt")

    # NOTE: This doesn't store outlier activation values
    def find_histogram(self, num_bins=10000, outlier_threshold=0.01):
        """
        计算激活值的直方图，用于分析激活值的分布
        
        Args:
            num_bins: 直方图的bin数量，默认10000
            outlier_threshold: 异常值阈值，用于过滤极端值，默认0.01（1%）
        
        Returns:
            dict: 包含每个激活层的直方图计数和bin中心点的字典
        """
        
        # 如果直方图还未计算，则进行初始化
        if self.histograms is None:
            # 合并所有激活值用于细粒度分析，不进行组合
            # for fine-grained analysis, do not combine activations
            self.activations = self.combine_activations()
            self.histograms = {}
        else:
            # 如果已经计算过直方图，直接返回缓存结果
            return self.histograms

        # 清空GPU缓存以释放内存
        torch.cuda.empty_cache()
        
        # 遍历每个激活层的数据
        for key, acts in self.activations.items():
            # 将激活值展平为一维张量，分离梯度并移到GPU
            acts = acts.flatten().detach().to('cuda')
            
            # 对激活值进行排序，便于后续计算分位数
            acts = torch.sort(acts)[0]

            # 计算下界：排除最小的outlier_threshold比例的值
            lower_bound = acts[int(outlier_threshold * len(acts))]
            
            # 计算上界：排除最大的outlier_threshold比例的值
            upper_bound = acts[-int(outlier_threshold * len(acts))]

            # 将数据移回CPU以节省GPU内存
            acts = acts.cpu()

            # 在下界和上界之间创建等间距的bin边界（num_bins-1个间隔）
            main_bins = torch.linspace(lower_bound, upper_bound, num_bins - 1)
            
            # 构建完整的bin边界：包含最小值、主要bins、最大值
            bins = torch.cat([torch.tensor([acts[0]]), main_bins, torch.tensor([acts[-1]])])

            # 计算直方图：统计每个bin中的激活值数量
            counts, _ = torch.histogram(acts, bins=bins)

            # 计算每个bin的中心点坐标
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # 存储直方图计数（转换为float并移到CPU）
            self.histograms[key] = counts.float().cpu()
            
            # 存储bin中心点坐标
            self.histograms[f"{key}_centers"] = bin_centers.float().cpu()
        
        return self.histograms
    
    def save_histogram(self):
        os.makedirs(self.file_path, exist_ok=True)
        torch.save(self.histograms, f"{self.file_path}/histograms.pt")

    def combine_activations(self):
        combined_activations = {}
        for key, acts in self.activations.items():
            combined_activations[key] = torch.cat(acts, dim=0)
        return combined_activations

from transformers import AutoConfig

def get_model_class_name(model_name):
    try:
        # Fetch the model config
        config = AutoConfig.from_pretrained(model_name)
        
        # Get the model class name from the config
        model_class_name = config.architectures[0] if config.architectures else None
        
        return model_class_name
    except Exception as e:
        print(f"Error fetching model class name: {e}")
        return None


def get_sparse_model(model_name, device, histogram_path, **kwargs):
    from teal.model import LlamaSparseForCausalLM, MistralSparseForCausalLM, LlamaSparseConfig, MistralSparseConfig

    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("llama_sparse", LlamaSparseConfig)
    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    class_name = get_model_class_name(model_name)

    assert class_name in ["LlamaForCausalLM", "MistralForCausalLM", "LlamaSparseForCausalLM", "MistralSparseForCausalLM"], f"Model class name {class_name} not supported"

    SparseModel = LlamaSparseForCausalLM if "Llama" in class_name else MistralSparseForCausalLM

    if device == 'auto':
        # multi gpu
        return SparseModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2", histogram_path=histogram_path, **kwargs)
    else:
        return SparseModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device, attn_implementation="flash_attention_2", histogram_path=histogram_path, **kwargs)

def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True, trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer


def get_module_device(module):
    return next(module.parameters()).device




def get_layer_greedy_sparsities(layer_sparsities, results_dir):
    import pandas as pd
    num_layers = len(layer_sparsities)
    projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
    sparsities = {proj: [0.0] * num_layers for proj in projs}
    
    for layer, target_sparsity in enumerate(layer_sparsities):
        file_path = os.path.join(results_dir, f'layer-{layer}', 'results.csv')
        df = pd.read_csv(file_path)
        
        # Find the row with the closest effective sparsity
        closest_row = df.iloc[(df['Effective Sparsity'] - target_sparsity).abs().argsort()[:1]]
        
        for proj in projs:
            sparsities[proj][layer] = closest_row[proj].values[0]
    
    return sparsities