# 🔬 Nano-vLLM 完整测试计划

**原则**：所有数据来自**同一套硬件环境**的完整测试，每个数据都需要说明测试方法和计算方式。

---

## 📋 测试环境规格

### 硬件配置
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: (待确认)
- **RAM**: (待确认)
- **驱动**: NVIDIA Driver 570.124.04
- **CUDA**: 12.8
- **PyTorch**: 2.4.1

### 测试模型
- **Model**: Qwen3-0.6B (或其他需要测试的模型)
- **模型大小**: (待确认)
- **精度**: FP16

### 工作负载
- **并发序列数**: 256
- **输入长度**: 512 tokens
- **输出长度**: 512 tokens
- **总请求数**: 按需确定

---

## 🧪 完整测试套件

### 第一部分：基础性能测试

#### 测试 1.1: 吞吐量测试 (Throughput Benchmark)

**测试目标**: 测量单位时间内处理的 tokens 数

**测试方法**:
```python
def test_throughput():
    """
    测试流程:
    1. 预热: 运行 1 个小批次使 GPU 预热
    2. 计时: 从第一个请求开始计时
    3. 处理: 处理 256 个序列
    4. 计算: tokens/时间 = 吞吐量
    """
    # 记录开始时间
    start_time = time.perf_counter()

    # 处理所有序列
    for seq in sequences:
        output = model.generate(seq)

    # 记录结束时间
    end_time = time.perf_counter()

    # 计算
    total_tokens = sum(len(seq) + output_len for seq in sequences)
    throughput = total_tokens / (end_time - start_time)

    return throughput  # tokens/s
```

**输出数据**:
- `prefill_throughput` (tokens/s)
- `decode_throughput` (tokens/s)
- `end_to_end_throughput` (tokens/s)
- `总耗时` (seconds)

**如何验证**: 运行多次，结果应在 ±5% 范围内

---

#### 测试 1.2: 延迟测试 (Latency Benchmark)

**测试目标**: 测量从输入到第一个输出 token 的时间

**测试方法**:
```python
def test_latency():
    """
    测试流程:
    1. 输入提示词 (512 tokens)
    2. 记录从输入到第一个 token 的时间 (TTFT)
    3. 记录后续每个 token 的平均生成时间 (TPOT)
    """

    # 首 Token 延迟 (Time To First Token)
    start = time.perf_counter()
    first_token = model.generate_one_token(prompt)
    ttft = (time.perf_counter() - start) * 1000  # ms

    # 平均 Token 时间 (Time Per Output Token)
    start = time.perf_counter()
    for i in range(128):  # 生成 128 个 tokens
        token = model.generate_one_token(...)
    tpot = (time.perf_counter() - start) * 1000 / 128  # ms/token

    return ttft, tpot
```

**输出数据**:
- `first_token_latency_ms`
- `time_per_output_token_ms`
- `total_latency_ms` (计算: TTFT + TPOT * output_length)

---

#### 测试 1.3: 内存使用测试 (Memory Profiling)

**测试目标**: 测量各个组件的内存占用

**测试方法**:
```python
def test_memory():
    """
    测试流程:
    1. 加载模型前: 记录基线内存
    2. 加载模型后: 记录模型权重内存
    3. 处理序列后: 记录 KV 缓存内存
    4. 处理过程中: 记录最大激活值内存
    """

    import torch

    # 基线内存
    baseline = torch.cuda.memory_allocated()

    # 加载模型
    model = load_model()
    model_memory = torch.cuda.memory_allocated() - baseline

    # 处理序列
    torch.cuda.reset_peak_memory_stats()
    output = model.generate(sequences)

    # KV 缓存内存 (峰值 - 模型)
    peak_memory = torch.cuda.max_memory_allocated()
    kv_cache_memory = peak_memory - model_memory - baseline

    return model_memory, kv_cache_memory, peak_memory
```

**输出数据**:
- `model_weight_memory_gb`
- `kv_cache_memory_gb`
- `activation_memory_gb`
- `total_peak_memory_gb`

---

### 第二部分：优化验证测试

#### 测试 2.1: 前缀缓存收益测试

**测试目标**: 验证前缀缓存能节省多少计算

**测试方法**:
```python
def test_prefix_cache_benefit():
    """
    测试流程:
    1. 准备 100 个请求，其中 70% 共享前缀
    2. 禁用缓存，测量性能 A
    3. 启用缓存，测试性能 B
    4. 计算改进比例
    """

    # 禁用缓存
    model.cache_enabled = False
    start = time.perf_counter()
    output_no_cache = model.generate_batch(requests_100)
    time_no_cache = time.perf_counter() - start
    tokens_no_cache = sum(len(req) + output_len for req in requests_100)

    # 启用缓存
    model.cache_enabled = True
    start = time.perf_counter()
    output_with_cache = model.generate_batch(requests_100)
    time_with_cache = time.perf_counter() - start
    tokens_with_cache = sum(len(req) + output_len for req in requests_100)

    # 计算节省
    compute_saved = (tokens_no_cache - tokens_with_cache) / tokens_no_cache * 100
    cache_hit_rate = (tokens_no_cache - tokens_with_cache) / tokens_no_cache * 100
    speedup = time_no_cache / time_with_cache

    return compute_saved, cache_hit_rate, speedup
```

**输出数据**:
- `cache_hit_rate` (%)
- `compute_saved_pct` (%)
- `actual_speedup_ratio` (倍数)

**重要**: 这个测试需要**两次运行**，结果应该能**相互验证**

---

#### 测试 2.2: 调度器效率测试

**测试目标**: 验证 Prefill/Decode 分离调度的效果

**测试方法**:
```python
def test_scheduler_efficiency():
    """
    测试流程:
    1. 运行混合调度 (vLLM 方式)
    2. 运行分离调度 (Nano-vLLM 方式)
    3. 比较吞吐量和延迟
    """

    # 混合调度
    mixed_scheduler = MixedScheduler()
    mixed_start = time.perf_counter()
    mixed_output = mixed_scheduler.generate(requests)
    mixed_time = time.perf_counter() - mixed_start

    # 分离调度
    separated_scheduler = SeparatedScheduler()
    sep_start = time.perf_counter()
    sep_output = separated_scheduler.generate(requests)
    sep_time = time.perf_counter() - sep_start

    improvement = (mixed_time - sep_time) / mixed_time * 100

    return improvement, mixed_time, sep_time
```

**输出数据**:
- `prefill_throughput`
- `decode_throughput`
- `improvement_vs_mixed_scheduler` (%)

---

#### 测试 2.3: CUDA 图捕获效益测试

**测试目标**: 验证 CUDA 图捕获能节省多少 CPU 开销

**测试方法**:
```python
def test_cuda_graph_benefit():
    """
    测试流程:
    1. 禁用 CUDA 图，测量 decode 阶段性能 A
    2. 启用 CUDA 图，测量 decode 阶段性能 B
    3. 计算改进
    """

    # 禁用 CUDA 图
    model.use_cuda_graph = False
    start = time.perf_counter()
    for i in range(decode_iterations):
        token = model.decode_step()
    time_no_graph = time.perf_counter() - start

    # 启用 CUDA 图
    model.use_cuda_graph = True
    start = time.perf_counter()
    for i in range(decode_iterations):
        token = model.decode_step()
    time_with_graph = time.perf_counter() - start

    improvement = (time_no_graph - time_with_graph) / time_no_graph * 100

    return improvement, time_no_graph, time_with_graph
```

**输出数据**:
- `cpu_overhead_reduction` (%)
- `throughput_improvement` (%)

---

### 第三部分：并行扩展性测试

#### 测试 3.1: 单卡基准

**测试目标**: 建立 1 卡的性能基线

**测试方法**:
```python
# 仅使用 GPU 0
torch.cuda.set_device(0)
throughput_1gpu = run_benchmark()
```

---

#### 测试 3.2: 多卡强缩放

**测试目标**: 固定问题规模，增加 GPU

**测试方法**:
```python
def test_strong_scaling():
    """
    强缩放: 问题规模固定，增加 GPU 数

    理想情况: N 个 GPU 应该有 N 倍的吞吐量
    实际情况: 通信开销会降低效率
    """

    baseline_throughput = throughput_1gpu  # 1362.13 tokens/s (从 1 GPU)

    for num_gpus in [1, 2, 4, 8]:
        throughput = run_benchmark_with_n_gpus(num_gpus)
        efficiency = throughput / (baseline_throughput * num_gpus) * 100
        # 记录: throughput, efficiency
```

**输出数据**:
- `1gpu_throughput`
- `2gpu_throughput` 和 `2gpu_efficiency`
- `4gpu_throughput` 和 `4gpu_efficiency`
- `8gpu_throughput` 和 `8gpu_efficiency`

---

#### 测试 3.3: 多卡弱缩放

**测试目标**: 每 GPU 工作量固定，增加 GPU

**测试方法**:
```python
def test_weak_scaling():
    """
    弱缩放: 每个 GPU 的工作量固定，增加 GPU 数

    理想情况: 总吞吐量应该接近线性增长
    """

    for num_gpus in [1, 2, 4, 8]:
        # 每个 GPU 处理相同数量的序列
        sequences_per_gpu = 32
        total_sequences = sequences_per_gpu * num_gpus

        throughput = run_benchmark_with_n_gpus(num_gpus, total_sequences)
        # 记录: throughput (应该接近线性增长)
```

**输出数据**:
- `1gpu_throughput`
- `2gpu_throughput`
- `4gpu_throughput`
- `8gpu_throughput`

---

### 第四部分：代码质量测试

#### 测试 4.1: 代码行数统计

**测试方法**:
```python
def analyze_code():
    """
    统计代码质量指标
    """
    total_lines = 0
    total_functions = 0
    total_classes = 0

    for python_file in all_python_files:
        with open(python_file) as f:
            content = f.read()
            lines = len(content.split('\n'))
            functions = content.count('def ')
            classes = content.count('class ')

            total_lines += lines
            total_functions += functions
            total_classes += classes

    return {
        'total_lines': total_lines,
        'total_functions': total_functions,
        'total_classes': total_classes,
        'avg_lines_per_function': total_lines / total_functions,
        'cyclomatic_complexity': calculate_complexity()
    }
```

**输出数据**:
- `total_loc`
- `function_count`
- `class_count`
- `avg_complexity`

---

## 📊 期望的输出格式

每个测试应该生成：

```json
{
  "test_name": "吞吐量测试",
  "test_date": "2026-02-19",
  "hardware": "RTX 3090",
  "method": "处理 256 个序列，每个 512+512 tokens",
  "parameters": {
    "batch_size": 256,
    "input_length": 512,
    "output_length": 512
  },
  "results": {
    "prefill_throughput": 395297.08,  // tokens/s
    "decode_throughput": 131072000.0,  // tokens/s
    "e2e_throughput": 790444.13,  // tokens/s
    "total_time": 0.3316  // seconds
  },
  "how_obtained": "使用 torch.cuda.synchronize() 精确计时，处理完整的生成流程",
  "verification": "运行 3 次，结果在 ±5% 范围内"
}
```

---

## ✅ 测试检查清单

- [ ] 所有测试使用相同硬件 (RTX 3090)
- [ ] 每个数据都记录了获取方法
- [ ] 每个数据都可以独立验证
- [ ] 没有引用外部数据（如 README 中的数据）
- [ ] 禁用/启用功能的对比测试都成对出现
- [ ] 所有计时都使用 `torch.cuda.synchronize()`
- [ ] 关键测试运行多次验证稳定性

---

## 🎯 最终输出

所有测试完成后，生成一份文档，清楚地说明：

```markdown
# Nano-vLLM 性能评估 - 完整测试报告

## 1. 推理吞吐量

**数据**: 1434.13 tokens/s

**获取方法**:
- 处理 256 个并发序列
- 每个序列 512 token 输入 + 512 token 输出
- 使用 torch.cuda.synchronize() 精确计时
- 总共处理 262,144 tokens，耗时 0.3316 秒
- 计算: 262,144 tokens / 0.3316 s = 790,444 tokens/s (端到端)

**验证方法**:
- 运行 3 次，结果分别为: xxx, xxx, xxx
- 波动范围: ±3% (可接受)

## 2. 前缀缓存收益

**数据**: 55.92% 计算节省，缓存命中率 55.92%

**获取方法**:
- 准备 100 个请求
- 其中 70% 请求共享 80% 的前缀
- 禁用缓存: 处理 51,200 tokens
- 启用缓存: 处理 22,570 tokens
- 计算: (51,200 - 22,570) / 51,200 = 55.92%

... 以此类推
```

---

这就是我的完整测试计划。所有数据都应该来自这套完整的测试框架。

你认为这个计划合理吗？还是需要调整某些测试方法？
