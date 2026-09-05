# GLM-OCR 多卡并行部署

自动在多张 GPU 上启动 sglang/vLLM 推理服务，均匀分配图像文件，并行运行 GLM-OCR 流水线以获得最大吞吐量。

每张 GPU 同时承载推理服务（sglang 或 vLLM）和版面检测模型，形成独立的处理单元，GPU 之间零通信开销。

## 特性

- **自动检测 GPU** — 自动发现所有可用 GPU，按空闲显存过滤
- **动态端口分配** — 自动跳过已被占用的端口
- **容错机制** — 失败的 GPU 自动跳过，文件重新分配到健康的 GPU 上
- **全局进度条** — 实时 `tqdm` 进度展示，汇总所有 GPU 的处理进度
- **优雅退出** — `Ctrl+C` 清理所有子进程；双击 `Ctrl+C` 强制终止
- **集中日志** — 所有引擎/Worker 日志保存在 `logs/<时间戳>/` 目录下
- **投机解码** — sglang 和 vLLM 均默认启用 MTP（多 Token 预测）

## 快速开始

```bash
# 使用所有可用 GPU，默认 sglang 引擎
python examples/multi-gpu-deploy/launch.py -i ./images -o ./output -m /path/to/GLM-OCR

# 指定 GPU 并使用 vLLM
python examples/multi-gpu-deploy/launch.py -i ./images -o ./output --engine vllm --gpus 0,1,2,3

# 自定义模型路径和显存阈值
python examples/multi-gpu-deploy/launch.py -i ./images -o ./output -m /path/to/GLM-OCR --min-free-mb 20000
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-i`, `--input` | *必填* | 输入图像文件或目录（支持递归扫描） |
| `-o`, `--output` | `./output` | 输出结果目录 |
| `-m`, `--model` | `zai-org/GLM-OCR` | 模型名称或本地路径 |
| `--engine` | `sglang` | 推理引擎：`sglang` 或 `vllm` |
| `--gpus` | `auto` | GPU 编号（逗号分隔）或 `auto` 自动检测 |
| `--base-port` | `8080` | 推理服务起始端口 |
| `--min-free-mb` | `16000` | 使用 GPU 所需的最小空闲显存（MB） |
| `--timeout` | `600` | 推理服务启动超时时间（秒） |
| `--engine-args` | *无* | 传递给推理引擎的额外参数 |
| `-c`, `--config` | *无* | 自定义 glmocr 配置 YAML 路径 |
| `--log-level` | `WARNING` | Worker 进程的日志级别 |


## 使用示例

### 基本用法

```bash
python examples/multi-gpu-deploy/launch.py -i /data/documents -o /data/results
```

### 使用 vLLM 并指定 GPU

```bash
python examples/multi-gpu-deploy/launch.py \
  -i /data/documents \
  -o /data/results \
  --engine vllm \
  --gpus 0,2,4,6
```

### 自定义引擎参数

```bash
# sglang 设置显存占用比例
python examples/multi-gpu-deploy/launch.py \
  -i /data/documents \
  -o /data/results \
  --engine-args "--mem-fraction-static 0.85"
```

### 使用自定义配置文件

```bash
python examples/multi-gpu-deploy/launch.py \
  -i /data/documents \
  -o /data/results \
  --config my_config.yaml
```

## 日志

所有日志保存在 `logs/<时间戳>/` 目录下：

| 文件 | 内容 |
|---|---|
| `main.log` | 协调器主进程的 stdout/stderr |
| `engine_gpu<N>_port<P>.log` | 各 GPU 的推理引擎输出 |
| `worker_gpu<N>.log` | 各 GPU 的 Worker 进程输出 |
| `failed_files.json` | 汇总的失败文件列表（如有） |

## 常见问题

**Q：某些端口被占用了，还能正常工作吗？**

可以。启动器会从 `--base-port` 开始自动扫描可用端口，跳过所有已被占用的端口。

**Q：某张 GPU 在处理过程中显存不足怎么办？**

该 GPU 上的 Worker 会失败，但其他 GPU 继续处理。失败的文件会记录在 `failed_files.json` 中，方便后续重新处理。

**Q：如何只重跑失败的文件？**

将失败的文件复制到一个目录中，然后重新运行启动器指向该目录即可。

## 文件结构

```
examples/multi-gpu-deploy/
├── launch.py        # 入口文件与命令行参数解析
├── coordinator.py   # 编排器：GPU 检测、引擎/Worker 生命周期管理
├── engine.py        # 推理引擎管理与进度追踪
├── worker.py        # Worker 进程：GLM-OCR 流水线执行
├── gpu_utils.py     # GPU 检测、端口检查、文件分片
├── README.md        # 英文文档
└── README_zh.md     # 本文件（中文文档）
```
