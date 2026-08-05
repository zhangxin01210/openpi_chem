# OpenPI 离线推理对比工具

这个工具用于把 LeRobot 数据集中的 episode 离线送入 OpenPI policy server，并把返回的 action chunk 与数据集真值 action 对比。主要用途是快速判断 checkpoint 推理链路、相机输入、动作维度和动作语义是否正常。

## 核心逻辑

1. 读取 LeRobot 数据集的 `meta/info.json`、`meta/episodes/*.parquet`、`data/**/*.parquet` 和可选 `meta/tasks.parquet`
2. 自动发现 image/video 相机，并按相机名称或 `--camera-map` 映射成常用 server key
3. 构造 observation：`state`、`prompt`、原始数据集相机 key，以及需要补齐的 `camera_head/camera_left_wrist/camera_right_wrist`
4. 通过 WebSocket 调用 OpenPI policy server，读取响应中的 `action`
5. 对齐预测维度和真值维度，计算 horizon-0、ensemble、不同 horizon offset 的指标
6. 保存 `npz/json/png` 结果用于排查训练或推理问题

## 依赖

```bash
pip install numpy polars matplotlib msgpack websockets pillow opencv-python-headless av
```

建议安装 PyAV。没有 PyAV 时会退回到 OpenCV/ffmpeg 的逐帧读取，长视频会慢很多。AV1 视频建议系统安装带 `libdav1d` 的 `ffmpeg`。

## 使用

先启动 policy server：

```bash
cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=.85 uv run --no-sync scripts/serve_policy.py policy:checkpoint --policy.config=pi0_chem --policy.dir=/path/to/checkpoint --port 8000

# example
env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.85 uv run --no-sync scripts/serve_policy.py policy:checkpoint --policy.config=pi05_chem --policy.dir=checkpoints/pi05_chem/scoop_right_v1/10000
```

再运行离线对比：

```bash
python tools/openpi_compare/compare_openpi_vs_dataset.py \
    --dataset-root /path/to/lerobot_dataset \
    --host localhost --port 8000 \
    --episode-index 0 \
    --prompt "pick up the orange cob" \
    --save-dir ./openpi_compare_results
```

只检查数据读取和 observation 构造，不连接 server：

```bash
python tools/openpi_compare/compare_openpi_vs_dataset.py \
    --dataset-root /path/to/lerobot_dataset \
    --episode-index 0 \
    --max-frames 5 \
    --dry-run \
    --skip-preview
```

结果会写入 `--save-dir/episode_XXXX/`。

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-root` | 必填 | LeRobot 数据集目录 |
| `--host` | `localhost` | policy server 主机 |
| `--port` | `8000` | policy server 端口 |
| `--api-key` | `OPENPI_API_KEY` | 可选 API key；不会写入运行配置 |
| `--episode-index` | `0` | 要评估的 episode |
| `--prompt` | 数据集任务文本 | 语言指令；不传时尝试从数据集读取 |
| `--save-dir` | `./openpi_compare_results` | 输出根目录 |
| `--max-frames` | 全部 | 最多处理多少帧 |
| `--stride` | `1` | 帧采样步长 |
| `--start-sec` | 无 | episode 内起始时间 |
| `--end-sec` | 无 | episode 内结束时间 |
| `--use-original-image-size` | `False` | 不 resize，按原始图像尺寸发送 |
| `--dry-run` | `False` | 只加载数据和构造 observation |
| `--skip-preview` | `False` | 不生成相机预览图 |
| `--verbose` | `False` | 输出 debug 日志 |

## 输出文件(看summary.png就行)

| 文件 | 说明 |
|------|------|
| `sample_preview.png` | 数据集实际相机预览 |
| `horizon0_overlay_dimXX.png` | 每个维度的 horizon-0 对比曲线 |
| `ensemble_overlay_dimXX.png` | 每个维度的 ensemble 对比曲线 |
| `offset_mae_curve.png` | 不同 horizon offset 的误差曲线 |
| `error_heatmap.png` | horizon-0 误差热力图 |
| `summary.png` | 综合汇总图 |

## 数据集要求（目前支持lerobot v3.0）

最低需要：

- `meta/info.json`
- `meta/episodes/*.parquet`
- `data/**/*.parquet`

可选：

- `meta/tasks.parquet`
- `videos/<camera_key>/chunk-XXX/file-XXX.mp4`

如果是视频数据集，优先使用 `meta/episodes` 中记录的 `videos/<camera_key>/chunk_index` 和 `file_index` 精确定位 episode 视频文件。
