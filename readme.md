# 环境配置
Please refer to [env](openpi/README.md).

# 模型训练
Please refer to [train](docs/2_train.md).


# 模型部署
```
nohup env CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_MEM_FRACTION=.85 uv run --no-sync scripts/serve_policy.py policy:checkpoint --policy.config=pi05_chem --policy.dir=checkpoints/pi05_chem/scoop_right_v1/10000 > output_serve.log 2>&1 &
```
# 数据采集注意事项
Please refer to [data](docs/1_data.md).

# 模型部署前注意事项
Please refer to [deploy](docs/3_deploy.md).

# 模型行为异常的解决方法
Please refer to [bug](docs/4_debug.md) and [problem](docs/5_problem.md).