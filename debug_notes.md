markdown
  
# 调试记录
## 1. 任务一调试
### 问题1：数据下载失败
- 现象：运行代码时提示`MNIST数据集下载超时`
- 原因分析：网络环境限制，无法直接从PyTorch官方源下载
- 修改点：手动下载MNIST数据集到`./data/MNIST/raw/`目录，或使用国内镜像源
  ```python
  # 替代方案：使用torchvision的国内镜像
  datasets.MNIST(root='./data', train=True, download=True, transform=transform,
                  target_transform=None, download=True,
                  url='https://mirrors.tuna.tsinghua.edu.cn/pytorch/datasets/mnist/')
问题2：CUDA内存不足
- 现象：GPU训练时提示 CUDA out of memory 
- 原因分析：批量大小过大，超出GPU显存
- 修改点：将 batch_size 从64调整为32，或使用CPU训练
2. 任务二调试
问题1：输入尺寸不匹配
- 现象：运行LeNet-5时提示 size mismatch 
- 原因分析：经典LeNet-5适配32×32输入，MNIST为28×28，未做padding处理
- 修改点：在Conv1层添加 padding=2 ，保持输入尺寸为28×28，确保后续池化后尺寸为5×5
问题2：训练收敛慢
 - 现象：10轮训练后准确率仅95%
- 原因分析：学习率设置过高（0.01），导致震荡
- 修改点：将学习率调整为0.001，使用Adam优化器，提升收敛速度
