# 猫狗猪图像分类项目

本项目基于深度学习技术，使用ResNet50模型实现对猫、狗和猪图像的自动分类识别。项目包含完整的数据预处理、模型训练、验证和部署流程。

## 项目概述

这是一个使用PyTorch框架构建的图像分类项目，能够识别并分类三种动物：猫、狗和猪。项目使用了预训练的ResNet50模型作为基础架构，并在特定数据集上进行微调以适应三分类任务。

## 功能特点

- 支持猫、狗、猪三类动物图像识别
- 基于ResNet50深度学习模型
- 提供Gradio图形界面用于图像上传和预测
- 使用SwanLab进行训练过程可视化和日志记录
- 支持多种计算设备（CUDA、MPS、CPU）

## 工作原理

- 采用 **前后端一体化架构设计开发**，基于 **PyTorch** + **Gradio** + **SwanLab** 构建完整的训练、验证和推理流程；
- 系统核心功能拆分为：图像预处理服务、模型训练服务、模型推理服务、用户界面服务；
- 基于 **ResNet50** 深度学习模型实现图像分类，通过 **torchvision** 进行图像预处理和数据增强；
- 采用 **迁移学习** 技术，利用在 ImageNet 上预训练的 ResNet50 权重进行模型初始化，提升训练效率和准确率；
- 利用 **SwanLab** 实现实验过程可视化监控，记录训练损失和验证准确率等关键指标；
- 基于 **Gradio** 构建交互式用户界面，支持图像上传和实时分类预测，提升用户体验；
- 支持 **多平台部署**，自动适配 CUDA、MPS 和 CPU 等不同计算设备环境。

## 项目结构

```
.
├── app.py                  # Gradio应用界面
├── train_2.py              # 模型训练脚本
├── load_datasets_2.py      # 数据集加载器
├── train.csv               # 训练集标签文件
├── val.csv                 # 验证集标签文件
├── train/                  # 训练数据集目录
├── val/                    # 验证数据集目录
├── checkpoint/             # 模型权重保存目录
└── README.md               # 项目说明文档
```

## 环境依赖

- Python 3.7+
- PyTorch 1.0+
- torchvision
- gradio
- swanlab
- pillow

安装依赖包：
```bash
pip install torch torchvision gradio swanlab pillow
```

## 使用方法

### 1. 数据准备

项目使用标准的猫、狗、猪图像数据集，按照以下结构组织：

```
train/
├── cat/
├── dog/
└── pig/

val/
├── cat/
├── dog/
└── pig/
```

[train.csv](file:///d%3A/project/dogcatpig/train.csv) 和 [val.csv](file:///d%3A/project/dogcatpig/val.csv) 文件包含图像路径和对应标签：
- 0: 猫
- 1: 狗
- 2: 猪

### 2. 模型训练

运行训练脚本开始训练模型：

```bash
python train_2.py
```

训练参数：
- 模型: ResNet50 (预训练权重)
- 优化器: Adam
- 学习率: 0.0001
- 批次大小: 8
- 训练轮数: 20

训练过程中会使用SwanLab记录训练损失和验证准确率，模型权重将保存在 [checkpoint/](file:///d%3A/project/dogcatpig/checkpoint/) 目录下。

### 3. 启动预测界面

训练完成后，运行以下命令启动Gradio界面进行图像分类预测：

```bash
python app.py
```

在界面中上传图像即可获得分类结果和各类别概率。

## 模型性能

模型在验证集上达到良好的分类准确率。通过迁移学习和微调，ResNet50网络能够有效提取图像特征并准确分类猫、狗和猪图像。

## 注意事项

1. 确保数据集路径正确配置
2. 根据硬件配置选择合适的计算设备（GPU/CPU）
3. 模型权重文件路径需要在 [app.py](file:///d%3A/project/dogcatpig/app.py) 中正确设置
4. 如需重新训练，确保 [checkpoint/](file:///d%3A/project/dogcatpig/checkpoint/) 目录有写入权限

## 项目扩展

可根据需要进行以下扩展：
- 增加更多动物类别
- 尝试其他深度学习模型
- 优化数据增强策略
- 部署到云端提供API服务