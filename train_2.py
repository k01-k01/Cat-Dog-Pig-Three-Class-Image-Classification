
import torch
import torchvision
from torchvision.models import ResNet50_Weights
import swanlab
from torch.utils.data import DataLoader
from load_datasets_2 import DatasetLoader
import os

# 设置SSL上下文，使其能通过不验证证书的https连接（可能为了下载资源等）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 定义训练函数，用来训练模型
def train(model, device, train_dataloader, optimizer, criterion, epoch):
    """
    训练模型的函数，每次调用训练一个轮次。

    参数:
    model：要训练的模型。
    device：训练使用的设备（如cpu、cuda、mps）。
    train_dataloader：训练数据的加载器。
    optimizer：优化器，用于更新模型参数。
    criterion：损失函数，衡量预测与真实值差异。
    epoch：当前训练轮次。
    """
    model.train()  # 设置模型为训练模式

    for iter, (inputs, labels) in enumerate(train_dataloader):
        """
        遍历训练数据加载器，每次获取一批数据和标签。

        iter：批次索引。
        inputs：输入数据。
        labels：对应标签。
        """
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据和标签移到指定设备
        optimizer.zero_grad()  # 清空优化器的梯度
        outputs = model(inputs)  # 用模型预测输出
        loss = criterion(outputs, labels)  # 计算损失值
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 根据梯度更新模型参数

        print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(TrainDataLoader),
                                                                      loss.item()))
        # 打印轮次、批次及损失值信息

        swanlab.log({"train_loss": loss.item()})  # 用swanlab记录训练损失

# 定义测试函数，用于评估模型在测试数据上的性能
def test(model, device, test_dataloader, epoch):
    """
    测试模型性能的函数。

    参数:
    model：要测试的模型。
    device：测试使用的设备。
    test_dataloader：测试数据的加载器。
    epoch：当前训练轮次（可能用于记录测试与训练关系）。
    """
    model.eval()  # 设置模型为评估模式

    correct = 0
    total = 0

    with torch.no_grad():
        """
        在不计算梯度情况下进行测试操作。
        """
        for inputs, labels in test_dataloader:
            """
            遍历测试数据加载器获取数据和标签。
            """
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据和标签移到指定设备
            outputs = model(inputs)  # 用模型预测输出
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)  # 统计总样本数
            correct += (predicted == labels).sum().item()  # 统计预测正确样本数

    accuracy = correct / total * 100  # 计算准确率

    print('Accuracy: {:.2f}%'.format(accuracy))  # 打印准确率

    swanlab.log({"test_acc": accuracy})  # 用swanlab记录测试准确率

if __name__ == "__main__":
    num_epochs = 20  # 设置训练总轮次数为20
    lr = 1e-4  # 设置学习率为0.0001
    batch_size = 8  # 设置每批数据量为8

    num_classes = 3  # 设置分类任务的类别数为3

    # 设置device，根据设备可用性确定训练测试用设备
    try:
        use_mps = torch.backends.mps.is_available()
        # 尝试查看是否有苹果芯片相关设备可用
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"  # 若有GPU，用GPU作为设备
    elif use_mps:
        device = "mps"  # 若有苹果芯片相关设备，用其作为设备
    else:
        device = "cpu"  # 否则用CPU作为设备

    # 初始化swanlab，记录实验相关信息
    swanlab.init(
        experiment_name="ResNet50",
        description="Train ResNet50 for cat and dog classification.",
        config={
            "model": "resnet50",
            "optim": "Adam",
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_class": num_classes,
            "device": device,
        }
    )

    TrainDataset = DatasetLoader("D:/project/dogcatpig/train.csv")
    # 用DatasetLoader加载训练数据集

    ValDataset = DatasetLoader("D:/project/dogcatpig/val.csv")
    # 用DatasetLoader加载验证数据集

    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    # 创建训练数据加载器，批次大小为8并打乱数据

    ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)
    # 创建验证数据加载器，批次大小为8不打乱数据

    # 载入ResNet50模型并使用指定预训练权重
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # 将模型全连接层替换为适合3分类任务的结构
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    model.to(torch.device(device))  # 将模型移到指定设备上

    criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 定义Adam优化器，设置学习率

    # 开始训练，循环多个轮次
    for epoch in range(1, num_epochs + 1):
        train(model, device, TrainDataLoader, optimizer, criterion, epoch)  # 训练一个轮次

        if epoch % 4 == 0:  # 每4个轮次进行一次测试
            accuracy = test(model, device, ValDataLoader, epoch)

    # 保存模型权重文件，若目录不存在则创建
    if not os.path_exists("D:/project/checkpoint"):
        os.makedirs("checkpoint")
    torch.save(model.state_dict(), 'D:/project/checkpoint/latest_checkpoint.pth')
    print("Training complete")