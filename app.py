
import gradio as gr
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision

def load_model(checkpoint_path, num_classes):
    """
    加载模型函数，根据给定路径和类别数加载模型并设置好。

    参数:
    checkpoint_path：模型权重文件路径。
    num_classes：分类类别数。

    返回:
    加载并设置好的模型。
    """
    # 加载预训练的ResNet50模型
    try:
        use_mps = torch.backends.mps.is_available()
        # 尝试获取是否有苹果芯片相关设备可用于加载模型
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
        # 若有GPU，设置设备为'cuda'用于加载模型
    elif use_mps:
        device = "mps"
        # 若有苹果芯片相关设备，设置为'mps'
    else:
        device = "cpu"
        # 都没有就用'cpu'

    model = torchvision.models.resnet50(weights=None)
    # 创建ResNet50模型，不加载预训练权重
    in_features = model.fc.in_features
    # 获取模型全连接层输入特征数
    model.fc = torch.nn.Linear(in_features, num_classes)
    # 替换全连接层适应分类任务
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # 从指定路径加载模型权重到对应设备
    model.eval()  
    # 设置模型为评估模式
    return model

# 加载图像并执行必要的转换的函数
def process_image(image, image_size):
    """
    处理图像函数，对输入图像做预处理。

    参数:
    image：输入图像。
    image_size：要调整的图像大小。

    返回:
    预处理后的图像。
    """
    # Define the same transforms as used during training
    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # 调整图像大小
        transforms.ToTensor(),
        # 转换图像为张量格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 对图像张量做归一化
    ])
    image = preprocessing(image).unsqueeze(0)
    # 应用预处理并增加一个维度
    return image


# 预测图像类别并返回概率的函数
def predict(image):
    """
    预测图像类别及概率的函数。

    参数:
    image：输入图像。

    返回:
    类别及对应概率的字典。
    """
    classes = {'0': 'cat', '1': 'dog','2':'pig'}  
    # 定义类别字典
    image = process_image(image, 256)  
    # 处理图像，大小设为256
    with torch.no_grad():
        outputs = model(image)
        # 用模型预测图像，不计算梯度
        probabilities = F.softmax(outputs, dim=1).squeeze()
        # 计算类别概率并压缩维度

    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    # 把概率对应到类别上
    return class_probabilities


# 定义到您的模型权重的路径
checkpoint_path = 'D:/project/checkpoint/latest_checkpoint.pth'
num_classes = 3
model = load_model(checkpoint_path, num_classes)

# 定义Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Cat vs Dog vs pig Classifier",
)

if __name__ == "__main__":
    iface.launch()
    # 启动Gradio界面