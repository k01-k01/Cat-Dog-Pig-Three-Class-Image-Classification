import csv
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(self, csv_path):
        """
        初始化方法，在创建DatasetLoader实例时调用。

        参数:
        csv_path: 存储数据集相关信息的CSV文件路径。
        """
        
        # 保存传入的CSV文件路径
        self.csv_file = csv_path
        
        # 打开CSV文件并读取数据
        with open(self.csv_file, 'r') as file:
            self.data = list(csv.reader(file))
            

        # 获取当前脚本文件所在目录的路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        

    def preprocess_image(self, image_path):
        """
        对指定路径的图像进行预处理。

        参数:
        image_path: 要处理的图像的相对路径。

        返回:
        经过预处理后的图像数据（张量形式）。
        """
        # 拼接出图像的完整路径
        full_path = os.path.join(self.current_dir, '', image_path)
        # 打开图像并转换为RGB模式
        image = Image.open(full_path).convert('RGB')
        # 定义一系列图像变换操作
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])

        return image_transform(image)

    def __getitem__(self, index):
        """
        根据索引获取数据集中的元素，包括预处理后的图像和对应的标签。

        参数:
        index: 元素在数据集中的索引。

        返回:
        包含预处理后的图像（张量形式）和对应标签（整数形式）的元组。
        """
        # 从数据中获取指定索引的图像路径和标签
        image_path, label = self.data[index]
        # 对图像进行预处理
        image = self.preprocess_image(image_path)
        # 将经过预处理后的图像数据和对应的标签，组成一个元组并返回
        return image, int(label)

    def __len__(self):
        """
        返回数据集的长度，即数据集中元素的个数。

        返回:
        数据集的长度。
        """
        return len(self.data)