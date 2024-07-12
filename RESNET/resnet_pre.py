import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import joblib

# 定义ResNet模型
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 加载模型和标签
num_classes = 19  # 根据保存的模型输出类别数进行修改
model = ResNetModel(num_classes=num_classes)
model.load_state_dict(torch.load('../resnet_classifier.pth'))
model.eval()

# 加载MultiLabelBinarizer
mlb = joblib.load('../mlb.pkl')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 确保输入图片大小一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 预测函数
def predict_image(image_path):
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批量维度

    # 模型预测
    with torch.no_grad():
        outputs = model(image)

    # 计算概率
    probabilities = torch.sigmoid(outputs).numpy()[0]

    # 获取前三个电影类型及其概率
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    top3_genres = mlb.classes_[top3_indices]
    top3_probabilities = probabilities[top3_indices]

    return top3_genres, top3_probabilities

# 用户输入一张海报
image_path = '../2.jpg'  # 替换为实际的图片路径
top3_genres, top3_probabilities = predict_image(image_path)

# 输出预测结果
for genre, prob in zip(top3_genres, top3_probabilities):
    print(f"Genre: {genre}, Probability: {prob:.4f}")
