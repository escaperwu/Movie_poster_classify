import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
from transformers import ViTModel


# 定义自定义ViT模型
class CustomViT(nn.Module):
    def __init__(self, num_classes, pretrained_model_name='google/vit-base-patch16-224'):
        super(CustomViT, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

# 加载模型和标签
num_classes = 19  # 根据训练时的num_classes进行修改
model = CustomViT(num_classes=num_classes)
model.load_state_dict(torch.load('vit_classifier.pth'))
model.eval()

# 加载MultiLabelBinarizer
mlb = joblib.load('mlb.pkl')

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
