import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 避免加载损坏的图像时崩溃
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 加载预处理后的数据
file_path = 'successful_movies.csv'
df = pd.read_csv(file_path)

# 处理 genres 列，使其适合训练分类模型
df['genres'] = df['genres'].apply(lambda x: [] if isinstance(x, float) else x.strip('[]').replace("'", "").split(', '))
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres'])

# 定义函数：清理文件名
def sanitize_filename(filename):
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

# 图像数据集类
class MoviePosterDataset(Dataset):
    def __init__(self, df, poster_dir, transform=None):
        self.df = df
        self.poster_dir = poster_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        poster_path = os.path.join(self.poster_dir, f"{sanitize_filename(row['title'])}.jpg")
        try:
            image = Image.open(poster_path).convert("RGB")
        except (OSError, IOError):
            # 如果图像损坏，返回一个黑色的图像
            image = Image.new("RGB", (224, 224))

        label = torch.tensor(genres_encoded[idx], dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据预处理和加载
poster_dir = 'posters'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = MoviePosterDataset(df, poster_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 创建保存目录
save_dir = 'RESNET'
os.makedirs(save_dir, exist_ok=True)

# 定义ResNet模型
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

num_classes = genres_encoded.shape[1]
model = ResNetModel(num_classes=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 使用混合精度训练
scaler = torch.cuda.amp.GradScaler()

# 训练模型
def train_model():
    epochs = 10
    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播和混合精度计算
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'resnet_classifier.pth'))
    joblib.dump(mlb, os.path.join(save_dir, 'mlb.pkl'))

    # 绘制训练损失
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.show()

# 评估模型
def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_preds_binary = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, all_preds_binary)
    f1 = f1_score(all_labels, all_preds_binary, average='micro')
    precision = precision_score(all_labels, all_preds_binary, average='micro')
    recall = recall_score(all_labels, all_preds_binary, average='micro')

    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'F1 Score: {f1:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')

    # 绘制评估指标
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [accuracy, f1, precision, recall]
    plt.figure()
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    plt.savefig(os.path.join(save_dir, 'evaluation_metrics.png'))
    plt.show()

if __name__ == '__main__':
    train_model()
    evaluate_model()
