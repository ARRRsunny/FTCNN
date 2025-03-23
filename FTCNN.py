import torch
import torchvision
import torch.nn as nn
import torch.fft
from torchvision import transforms

class FFTTransformer:
    def __init__(self, log_scale=True):
        self.log_scale = log_scale  # 是否使用对数缩放
    
    def __call__(self, img):
        """
        输入: PIL Image (3, 32, 32)
        输出: 频谱图 Tensor (3, 32, 32)
        """
        # 转换为Tensor并转为float类型
        img_tensor = transforms.functional.to_tensor(img).float()
        
        # 对每个通道进行FFT
        spectrum = torch.zeros_like(img_tensor)
        for c in range(3):
            channel_data = img_tensor[c]
            fft = torch.fft.fft2(channel_data)
            fft_shift = torch.fft.fftshift(fft)
            mag = torch.abs(fft_shift)
            
            if self.log_scale:
                mag = torch.log(1 + mag)  # 对数变换增强对比度
            
            # 归一化到[0,1]
            mag = (mag - mag.min()) / (mag.max() - mag.min())
            spectrum[c] = mag
        
        return spectrum

# 数据加载配置
transform = transforms.Compose([
    FFTTransformer(),
    transforms.Normalize((0.5,), (0.5,))  # 单通道归一化
])

# 加载数据集
train_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False)
class SpectralCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 输入: 3通道频谱图
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 保持尺寸
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpectralCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss/len(train_loader)
        acc = 100.*correct/total
        
        # 验证集测试
        val_loss, val_acc = evaluate()
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f} Acc: {acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(test_loader), 100.*correct/total

# 开始训练
train(epochs=20)