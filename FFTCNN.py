import torch
import torchvision
import torch.nn as nn
import torch.fft
import math
from torchvision import transforms

class EnhancedFFTTransformer:
    def __call__(self, img):
        """Process image into spatial and frequency components"""
        img_tensor = transforms.functional.to_tensor(img).float()
        
        mag_spectrum = torch.zeros(3, 32, 32)
        phase_spectrum = torch.zeros(3, 32, 32)
        
        for c in range(3):
            channel_data = img_tensor[c]
            fft = torch.fft.fft2(channel_data)
            fft_shift = torch.fft.fftshift(fft)
            mag = torch.abs(fft_shift)
            phase = torch.angle(fft_shift)
            
            # Percentile-based magnitude normalization
            mag = torch.log(1 + mag)
            q05 = torch.quantile(mag, 0.05)
            q95 = torch.quantile(mag, 0.95)
            mag = (mag - q05) / (q95 - q05 + 1e-7)
            
            mag_spectrum[c] = mag.clamp(0, 1)
            phase_spectrum[c] = (phase + math.pi) / (2 * math.pi)
        
        return img_tensor, torch.cat([mag_spectrum, phase_spectrum], dim=0)

class CustomTransform:
    """Apply FFT transform and normalize components"""
    def __init__(self):
        self.fft_transform = EnhancedFFTTransformer()
        self.spatial_normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261]
        )
    
    def __call__(self, img):
        spatial, freq = self.fft_transform(img)
        return self.spatial_normalize(spatial), freq

# Data loading
train_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=CustomTransform()
)

test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=CustomTransform()
)

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False)

class HybridCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Spatial processing branch
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2)
        )
        
        # Frequency processing branch
        self.freq_stream = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128*4*4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_spatial, x_freq):
        spatial_feat = self.spatial_stream(x_spatial)
        freq_feat = self.freq_stream(x_freq)
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.combined(combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epochs):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for (spatial_inputs, freq_inputs), labels in train_loader:
            spatial_inputs = spatial_inputs.to(device)
            freq_inputs = freq_inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spatial_inputs, freq_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        acc = 100. * correct / total
        
        # Validation
        val_loss, val_acc = evaluate()
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f} Acc: {acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for (spatial_inputs, freq_inputs), labels in test_loader:
            spatial_inputs = spatial_inputs.to(device)
            freq_inputs = freq_inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(spatial_inputs, freq_inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(test_loader), 100. * correct / total

if __name__ == "__main__":
    train(epochs=50)