import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 数据预处理与加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集标准化参数
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 极简CNN网络结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1：输入1通道，输出16通道，3x3卷积核
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2：输入16通道，输出32通道，3x3卷积核
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 全连接层：32*7*7 -> 10（10分类）
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入尺寸：[batch, 1, 28, 28]
        x = self.pool(self.relu(self.conv1(x)))  # 输出：[batch, 16, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # 输出：[batch, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)  # 展平
        x = self.fc1(x)
        return x

# 3. 训练与评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, epochs=5)
    test(model, test_loader)
