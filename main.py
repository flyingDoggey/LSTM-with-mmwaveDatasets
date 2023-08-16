import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary
# 设置随机种子。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
# 自己定的。一般都是42.
set_seed(42)
# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GestureDataset(Dataset):
    '''加载数据集类'''
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = self._get_file_list()

        # 创建标签到整数的映射字典
        self.label_mapping = {}
        label_names = os.listdir(root_dir)
        for idx, label in enumerate(label_names):
            self.label_mapping[label] = idx

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = pd.read_csv(file_path).drop(["FrameNumber"], axis=1)
        # 获取数据的标签，即文件夹名称
        label = os.path.basename(os.path.dirname(file_path))
        label_id = self.label_mapping[label]

        desired_length = 250
        if len(data) < desired_length:
            # 如果数据长度小于200，则进行填充
            data = self._pad_sequence(data, desired_length)
        else:
            # 如果数据长度大于200，则进行截断
            data = data[:desired_length]

        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)
    
    def _pad_sequence(self, data, desired_length):
        '''数据填充方法'''
        # 获取数据长度
        data_length = len(data)
        # 计算需要填充的长度
        pad_length = desired_length - data_length
        # 生成填充数据
        pad_data = pd.DataFrame([[0, 0, 0, 0, 0, 0]] * pad_length, columns=data.columns)
        # 将填充数据和原始数据拼接
        data = pd.concat([data, pad_data], axis=0)
        return data

    def _get_file_list(self):
        '''获取文件夹内容列表方法'''
        file_list = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
        return file_list


# 指定数据集文件夹路径
train_root = "F:/dataSet/1/1/HMC"
test_root = "F:/dataSet/1/1/HMC_TEST"


# 创建数据集实例
train_dataset = GestureDataset(train_root)
test_dataset = GestureDataset(test_root)


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 增加一层
        #self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        #增加dropout
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        #增加
        #_, (h_n2, _) = self.lstm2(h_n)
        #h_n2 = h_n2[-1]
        h_n = h_n[-1]  # 取LSTM最后一个时间步的隐藏状态作为输出
        out = self.dropout(h_n)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# 设置训练参数
input_size = 6 #输入个数
hidden_size = 128 #隐含层个数
num_classes = 10 #分类数
batch_size = 64
num_epochs = 150 #学习次数
learning_rate = 0.005

# 创建模型实例和优化器
model = LSTMClassifier(input_size, hidden_size, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def main():
# 训练模型
    summary(model, (batch_size, 1, input_size))# 打印模型
    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


# 测试模型
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()