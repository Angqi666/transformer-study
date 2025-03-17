import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import trange

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用 GPU 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数定义 / Hyperparameters
batch_size = 32  # 训练批次大小 / Training batch size
epochs = 10  # 训练轮数 / Number of training epochs
learning_rate = 1e-4  # 学习率 / Learning rate
n_heads = 8  # 多头注意力的头数 / Number of attention heads


# 数据集类 / Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, encoded_texts):
        self.input_ids = encoded_texts['input_ids']  # 输入 token ID / Tokenized input sequences
        self.attention_masks = encoded_texts['attention_mask']  # 注意力掩码 / Attention masks
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].clone().detach()  # 获取 input_ids / Get input token IDs
        attention_mask = self.attention_masks[idx].clone().detach().unsqueeze(0).repeat(n_heads, 1, 1)  # 确保 mask 维度一致 / Ensure mask shape matches n_heads
        target_ids = torch.tensor(self.input_ids[idx][1:].tolist() + [0])  
    
        return input_ids, attention_mask, target_ids

save_path = '/egr/research-slim/liangqi1/LLM/transformer-study/data/encoded_texts.pt'  # 替换为你的文件路径
encoded_texts = torch.load(save_path)

# 加载预处理数据 / Load Preprocessed Data
# 假设 encoded_texts 是 BertTokenizer 处理后的结果 / Assume encoded_texts is preprocessed by BertTokenizer
train_dataset = TextDataset(encoded_texts)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化 Transformer 模型 / Initialize Transformer Model
model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 交叉熵损失函数 / Cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环 / Training Loop
for epoch in trange(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        inputs, attention_masks, targets = batch  # 获取批次数据 / Get batch data
        inputs, attention_masks, targets = inputs.to(device), attention_masks.to(device), targets.to(device)
        
        optimizer.zero_grad()  # 清空梯度 / Clear previous gradients
        
        outputs = model(inputs)  # 前向传播 / Forward pass
        loss = criterion(outputs, targets.view(-1))  # 计算损失 / Compute loss
        
        loss.backward()  # 反向传播 / Backpropagation
        optimizer.step()  # 更新参数 / Update model weights
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")  # 输出训练损失 / Print training loss

# 保存训练后的模型 / Save the trained model
torch.save(model.state_dict(), "decoder_transformer.pth")
print("Model saved successfully!")

# 评估函数 / Evaluation Function
def generate_text(model, start_sequence, max_length=20):
    model.eval()
    generated_sequence = start_sequence.copy()
    
    for _ in range(max_length):
        input_tensor = torch.tensor(generated_sequence).unsqueeze(0).to(device)
        output_logits = model(input_tensor)
        next_token = torch.argmax(output_logits[-1]).item()  # 选择概率最高的下一个词 / Select most probable next token
        generated_sequence.append(next_token)
        
        if next_token == 0:  # 停止标记 / Stop token
            break
    
    return generated_sequence

# 示例文本生成 / Example text generation
start_seq = [101, 2769, 812, 4495]  # 初始输入 / Initial input
result = generate_text(model, start_seq)
print("Generated sequence:", result)
