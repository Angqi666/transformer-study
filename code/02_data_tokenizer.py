import pandas as pd
from transformers import BertTokenizer
import torch

# Step 1: 读取 CSV 文件，并提取 'text' 列
# Step 1: Load the CSV file and extract the 'text' column
csv_path = '/egr/research-slim/liangqi1/LLM/transformer-study/data/pretrain_data.csv'  # 替换为你的 CSV 文件路径 / Replace with your CSV file path
df = pd.read_csv(csv_path)

# Step 2: 初始化 Tokenizer（这里使用中文预训练模型 bert-base-chinese）
# Step 2: Initialize the tokenizer (using a Chinese pretrained model 'bert-base-chinese' here)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# Step 3: 批量编码文本，获得 token ids, attention mask 等信息
# Step 3: Batch encode the texts to obtain token ids, attention mask, etc.
encoded_texts = tokenizer(
    df['text'].tolist(),   # 将 DataFrame 中 'text' 列转换为列表 / Convert the 'text' column to a list
    padding=True,          # 自动填充到相同长度 / Automatically pad sequences to the same length
    truncation=True,       # 对超过 max_length 的文本进行截断 / Truncate texts longer than max_length
    max_length=512,        # 设置最大长度 / Set maximum length
    return_tensors="pt"    # 返回 PyTorch 的 tensor 格式 / Return results as PyTorch tensors
)

# 打印 input_ids 的形状以确认维度
# Print the shape of input_ids to verify dimensions
print("input_ids shape:", encoded_texts.input_ids.shape)

# Step 4: 保存编码结果到磁盘，方便后续使用
# Step 4: Save the encoded results to disk for later use
save_path = '/egr/research-slim/liangqi1/LLM/transformer-study/data/encoded_texts.pt'  # 替换为你希望保存的路径 / Replace with your desired save path
torch.save(encoded_texts, save_path)
print(f"Encoded texts saved to {save_path}")
