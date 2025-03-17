import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import sys
from model import Transformer  # 确保 model.py 中定义了 Transformer 类
from transformers import BertTokenizer

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型参数
model = Transformer().to(device)
model.load_state_dict(torch.load("/egr/research-slim/liangqi1/LLM/transformer-study/checkpoint/decoder_transformer.pth", map_location=device))
model.eval()

# 加载对应的 tokenizer（这里以 bert-base-chinese 为例，根据实际情况替换）
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def generate_text(model, start_sequence, max_length=50, temperature=1.0, top_k=20):
    """
    根据给定的 token 序列生成文本，采用温度调节和 top-k 采样策略，
    避免贪婪搜索可能导致的重复生成问题。
    如果生成的 token 为停止符（例如 [SEP]、[PAD] 或者 0），则终止生成。
    """
    generated_sequence = start_sequence.copy()
    
    for _ in range(max_length):
        # 将当前生成的 token 序列转换为 tensor，并添加 batch 维度
        input_tensor = torch.tensor(generated_sequence).unsqueeze(0).to(device)
        # 获取模型输出，假设输出形状为 [batch, vocab_size]
        output_logits = model(input_tensor)
        # 对 logits 应用温度调节
        logits = output_logits[0] / temperature
        
        # 采用 top-k 采样：选出概率最高的 top_k 个 token，并从中采样
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        probabilities = torch.softmax(values, dim=-1)
        next_token = indices[torch.multinomial(probabilities, num_samples=1)].item()
        generated_sequence.append(next_token)
        
        # 如果生成了停止符，则结束生成
        if next_token in [tokenizer.sep_token_id, tokenizer.pad_token_id, 0]:
            break
    
    return generated_sequence

if __name__ == "__main__":
    # 获取用户输入的中文文本
    user_input = input("请输入中文文本：")
    # 使用 tokenizer 对中文文本进行编码，添加必要的特殊 token（如 [CLS] 和 [SEP]）
    input_tokens = tokenizer.encode(user_input, add_special_tokens=True)
    # 根据输入 token 生成后续文本
    generated_tokens = generate_text(model, input_tokens, max_length=50, temperature=2.0, top_k=50)
    # 将生成的 token 序列解码为中文文本，跳过特殊 token
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("生成的文本：", generated_text)
