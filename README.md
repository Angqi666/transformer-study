# Transformer Study

This project aims to study and implement text generation and pretraining tasks based on the Transformer model, including the following key modules:

- **Model Construction**: Build a text generation model based on a custom Transformer structure.
- **Data Preprocessing**: Use a tokenizer (e.g., BertTokenizer) to encode Chinese text into fixed-length token sequences (e.g., 512), with padding support.
- **Training Process**: Implement pretraining tasks, supporting both single-machine and distributed training (using DistributedDataParallel), combined with mixed-precision training and gradient accumulation to optimize performance.
- **Text Generation**: Provide an interactive test script that allows users to input Chinese text and generate subsequent content using the model. Multiple decoding strategies (greedy search, temperature adjustment, top-k sampling) are supported to enhance output diversity.

---

## Project Structure

```
├── README.md                   # Project documentation
├── code
│   ├── model.py                # Transformer model definition
│   ├── train.py                # Training script
│   └── test.py                 # Testing script

├── notebook
│   ├── 01_data_preview.ipynb           # Data preview for training
│   ├── 02_data_tokenizer.ipynb         # Tokenizer breakdown and learning
│   ├── 03_embedding.ipynb              # Embedding breakdown and learning
│   ├── 04_positional_encoding.ipynb    # Positional encoding breakdown and learning
│   └── 05_attention_block.ipynb        # Attention block breakdown and learning
└── environment.yml                     # Conda environment packages
```

---

## Environment Requirements

- **Python**: 3.8+
- **PyTorch**: 1.7+
- **Transformers**: Latest version (for loading BertTokenizer, etc.)
Installation can be done via the `.yml` file using Anaconda.

---

## Data Download

[Dataset Download Link](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)

---

## Text Generation & Testing

The testing script `test.py` implements interactive text generation:

- Users input Chinese text directly, and the program internally uses BertTokenizer to encode the text into token sequences.
- The trained Transformer model parameters are loaded, and the model performs autoregressive text generation.
- Temperature adjustment and top-k sampling strategies are adopted (parameters can be adjusted as needed) to improve output diversity and avoid repetitive punctuation or word output.
- Finally, the generated token sequence is decoded back into Chinese text and displayed.

Example command to start testing:

```bash
python test.py
```

After the prompt `请输入中文文本：`, enter a phrase such as "Himalayas" (喜马拉雅), and the model will generate and display the corresponding text.

---

## Summary

This project implements pretraining and text generation tasks using a custom Transformer model and training process. Through efficient data preprocessing, mixed-precision training, gradient accumulation, and distributed training support, it aims to train a text generation model effectively within limited resources. Additionally, an easy-to-use test script is provided for users to input Chinese text and observe the generated results.

We welcome feedback and suggestions to improve the code and methodology to further develop this project together!

---
### Reference:

* https://zhuanlan.zhihu.com/p/104393915

* https://wmathor.com/index.php/archives/1455/

* https://arxiv.org/pdf/1706.03762

* https://github.com/jingyaogong/minimind


