import json

import torch
from utils import load_json

label2idx_path = "./data/label2idx.json"
train_data_path = "./data/train.txt"
dev_data_path = "./data/dev.txt"
test_data_path = "./data/test.txt"

pretrained_bert_name = "bert-base-chinese"

save_model_path = "./best_model.bin"

device = "cuda:3" if torch.cuda.is_available() else "cpu"
class_num = len(load_json(label2idx_path))
max_seq_len = 32
batch_size = 128
learning_rate = 2e-5
epochs = 10
weight_decay = 0.01
crf_learning_rate = 2e-2



