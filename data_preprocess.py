from utils import *
from config import *


def preprocess_data():
    """生成label2idx文件"""
    labels = set()
    with open(train_data_path, encoding="utf-8") as f:
        for line in f:
            lis = line.strip().split()
            if len(lis) == 2:
                labels.add(lis[1])
    labels = list(labels)
    labels.sort()
    label2idx = {l: i for i, l in enumerate(labels)}
    print("类别数:{}".format(len(label2idx)))
    dump_json(label2idx, label2idx_path)


if __name__ == '__main__':
    preprocess_data()





