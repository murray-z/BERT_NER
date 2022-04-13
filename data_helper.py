# _*_ coding:utf-8 _*_

from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils import *
from config import *


class TheseusDataSet(Dataset):
    def __init__(self, data_path):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.labl2idx = load_json(label2idx_path)
        self.PAD_IDX = self.labl2idx["O"]
        sents = []
        labels = []
        with open(data_path, encoding="utf-8") as f:
            texts = f.read().split("\n\n")
            for text in texts:
                sent = []
                label = []
                for line in text.split("\n"):
                    lis = line.strip().split()
                    if len(lis) == 2:
                        sent.append(lis[0])
                        label.append(lis[1])
                assert len(sent) == len(label)
                sents.append(sent)
                labels.append(label)

        self.input_ids, self.token_type_ids, self.attention_mask, self.tag_ids = self.encode(sents, labels)

        print("=====data_set:{}====".format(data_path))
        print("input_ids size: ", self.input_ids.size())
        print("token_type_ids size: ", self.token_type_ids.size())
        print("attention_mask size: ", self.attention_mask.size())
        print("tag_ids size: ", self.tag_ids.size())
        print("\n")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx],\
               self.attention_mask[idx], self.tag_ids[idx]

    def encode(self, sents, tags):
        input_ids, token_type_ids, attention_mask = [], [], []
        for text in sents:
            res = self.tokenizer.encode_plus(" ".join(text),
                                             add_special_tokens=True,
                                             max_length=max_seq_len,
                                             pad_to_max_length=True,
                                             truncation=True)
            input_ids.append(res['input_ids'])
            token_type_ids.append(res['token_type_ids'])
            attention_mask.append(res['attention_mask'])

        tag_ids = []
        for tag in tags:
            if len(tag) <= max_seq_len - 2:
                tmp_id = [self.PAD_IDX] + [self.labl2idx[t] for t in tag] + \
                         [self.PAD_IDX] + [self.PAD_IDX] * (max_seq_len - 2 - len(tag))
            else:
                tmp_id = [self.PAD_IDX] + [self.labl2idx[t] for t in tag[:(max_seq_len - 2)]] + [self.PAD_IDX]
            tag_ids.append(tmp_id)

        return torch.tensor(input_ids), torch.tensor(attention_mask), \
               torch.tensor(token_type_ids), torch.tensor(tag_ids)


if __name__ == '__main__':
    # data_set = TheseusDataSet("./data/dev.txt")
    # print(data_set.input_ids[:5])
    # print(data_set.token_type_ids[:5])
    # print(data_set.attention_mask[:5])
    # print(data_set.tag_ids[:5])

    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
    res = tokenizer.encode_plus("h e l l o 你 好", add_special_tokens=True,
                                             max_length=max_seq_len,
                                             pad_to_max_length=True,
                                             truncation=True)

    print(res)

    print(tokenizer.decode(res["input_ids"]))
