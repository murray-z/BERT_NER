# _*_ coding:utf-8 _*_


import torch
import torch.nn as nn
import transformers
from transformers import BertModel, BertConfig
from config import class_num, pretrained_bert_name


class BertSoftmaxNer(nn.Module):
    def __init__(self):
        super(BertSoftmaxNer, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_bert_name)
        self.bert = BertModel.from_pretrained(pretrained_bert_name)
        self.drop = nn.Dropout(self.config.hidden_dropout_prob)
        self.hidden_size = self.config.hidden_size
        self.cls_layer = nn.Linear(self.hidden_size, class_num)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs = self.drop(outputs[0])
        logits = self.cls_layer(outputs)
        return logits




