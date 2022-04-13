# _*_ coding:utf-8 _*_


import torch
import torch.nn as nn
from crf import CRF
from transformers import BertModel, BertConfig
from config import class_num


class BertCrfNer(nn.Module):
    def __init__(self, model_name="bert-base-chinese"):
        super(BertCrfNer, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(self.config.hidden_dropout_prob)
        self.hidden_size = self.config.hidden_size
        self.cls_layer = nn.Linear(self.hidden_size, class_num)
        self.crf = CRF(class_num, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # Get the emission scores from the BiLSTM
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        outputs = self.drop(outputs[0])
        logits = self.cls_layer(outputs)
        tags = self.crf.decode(emissions=logits, mask=attention_mask)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            return -1*loss, tags
        return tags


