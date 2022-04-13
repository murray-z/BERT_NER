# _*_ coding:utf-8 _*_
import os

import torch
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from bert_crf_model import BertCrfNer
from utils import *
from config import *
from data_helper import TheseusDataSet
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report


label2idx = load_json(label2idx_path)
idx2label = {i: l for l, i in label2idx.items()}
PAD_IDX = label2idx["O"]

train_dataloader = DataLoader(TheseusDataSet(train_data_path), batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(TheseusDataSet(dev_data_path), batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(TheseusDataSet(test_data_path), batch_size=batch_size, shuffle=False)


def calculate(true_labels, pred_labels):
    true_labels = [true_labels]
    pred_labels = [pred_labels]
    f1 = f1_score(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    return f1, acc, report


def dev(model, data_loader):
    model.eval()
    model.to(device)
    all_pred_tags = []
    all_true_tags = []
    all_loss = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = [d.to(device) for d in batch]
            true_tags = batch[-1]
            loss, pred_tags = model(*batch)
            pred_tags = pred_tags.squeeze(0)

            pred_tags = pred_tags.contiguous().view(-1)
            true_tags = true_tags.view(-1)
            active_loss = batch[1].view(-1) == 1

            flatten_pred_tags = pred_tags[active_loss]
            flatten_true_tags = true_tags[active_loss]

            flatten_pred_tags = flatten_pred_tags.cpu()
            flatten_true_tags = flatten_true_tags.cpu()

            flatten_pred_tags = [idx2label[id.item()] for id in flatten_pred_tags]
            flatten_true_tags = [idx2label[id.item()] for id in flatten_true_tags]

            all_pred_tags.extend(flatten_pred_tags)
            all_true_tags.extend(flatten_true_tags)

            all_loss.append(loss.item())

    loss = sum(all_loss) / len(all_loss)
    f1, acc, report = calculate(all_true_tags, all_pred_tags)

    return f1, acc, report, loss


def train(model, model_save_path):
    # 开始训练
    print("Training ......")
    print(model)

    # if os.path.exists(model_save_path):
    #     model.load_state_dict(torch.load(model_save_path))

    model.to(device)

    # 优化器
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    # 开始训练
    best_f1 = 0.
    for epoch in range(1, epochs+1):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            true_tags = batch[-1]
            loss, pred_tags = model(*batch)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                pred_tags = pred_tags.squeeze(0)
                pred_tags = pred_tags.contiguous().view(-1)
                true_tags = true_tags.view(-1)
                active_loss = batch[1].view(-1) == 1

                flatten_pred_tags = pred_tags[active_loss]
                flatten_true_tags = true_tags[active_loss]

                flatten_pred_tags = flatten_pred_tags.cpu()
                flatten_true_tags = flatten_true_tags.cpu()

                flatten_pred_tags = [idx2label[id.item()] for id in flatten_pred_tags]
                flatten_true_tags = [idx2label[id.item()] for id in flatten_true_tags]

                f1, acc, report = calculate(flatten_true_tags, flatten_pred_tags)

                print("TRAIN STEP:{} F1:{} ACC:{} LOSS:{}".format(i, f1, acc, loss.item()))
        # scheduler.step()
        # 验证
        print("start deving .....")
        f1, acc, report, loss = dev(model, dev_dataloader)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_save_path)
        print("DEV EPOCH:{} F1:{} ACC:{} LOSS:{}".format(epoch, f1, acc, loss))
        print("REPORT:\n{}".format(report))

    # 测试
    print("start testing ....")
    model.load_state_dict(torch.load(model_save_path))
    f1, acc, report, loss = dev(model, test_dataloader)
    print("TEST F1:{} ACC:{} LOSS:{}".format(f1, acc, loss))
    print("REPORT:\n{}".format(report))


if __name__ == '__main__':
    model = BertCrfNer()
    train(model, save_model_path)






