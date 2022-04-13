# _*_ coding:utf-8 _*_

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from bert_softmax_model import BertSoftmaxNer
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


def dev(model, data_loader, criterion):
    model.eval()
    model.to(device)
    all_pred_tags = []
    all_true_tags = []
    all_loss = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = [d.to(device) for d in batch]
            true_tags = batch[-1]
            pred_tags = model(*batch[:3])
            flatten_pred_tags = pred_tags.view(-1, pred_tags.size()[2])
            flatten_true_tags = true_tags.view(-1)

            active_loss = batch[1].view(-1) == 1
            flatten_pred_tags = flatten_pred_tags[active_loss]
            flatten_true_tags = flatten_true_tags[active_loss]

            loss = criterion(flatten_pred_tags, flatten_true_tags)
            all_loss.append(loss.item())

            flatten_pred_tags = flatten_pred_tags.cpu()
            flatten_true_tags = flatten_true_tags.cpu()
            flatten_pred_tags = torch.argmax(flatten_pred_tags, dim=1)

            flatten_pred_tags = [idx2label[id.item()] for id in flatten_pred_tags]
            flatten_true_tags = [idx2label[id.item()] for id in flatten_true_tags]

            all_pred_tags.extend(flatten_pred_tags)
            all_true_tags.extend(flatten_true_tags)

    loss = sum(all_loss) / len(all_loss)
    f1, acc, report = calculate(all_true_tags, all_pred_tags)

    return f1, acc, report, loss


def train(model, model_save_path):
    # 开始训练
    print("Training ......")
    print(model)

    model.to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    criterion = nn.CrossEntropyLoss()


    # 开始训练
    best_f1 = 0.
    for epoch in range(1, epochs+1):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            true_tags = batch[-1]
            pred_tags = model(*batch[:3])
            flatten_pred_tags = pred_tags.view(-1, class_num)
            flatten_true_tags = true_tags.view(-1)

            active_loss = batch[1].view(-1) == 1
            flatten_pred_tags = flatten_pred_tags[active_loss]
            flatten_true_tags = flatten_true_tags[active_loss]

            loss = criterion(flatten_pred_tags, flatten_true_tags)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                flatten_pred_tags = flatten_pred_tags.cpu()
                flatten_true_tags = flatten_true_tags.cpu()
                flatten_pred_tags = torch.argmax(flatten_pred_tags, dim=1)

                # 类别id转成汉字
                flatten_pred_tags = [idx2label[id.item()] for id in flatten_pred_tags]
                flatten_true_tags = [idx2label[id.item()] for id in flatten_true_tags]

                f1, acc, report = calculate(flatten_true_tags, flatten_pred_tags)

                print("TRAIN STEP:{} F1:{} ACC:{} LOSS:{}".format(i, f1, acc, loss.item()))
        scheduler.step()
        # 验证
        f1, acc, report, loss = dev(model, dev_dataloader, criterion)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_save_path)
        print("DEV EPOCH:{} F1:{} ACC:{} LOSS:{}".format(epoch, f1, acc, loss))
        print("REPORT:\n{}".format(report))

    # 测试
    model.load_state_dict(torch.load(model_save_path), strict=False)
    f1, acc, report, loss = dev(model, test_dataloader, criterion)
    print("TEST F1:{} ACC:{} LOSS:{}".format(f1, acc, loss))
    print("REPORT:\n{}".format(report))


if __name__ == '__main__':
    model = BertSoftmaxNer()
    train(model, save_model_path)






