import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch import sigmoid as sig



def dataloader(train_data, symps_dict):
    temp_data = []
    for idx in train_data:
        temp = [0] * len(symps_dict)
        for idy in idx:
            temp[symps_dict[idy]] = 1

        temp_data.append(temp)
        temp = []

    return temp_data


def label_to_vect(train_label, num_label):
    new_train_label = []

    for idx in train_label:
        temp = [0] * num_label
        temp[idx] = 1
        new_train_label.append(temp)

    return new_train_label






saved_path = '.\data\processed_training_data'

file_name_set = ["symp_diagnosis_relation_training", "complaint_training_data", "combine_complaint_symp"]

data_path = os.path.join(saved_path, file_name_set[0])

df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
df_val = pd.read_csv(os.path.join(data_path, 'val.csv'))
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))

df_val_test = pd.read_csv(os.path.join(data_path, 'val_test.csv'))

data_path = '.\data\symp_diagnosis_relation_training.csv'

total_symp = pd.read_csv(data_path)

symp_name_data = total_symp['symptoms']
symps_list = []
for idx in symp_name_data:
    symps_list += idx.split('.')

symps_list = sorted(set(symps_list), key=symps_list.index)

symps_dict = {}

for (idx, idy) in zip(symps_list, range(0, len(symps_list))):
    symps_dict.update({idx: idy})

######################################################################################
labels_5 = {
    'G40': 0,
    'R51': 1,
    'M54': 2,
    'I63': 3

}


train_data = [idx.split('.') for idx in df_train['symptoms']]
training_data = dataloader(train_data, symps_dict)
train_label = [labels_5[df_train.values[idx][0]] for idx in range(0, len(df_train))]
train_label = label_to_vect(train_label, num_label=4)
training_data = torch.tensor(training_data).type(torch.FloatTensor)
train_label = torch.tensor(train_label).type(torch.FloatTensor)


test_data = [idx.split('.') for idx in df_val_test['symptoms']]
testing_data = dataloader(test_data, symps_dict)
test_label = [labels_5[df_val_test.values[idx][0]] for idx in range(0, len(df_val_test))]
test_label = label_to_vect(test_label, num_label=4)

testing_data = torch.tensor(testing_data).type(torch.FloatTensor)
test_label = torch.tensor(test_label).type(torch.FloatTensor)


##########################################################################################


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(len(symps_dict), 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = self.lr(X)
        X = self.sigmoid(X)
        return X


model = LogisticRegression()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)  #采用随机梯度下降的方法
iter_num = []
acc_num = []
loss_num = []

epochs = 10000

for epoch in range(epochs):
    logit = model(training_data)
    loss = criterion(logit, train_label)

    y_pred = logit.ge(0.5).float()
    correct = (y_pred.argmax(dim=1) == train_label.argmax(dim=1)).sum()
    acc = correct.item() / training_data.size(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    model.eval()
    with torch.no_grad():
        val_logit = model(testing_data)
        val_loss = criterion(val_logit, test_label)

        # val_y_pred = logit.ge(0.5).float()
        val_correct = (val_logit.argmax(dim=1) == test_label.argmax(dim=1)).sum()
        val_acc = val_correct.item() / testing_data.size(0)


    if (epoch + 1) % 1 == 0:
        iter_num.append(epoch)
        acc_num.append(val_acc)
        loss_num.append(loss.data.item())
        print('epoch:{}'.format(epoch + 1), ',', 'loss:{:.4f}'.format(loss.data.item()), ',', 'acc:{:.4f}'.format(val_acc))



fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(0, epochs)), loss_num, '-r')

ax2 = ax.twinx()
ax2.plot(list(range(0, epochs)), acc_num)
plt.show()