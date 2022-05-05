import os

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer
from matplotlib import pyplot as plt



# df = pd.read_csv('./data/symp_diagnosis_relation_training.csv')
# df = pd.read_csv('./data/combine_complaint_symp.csv')
# df = pd.read_csv('./data/complaint_training_data.csv')


saved_path = 'C:\Ruibin\Code\BERT_text_classification\data\processed_training_data'

file_name_set = ["symp_diagnosis_relation_training", "complaint_training_data", "combine_complaint_symp"]

data_path = os.path.join(saved_path, file_name_set[2])

df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
df_val = pd.read_csv(os.path.join(data_path, 'val.csv'))
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))



tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


labels_6 = {
    'G40': 0,
    'R51': 1,
    'M54': 2,
    'I63': 3,
    'G43': 4,
    'S06': 5

}

labels_5 = {
    'G40': 0,
    'R51': 1,
    'M54': 2,
    'I63': 3,
    'G43': 4

}

labels_4 = {
    'G40': 0,
    'R51': 1,
    'M54': 2,
    'I63': 3

}

# labels_4 = {
#     'G40': 0,
#     'R51': 1,
#     'I63': 2,
#     'G43': 3
#
# }



# labels = {
#     'R568 - Other and unspecified convulsions': 0,
#     'R51X - Headache': 1,
#     'I639 - Cerebral infarction, unspecified': 2,
#     'G439 - Migraine, unspecified': 3,
#     'R55X - Syncope and collapse': 4,
#     'G35X - Multiple sclerosis': 5,
#     'G409 - Epilepsy, unspecified': 6,
#     'N390 - Urinary tract infection, site not specified': 7,
#     'R298 - Other and unspecified symptoms and signs involving the nervous and musculoskeletal systems': 8,
#     'G403 - Generalized idiopathic epilepsy and epileptic syndromes': 9
#
# }


# labels = {
#     'R568 - Other and unspecified convulsions': 0,
#     'R51X - Headache': 1,
#     'I639 - Cerebral infarction, unspecified': 2,
#     'G439 - Migraine, unspecified': 3
#
# }


# labels = {
#     'R56': 0,
#     'G40': 1,
#     'R51': 2,
#     'M54': 3
#
# }


labels = labels_4

class Dataset(torch.utils.data.Dataset):

    def __init__(self, pre_data):

        self.labels = [labels[label] for label in pre_data['diagnosis']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in pre_data['symptoms']]
        # temp = []
        # for text in pre_data['symptoms']:
        #     temp.append(tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt"))
        #
        # self.text = temp

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        # print(idx)

        return batch_texts, batch_y



# np.random.seed(112)
# df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
#                                      [int(.8*len(df)), int(.9*len(df))])



class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            # train_label = train_label.long()
            train_label = train_label.to(device).long()
            mask = train_input['attention_mask'].to(device).long()
            input_id = train_input['input_ids'].squeeze(1).to(device).long()

            # postion_ids =

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                # val_label = val_label.long()
                val_label = val_label.to(device).long()
                mask = val_input['attention_mask'].to(device).long()
                input_id = val_input['input_ids'].squeeze(1).to(device).long()

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

                # model_to_save = model.module if hasattr(
                #     model, 'module') else model
                # torch.save(model_to_save.state_dict(), "bbc_output.pkl")
                # with open("bbc_output.pkl", 'w') as f:
                #     f.write(model_to_save.config.to_json_string())



        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

    # torch.save(model.state_dict(), 'bert_output.pkl')

    training_loss_list.append(total_loss_train)
    total_acc_val_list.append(total_acc_val)
    return training_loss_list, total_acc_val_list


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            # test_label = test_label.long()
            test_label = test_label.to(device).long()
            mask = test_input['attention_mask'].to(device).long()
            input_id = test_input['input_ids'].squeeze(1).to(device).long()

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')




EPOCHS = 50
model = BertClassifier()
LR = 1e-6
learning_rate = 0.25

training_loss_list = []
total_acc_val_list = []
training_loss_list, total_acc_val_list = train(model, df_train, df_val, LR, EPOCHS)
# train(model, df_train, df_val, learning_rate, EPOCHS)

evaluate(model, df_test)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(0, EPOCHS)), training_loss_list, '-r')

ax2 = ax.twinx()
ax2.plot(list(range(0, EPOCHS)), total_acc_val_list)
plt.show()


# model.load_state_dict(torch.load('bert_output.pkl'))
#
# evaluate(model, df_test)
#
# print('11')








