import torch
from torch import nn
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt



class InputExample(object):
    def __init__(self, id, text, labels=None):
        self.id = id
        self.text = text
        self.labels = labels

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def get_train_examples(train_file, labels_5):
    train_df = pd.read_csv(train_file)
    # ids = train_df['diagnosis'].values
    # ids = range(0, len(train_df['symptoms']))
    ids = [idx for idx in range(0,len(train_df['symptoms']))]
    text = train_df['symptoms'].values
    # labels = train_df[train_df.columns[2:]].values
    test_label = train_df['diagnosis'].values

    labels = []
    for label in test_label:
        temp_matrix = [0] * 5
        temp_matrix[labels_5[label]] = 1
        labels.append(temp_matrix)
    labels = np.array(labels)

    examples = []
    for i in range(len(train_df)):
        examples.append(InputExample(ids[i], text[i], labels=labels[i]))
    return examples




def get_features_from_examples(examples, max_seq_len, tokenizer):
    features = []
    for i,example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:(max_seq_len - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(tokens)
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        label_ids = [float(label) for label in example.labels]
        # label_ids = [float(label_num[example.labels])]
        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_ids=label_ids))
    return features


def get_dataset_from_features(features):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.float)
    dataset = TensorDataset(input_ids,
                            input_mask,
                            segment_ids,
                            label_ids)
    return dataset


def label_list_to_single_label(label_ids):
    train_label = []
    for label in label_ids:
        for idx in range(0, len(label)):
            if label[idx] == 1:
                train_label.append(idx)

    train_label = torch.tensor(train_label).cuda()
    # train_label = train_label.clone().detach()

    return train_label



class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        # self.relu = nn.ReLU()
        self.relu = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)


        return final_layer


def generate_dataloader(data_path, batch_size, seq_len, tokenizer, labels):
    train_examples = get_train_examples(data_path, labels)
    train_features = get_features_from_examples(train_examples, seq_len, tokenizer)
    train_dataset = get_dataset_from_features(train_features)


    if data_path.split('\\')[-1] is 'train.csv':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    return train_dataloader


def evaluate(bert_model, test_dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        bert_model = bert_model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            test_input_ids, test_input_mask, test_segment_ids, test_label_ids = batch
            test_label = label_list_to_single_label(test_label_ids)

            test_bert_output = bert_model(test_input_ids, test_input_mask)


            test_logits = test_bert_output


            acc = (test_logits.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc


        print(f'Test Accuracy: {total_acc_test / len(test_dataloader.dataset): .3f}')



def generate_dataloader(data_path, batch_size, seq_len, tokenizer, labels):
    train_examples = get_train_examples(data_path, labels)
    train_features = get_features_from_examples(train_examples, seq_len, tokenizer)
    train_dataset = get_dataset_from_features(train_features)


    if data_path.split('\\')[-1] is 'train.csv':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    return train_dataloader



device = torch.device(type='cuda')
pretrained_weights = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


basemodel = BertClassifier()


labels = ['G40', 'R51', 'I63', 'G43']
labels_5 = {
    'G40': 0,
    'R51': 1,
    'M54': 2,
    'I63': 3

}


num_labels = len(labels)
seq_len = 512


#############################################################################################

saved_path = '.\data\processed_training_data'

file_name_set = ["symp_diagnosis_relation_training", "complaint_training_data", "combine_complaint_symp"]

data_path = os.path.join(saved_path, file_name_set[2])

df_train_path = os.path.join(data_path, 'train.csv')
df_val_path = os.path.join(data_path, 'val.csv')
df_test_path = os.path.join(data_path, 'test.csv')



df_val = pd.read_csv(df_val_path)
df_test = pd.read_csv(df_test_path)

val_test = pd.concat([df_val, df_test])
val_test.to_csv(os.path.join(data_path, 'val_test.csv'), index=False, encoding="utf-8")



df_val_path = os.path.join(data_path, 'val_test.csv')
df_test_path = os.path.join(data_path, 'val_test.csv')


batch_size = 4


train_dataloader = generate_dataloader(df_train_path, batch_size, seq_len, tokenizer, labels_5)
val_dataloader = generate_dataloader(df_val_path, batch_size, seq_len, tokenizer, labels_5)
test_dataloader = generate_dataloader(df_test_path, batch_size, seq_len, tokenizer, labels_5)


#############################################################################################


embed_num = seq_len
cnn_embed_num = 300
embed_dim = 768
cnn_embed_dim = 300
dropout = 0.5
# dropout = 0.3
alpha = 0.3
alpha_lr = 1e-5

kernel_sizes = [2,3,4]
kernel_num = len(kernel_sizes)


##############################################################################################


# lr = 3e-5
lr = 1e-6
epochs = 100
bert_optimizer = torch.optim.Adam(basemodel.parameters(), lr=lr)


criterion = nn.CrossEntropyLoss().cuda()
basemodel = basemodel.to(device)
training_loss_list = []
total_acc_val_list = []


for i in range(epochs):
    print('-----------EPOCH #{}-----------'.format(i + 1))
    # print('training...')

    total_acc_train = 0
    total_loss_train = 0


    basemodel.train()

    for batch in tqdm(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch


        bert_output = basemodel(input_ids, input_mask)
        train_label = label_list_to_single_label(label_ids)
        bert_loss = criterion(bert_output, train_label)


        loss = bert_loss

        total_loss_train += loss.item()

        logits = bert_output

        acc = (logits.argmax(dim=1) == train_label).sum().item()
        total_acc_train += acc

        basemodel.zero_grad()

        loss.backward()
        bert_optimizer.step()


    y_true = []
    y_pred = []
    total_acc_val = 0
    total_loss_val = 0


    basemodel.eval()
    print('evaluating...')
    with torch.no_grad():

        for step, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch
            val_label = label_list_to_single_label(val_label_ids)


            val_bert_output = basemodel(val_input_ids, val_input_mask)

            val_bert_loss = criterion(val_bert_output, val_label)

            # val_logits = val_bert_output

            val_logits = val_bert_output


            # val_loss = val_bert_loss
            val_loss = criterion(val_logits, val_label)
            total_loss_val += val_loss.item()

            acc = (val_logits.argmax(dim=1) == val_label).sum().item()
            total_acc_val += acc

        print(
            f'Epochs: {i + 1} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_dataloader.dataset): .3f} \
                        | Val Loss: {total_loss_val / len(val_dataloader.dataset): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')

    training_loss_list.append(total_loss_train)
    total_acc_val_list.append(total_acc_val)

torch.save(basemodel.state_dict(), 'bert_output.pkl')
evaluate(basemodel, test_dataloader)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(0, epochs)), training_loss_list, '-r')

ax2 = ax.twinx()
ax2.plot(list(range(0, epochs)), total_acc_val_list)
plt.show()

# torch.save(basemodel.state_dict(), 'bert_output.pkl')

# model = BertClassifier()
# model.load_state_dict(torch.load('bert_output.pkl'))
# evaluate(model, test_dataloader)






