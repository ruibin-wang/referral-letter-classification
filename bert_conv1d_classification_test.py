import torch
import pickle
from torch import nn
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
from tqdm import tqdm
import os
import torch.nn.functional as F



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



class KimCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, dropout=0.2, kernel_num=3, kernel_sizes=[2, 3, 4], num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.embed_dim)) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.num_labels)

    def forward(self, inputs, labels=None):
        output = inputs.unsqueeze(1)
        output = [nn.functional.relu(conv(output)).squeeze(3) for conv in self.convs]
        output = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        output = self.dropout(output)
        logits = self.classifier(output)

        if labels is not None:
            labels = label_list_to_single_label(labels)


        # if labels is not None:
            # loss_fct = nn.BCEWithLogitsLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            return loss, logits
        else:
            return logits


def generate_dataloader(data_path, batch_size, seq_len, tokenizer, labels):
    train_examples = get_train_examples(data_path, labels)
    train_features = get_features_from_examples(train_examples, seq_len, tokenizer)
    train_dataset = get_dataset_from_features(train_features)


    if data_path.split('\\')[-1] is 'train.csv':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    return train_dataloader


def evaluate(model, test_dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        test_input_ids, test_input_mask, test_segment_ids, test_label_ids = batch
        with torch.no_grad():
            # val_inputs, _ = basemodel(val_input_ids, val_segment_ids, val_input_mask)
            test_bert_output = basemodel(test_input_ids, test_segment_ids, test_input_mask)
            logits = model(test_bert_output.last_hidden_state)

        test_label = label_list_to_single_label(test_label_ids)

        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits, test_label)
        # total_loss_val += loss.item()

        acc = (logits.argmax(dim=1) == test_label).sum().item()
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


class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_num=768,
                 embed_dim=768,
                 filter_sizes=[2,3,4],
                 num_filters=[2,2,2],
                 num_classes=4,
                 dropout=0.5):


        super(CNN_NLP, self).__init__()

        self.embed_num = embed_num
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.embed_num, self.embed_dim)

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        # x_embed = self.embedding(input_ids).float()

        x_embed = input_ids
        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits



device = torch.device(type='cuda')
pretrained_weights = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
basemodel = BertModel.from_pretrained(pretrained_weights)
basemodel.to(device)



# labels = ['G40', 'R51', 'M54', 'I63', 'G43']
# labels_5 = {
#     'G40': 0,
#     'R51': 1,
#     'M54': 2,
#     'I63': 3,
#     'G43': 4
#
# }


labels = ['G40', 'R51', 'I63', 'G43']
# labels_5 = {
#     'G40': 0,
#     'R51': 1,
#     'I63': 2,
#     'G43': 3
#
# }


labels_5 = {
    'G40': 0,
    'R51': 1,
    'M54': 2,
    'I63': 3

}


num_labels = len(labels)
seq_len = 512



#############################################################################################

saved_path = 'C:\Ruibin\Code\BERT_text_classification\data\processed_training_data'

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
embed_dim = basemodel.config.hidden_size
dropout = basemodel.config.hidden_dropout_prob
# dropout = 0.3

kernel_sizes = [2,3,4]
kernel_num = len(kernel_sizes)



# model = KimCNN(embed_num, embed_dim, dropout=dropout, kernel_num=kernel_num, kernel_sizes=kernel_sizes, num_labels=num_labels)

model = CNN_NLP(embed_num, embed_dim, filter_sizes=kernel_sizes, num_filters=[2,2,2], num_classes=4, dropout=0.5)
model.to(device)


lr = 3e-5
epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.25, rho=0.95)

loss_fct = nn.CrossEntropyLoss()


training_loss_list = []
total_acc_val_list = []

for i in range(epochs):
    print('-----------EPOCH #{}-----------'.format(i + 1))
    # print('training...')

    total_acc_train = 0
    total_loss_train = 0

    model.train()
    # for step, batch in tqdm(enumerate(train_dataloader)):
    for batch in tqdm(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            # inputs, _ = basemodel(input_ids, segment_ids, input_mask)
            bert_output = basemodel(input_ids, segment_ids, input_mask)

        cnn_input = bert_output.last_hidden_state

        # cnn_input = input_ids.clone().to(torch.int64)

        logits = model(cnn_input)

        labels = label_list_to_single_label(label_ids)
        loss = loss_fct(logits, labels)

        total_loss_train += loss.item()
        # loss = loss.mean()

        train_label = label_list_to_single_label(label_ids)

        acc = (logits.argmax(dim=1) == train_label).sum().item()
        total_acc_train += acc

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    y_true = []
    y_pred = []
    total_acc_val = 0
    total_loss_val = 0

    model.eval()
    # print('evaluating...')
    for step, batch in enumerate(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch

        with torch.no_grad():
            # val_inputs, _ = basemodel(val_input_ids, val_segment_ids, val_input_mask)
            val_bert_output = basemodel(val_input_ids, val_segment_ids, val_input_mask)

            cnn_val_input = val_bert_output.last_hidden_state


            logits = model(cnn_val_input)

        val_label = label_list_to_single_label(val_label_ids)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, val_label)
        total_loss_val += loss.item()

        acc = (logits.argmax(dim=1) == val_label).sum().item()
        total_acc_val += acc

    print(
        f'Epochs: {i + 1} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .3f} \
                    | Train Accuracy: {total_acc_train / len(train_dataloader.dataset): .3f} \
                    | Val Loss: {total_loss_val / len(val_dataloader.dataset): .3f} \
                    | Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')

    training_loss_list.append(total_loss_train)
    total_acc_val_list.append(total_acc_val)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(0, epochs)), training_loss_list, '-r')

ax2 = ax.twinx()
ax2.plot(list(range(0, epochs)), total_acc_val_list)
plt.show()

# torch.save(model.state_dict(), 'bert_output.pkl')
#
#
#
#
# model.load_state_dict(torch.load('bert_output.pkl'))
# evaluate(model, test_dataloader)




