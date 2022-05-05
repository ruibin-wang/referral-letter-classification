import torch
import pickle
from torch import nn
import numpy as np
import pandas as pd
from transformers import *
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
from tqdm import tqdm
import os
import gensim
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, GlobalMaxPool1D, Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from function_sets import plot_graphs



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



# def get_train_examples(train_file):
#     train_df = pd.read_csv(train_file)
#     ids = train_df['id'].values
#     text = train_df['comment_text'].values
#     labels = train_df[train_df.columns[2:]].values
#     examples = []
#     for i in range(len(train_df)):
#         examples.append(InputExample(ids[i], text[i], labels=labels[i]))
#     return examples




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
    def __init__(self, embed_model, embed_num, embed_dim=300, dropout=0.5, kernel_num=3, kernel_sizes=[2, 3, 4], freeze_embeddings=True, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)

        if embed_model is not None:
            self.vocab_size, self.embed_dim = embed_model.vectors.shape

            # self.embedding = nn.Embedding.from_pretrained(torch.tensor(embed_model.vectors),
            #                                               freeze=freeze_embeddings)
            self.pretrained_weights = torch.FloatTensor(embed_model.vectors)
            self.embedding = nn.Embedding.from_pretrained(self.pretrained_weights,
                                                                      freeze=freeze_embeddings)


        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.embed_dim)) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.num_labels)

    def forward(self, inputs, labels=None):
        inputs = self.embedding(inputs)
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


def generate_dataloader(data_path, batch_size, seq_len, tokenizer, labels):
    train_examples = get_train_examples(data_path, labels)
    train_features = get_features_from_examples(train_examples, seq_len, tokenizer)
    train_dataset = get_dataset_from_features(train_features)


    if data_path.split('\\')[-1] is 'train.csv':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    return train_dataloader


def evaluate(model, embed_lookup, tokenizer, seq_len, test_dataloader):
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
            test_bert_output = basemodel(test_input_ids, test_input_mask)


        test_label = label_list_to_single_label(test_label_ids)
        cnn_test_input_ids = format_batch_cnn(embed_lookup, tokenizer, test_input_ids, seq_len)
        cnn_test_logits = model(cnn_test_input_ids, test_label_ids)

        logits = (test_bert_output + cnn_test_logits[1]) / 2

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


def tokenize_all_reviews(embed_lookup, reviews_split):
    # split each review into a list of words
    reviews_words = [review.split() for review in reviews_split]

    tokenized_reviews = []
    for review in reviews_words:
        ints = []
        for word in review:
            try:
                idx = embed_lookup.key_to_index[word]
            except:
                idx = 0
            ints.append(idx)
        tokenized_reviews.append(ints)

    return tokenized_reviews



def format_batch_cnn(embed_lookup, tokenizer, input_ids, seq_len):
    batch_input_text = [tokenizer.decode(idx) for idx in input_ids]
    batch_input_text = [idx.strip('[CLS]').strip(' [PAD] ').strip('[SE').rstrip() for idx in batch_input_text]
    tokenized_symptoms = tokenize_all_reviews(embed_lookup, batch_input_text)

    new_tokenized_symptoms = []
    for idx in tokenized_symptoms:
        if len(idx) < seq_len:
            idx = idx + [0] * (seq_len - len(idx))  ## [0,0,0,0... 1,123,1215,15]
            # idx = [0] * (seq_len - len(idx)) + idx   ## [1,123,1215,15, 0,0,0,0... ]
            new_tokenized_symptoms.append(idx)

    # cnn_input_ids = []
    # for idx in new_tokenized_symptoms:
    #     cnn_input_ids.append(embed_lookup[idx])

    cnn_input_ids = torch.LongTensor(np.array(new_tokenized_symptoms)).cuda()

    return cnn_input_ids


labels_5 = {
    'G40': 0,
    'R51': 1,
    'I63': 2,
    'G43': 3

}
labels = labels_5


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




embed_lookup = gensim.models.KeyedVectors.load_word2vec_format('./fast_embed/saved_model_gensim'+".bin", binary=True)


device = torch.device(type='cuda')
pretrained_weights = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


basemodel = BertClassifier()


labels = ['G40', 'R51', 'I63', 'G43']
labels_5 = {
    'G40': 0,
    'R51': 1,
    'I63': 2,
    'G43': 3

}
labels = labels_5


num_labels = len(labels)
seq_len = 512

# train_file = 'C:/Users/Ruibin/Desktop/train.csv'
# train_examples = get_train_examples(train_file)




#############################################################################################

# saved_path = 'C:\Ruibin\Code\BERT_text_classification\data\processed_training_data'
#
# file_name_set = ["symp_diagnosis_relation_training", "complaint_training_data", "combine_complaint_symp"]
#
# data_path = os.path.join(saved_path, file_name_set[0])
#
# df_train_path = os.path.join(data_path, 'train.csv')
# df_val_path = os.path.join(data_path, 'val.csv')
# df_test_path = os.path.join(data_path, 'test.csv')
#
# batch_size = 8
#
#
#
# train_dataloader = generate_dataloader(df_train_path, batch_size, seq_len, tokenizer, labels_5)
# val_dataloader = generate_dataloader(df_val_path, batch_size, seq_len, tokenizer, labels_5)
# test_dataloader = generate_dataloader(df_test_path, batch_size, seq_len, tokenizer, labels_5)


#############################################################################################


# saved_path = 'C:\Ruibin\Code\BERT_text_classification\data\processed_training_data'
#
# file_name_set = ["symp_diagnosis_relation_training", "complaint_training_data", "combine_complaint_symp"]
#
# data_path = os.path.join(saved_path, file_name_set[0])
#
# df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
# df_val = pd.read_csv(os.path.join(data_path, 'val.csv'))
# df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))
#
# batch_size = 8
#
# train, val = Dataset(df_train), Dataset(df_val)
#
# train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)





#############################################################################################

embed_num = seq_len
cnn_embed_num = 100
embed_dim = 768
cnn_embed_dim = 300
dropout = 0.5
# dropout = 0.3
alpha = 0.6

kernel_sizes = [2,3,4]
kernel_num = len(kernel_sizes)



model = KimCNN(embed_lookup, cnn_embed_num, cnn_embed_dim, dropout=dropout, kernel_num=kernel_num, kernel_sizes=kernel_sizes, num_labels=num_labels)
model.to(device)

# lr = 3e-5
lr = 1e-6
epochs = 200
bert_optimizer = torch.optim.Adam(basemodel.parameters(), lr=lr)
cnn_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss().cuda()
basemodel = basemodel.to(device)

for i in range(epochs):
    print('-----------EPOCH #{}-----------'.format(i + 1))
    # print('training...')

    total_acc_train = 0
    total_loss_train = 0

    model.train()

    # for batch in tqdm(train_dataloader):
    #     batch = tuple(t.to(device) for t in batch)
    #     input_ids, input_mask, segment_ids, label_ids = batch

    for train_input, train_label in tqdm(train_dataloader):
        # train_label = train_label.long()
        train_label = train_label.to(device).long()
        mask = train_input['attention_mask'].to(device).long()
        input_id = train_input['input_ids'].squeeze(1).to(device).long()


        # bert_output = basemodel(input_ids, input_mask)
        # train_label = label_list_to_single_label(label_ids)
        # bert_loss = criterion(bert_output, train_label)

        # cnn_input_ids = format_batch_cnn(embed_lookup, tokenizer, input_ids, seq_len)
        # cnn_loss, cnn_logits = model(cnn_input_ids, label_ids)
        #
        # loss = alpha*cnn_loss + (1-alpha)*bert_loss


        output = basemodel(input_id, mask)
        batch_loss = criterion(output, train_label)

        loss = batch_loss

        total_loss_train += loss.item()

        # logits = (bert_output + cnn_logits)/2
        logits = output



        acc = (logits.argmax(dim=1) == train_label).sum().item()
        total_acc_train += acc

        # loss.requires_grad = True
        basemodel.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # optimizer.zero_grad()

    y_true = []
    y_pred = []
    total_acc_val = 0
    total_loss_val = 0

    model.eval()
    # print('evaluating...')
    # with torch.no_grad():
    #
    #     for step, batch in enumerate(val_dataloader):
    #         batch = tuple(t.to(device) for t in batch)
    #         val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch
    #         val_label = label_list_to_single_label(val_label_ids)
    #
    #         cnn_val_input_ids = format_batch_cnn(embed_lookup, tokenizer, val_input_ids, seq_len)
    #
    #
    #         # val_bert_output = basemodel(val_input_ids, val_input_mask)
    #
    #         val_cnn_logits = model(cnn_val_input_ids)
    #
    #         # val_bert_loss = criterion(val_bert_output, val_label)
    #         # val_logits = val_bert_output
    #
    #         val_logits = val_cnn_logits
    #
    #         # logits = (bert_output + cnn_logits[1]) / 2
    #
    #         # val_loss = val_bert_loss
    #         val_loss = criterion(val_logits, val_label)
    #         total_loss_val += val_loss.item()
    #
    #         acc = (val_logits.argmax(dim=1) == val_label).sum().item()
    #         total_acc_val += acc
    #
    #     print(
    #         f'Epochs: {i + 1} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .3f} \
    #                     | Train Accuracy: {total_acc_train / len(train_dataloader.dataset): .3f} \
    #                     | Val Loss: {total_loss_val / len(val_dataloader.dataset): .3f} \
    #                     | Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
    #
    #     # torch.save(model.state_dict(), 'bert_output.pkl')


    with torch.no_grad():

        for val_input, val_label in val_dataloader:
            val_label = val_label.to(device).long()
            mask = val_input['attention_mask'].to(device).long()
            input_id = val_input['input_ids'].squeeze(1).to(device).long()

            output = basemodel(input_id, mask)

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
        f'Epochs: {i + 1} | Train Loss: {total_loss_train / len(df_train): .3f} \
            | Train Accuracy: {total_acc_train / len(df_train): .3f} \
            | Val Loss: {total_loss_val / len(df_val): .3f} \
            | Val Accuracy: {total_acc_val / len(df_val): .3f}')

# torch.save(model.state_dict(), 'bert_output.pkl')



# model.load_state_dict(torch.load('bert_output.pkl'))
# # evaluate(model, test_dataloader)
# evaluate(model, embed_lookup, tokenizer, seq_len, test_dataloader)




