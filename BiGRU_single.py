import torch
import pickle
from torch import nn
import numpy as np
from transformers import BertModel, BertTokenizer
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
from tqdm import tqdm
import os
import gensim
import torch.nn.functional as F
import pandas as pd
from nltk.tokenize import word_tokenize
from function_sets import tokenize, set_seed, encode
import torch.optim as optim
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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional,
                 dropout_rate):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=bidirectional,
                           dropout=dropout_rate)

    def forward(self, x, length):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, length)
        packed_output, (hidden, cell) = self.rnn(packed_x)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        return hidden, output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional,
                 dropout_rate):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional,
                          dropout=dropout_rate)

    def forward(self, x, length):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, length)
        packed_output, hidden = self.rnn(packed_x)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        return hidden, output


class TextRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, bidirectional, out_dim,
                 dropout_rate, pretrained_embed, use_gru=False, freeze=True,
                 random_embed=False, vocab_size=None):
        super(TextRNN, self).__init__()
        if random_embed:
            self.embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=True)
        if use_gru:
            self.rnn = GRU(embed_size, hidden_size, num_layers, bidirectional,
                           dropout_rate)
        else:
            self.rnn = LSTM(embed_size, hidden_size, num_layers, bidirectional,
                            dropout_rate)
        self.proj = nn.Linear(2 * hidden_size, out_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        # text, text_length = x  # text: [seq_len, bs]
        # text = text.permute(1, 0)  # text: [bs, seq_len]
        # embed_x = self.embed(text)  # embed_x: [bs, seq_len, embed_dim]
        # embed_x = embed_x.permute(1, 0, 2)  # embed_x: [seq_len, bs, embed_dim]
        # hidden, _ = self.rnn(embed_x, text_length)  # hidden: [2*num_layers, bs, hidden_size]
        # hidden = torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1)

        text_length = [len(idx) for idx in x]
        # text_length = torch.tensor([len(idx) for idx in x])
        text = x.permute(1, 0)  # text: [bs, seq_len]
        embed_x = self.embed(text).double()  # embed_x: [bs, seq_len, embed_dim]
        embed_x = embed_x.permute(1, 0, 2)  # embed_x: [seq_len, bs, embed_dim]
        hidden, _ = self.rnn(embed_x, text_length)  # hidden: [2*num_layers, bs, hidden_size]
        hidden = torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1)

        return self.proj(self.dropout(hidden))


class RateGRU(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size):
        super(RateGRU, self).__init__()
        self.batch = batch_sz
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.output_size = output_size

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units)
        self.fc = nn.Linear(self.hidden_units, self.output_size)
        self.relu = nn.Sigmoid()

    def initialize_hidden_state(self, len_batch, device):
        return torch.zeros((1, len_batch, self.hidden_units)).to(device)

    def forward(self, x, device):
        input = self.embedding(x)
        input = input.permute(1, 0, 2)
        self.hidden = self.initialize_hidden_state(len(x), device)
        output, self.hidden = self.gru(input, self.hidden)
        out = output[-1, :, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.relu(out)

        return out



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


def evaluate(cnn_model, tokenizer, test_dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        cnn_model = cnn_model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            test_input_ids, test_input_mask, test_segment_ids, test_label_ids = batch
            test_label = label_list_to_single_label(test_label_ids)

            cnn_test_input_ids = format_embedding_to_batch_cnn(word2idx, tokenizer, test_input_ids)


            test_cnn_logits = cnn_model(cnn_test_input_ids, device)

            test_logits = test_cnn_logits


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


def format_embedding_to_batch_cnn(word2idx, tokenizer, input_ids):

    list_input_ids = input_ids.data.cpu().numpy().tolist()

    trans_input_ids = []
    for idx in list_input_ids:
        temp = []
        for idy in idx:
            if idy is not 0:
                temp.append(idy)

        trans_input_ids.append(temp)

    batch_input_text = [tokenizer.decode(torch.tensor(idx)) for idx in trans_input_ids]

    batch_input_text = [idx.strip('[CLS]') for idx in batch_input_text]
    # batch_input_text = [idx.strip('[CLS]').strip(' [PAD] ').rstrip() for idx in batch_input_text]
    batch_input_text = [idx+']' for idx in batch_input_text]


    batch_input_text = [word_tokenize(idx) for idx in batch_input_text]
    cnn_input_ids = encode(batch_input_text, word2idx, max_len)

    for idx in range(0, len(cnn_input_ids)):
        for idy in cnn_input_ids[idx]:
            if idy is None:
                print(batch_input_text[idx])


    cnn_input_ids = torch.LongTensor(cnn_input_ids).cuda()

    return cnn_input_ids


def load_pretrained_vectors(word2idx, fname):
    """Load pretrained vectors and create embedding layers.

    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings


embed_lookup = gensim.models.KeyedVectors.load_word2vec_format('./fast_embed/saved_model_gensim'+".bin", binary=True)



device = torch.device(type='cuda')
pretrained_weights = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


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

text_path = '.\data\\' + file_name_set[2] + '.csv'
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


batch_size = 8


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
cnn_learning_rate = 0.2

num_layers = 2
bidirectional = True
out_dim = num_labels

freeze = True
use_gru = True
random_embed = False
units = 1024


##############################################################################################

text = pd.read_csv(text_path)
text = np.array(text.symptoms)

tokenized_texts, word2idx, max_len = tokenize(tokenizer, text)
max_len = max_len + 3  ## we add 3 characters '[', 'SEP', ']'
embeddings = load_pretrained_vectors(word2idx, "./fast_embed//crawl-300d-2M.vec")
embeddings = torch.tensor(embeddings)
vocab_size = len(word2idx)
##############################################################################################
# cnn_model = KimCNN(embed_lookup, cnn_embed_num, cnn_embed_dim, dropout=dropout, kernel_num=kernel_num, kernel_sizes=kernel_sizes, num_labels=num_labels)

cnn_model = RateGRU(vocab_size, cnn_embed_dim, units, batch_size, out_dim)
cnn_model.to(device)

# lr = 3e-5
lr = 1e-6
epochs = 200

# cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
cnn_optimizer = optim.Adadelta(cnn_model.parameters(), lr=cnn_learning_rate, rho=0.95)

criterion = nn.CrossEntropyLoss().cuda()

training_loss_list = []
total_acc_val_list = []


for i in range(epochs):
    print('-----------EPOCH #{}-----------'.format(i + 1))
    # print('training...')

    total_acc_train = 0
    total_loss_train = 0

    cnn_model.train()

    for batch in tqdm(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch


        train_label = label_list_to_single_label(label_ids)


        cnn_input_ids = format_embedding_to_batch_cnn(word2idx, tokenizer, input_ids)


        cnn_logits = cnn_model(cnn_input_ids, device)
        cnn_loss = criterion(cnn_logits, train_label)


        loss = cnn_loss

        total_loss_train += loss.item()

        # logits = bert_output
        logits = cnn_logits

        acc = (logits.argmax(dim=1) == train_label).sum().item()
        total_acc_train += acc

        cnn_model.zero_grad()
        loss.backward()
        cnn_optimizer.step()


    y_true = []
    y_pred = []
    total_acc_val = 0
    total_loss_val = 0


    cnn_model.eval()

    print('evaluating...')
    with torch.no_grad():

        for step, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch
            val_label = label_list_to_single_label(val_label_ids)

            cnn_val_input_ids = format_embedding_to_batch_cnn(word2idx, tokenizer, val_input_ids)

            val_cnn_logits = cnn_model(cnn_val_input_ids, device)

            val_cnn_loss = criterion(val_cnn_logits, val_label)

            val_logits = val_cnn_logits

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


evaluate(cnn_model, tokenizer, test_dataloader)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(0, epochs)), training_loss_list, '-r')

ax2 = ax.twinx()
ax2.plot(list(range(0, epochs)), total_acc_val_list)
plt.show()





