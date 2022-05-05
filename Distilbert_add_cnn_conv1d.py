import torch
import pickle
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
from tqdm import tqdm
import os
import gensim
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from nltk.tokenize import word_tokenize
from function_sets import tokenize, set_seed, encode
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel




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
        temp_matrix = [0] * len(labels_5)
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, labels=None):
        inputs = self.embedding(inputs)
        output = inputs.unsqueeze(1)
        output = [nn.functional.relu(conv(output)).squeeze(3) for conv in self.convs]
        output = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        output = self.dropout(output)
        logits = self.classifier(output)
        logits = self.sigmoid(logits)

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

class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=4,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
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
        x_embed = self.embedding(input_ids).float()

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



class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output


def generate_dataloader(data_path, batch_size, seq_len, tokenizer, labels):
    train_examples = get_train_examples(data_path, labels)
    train_features = get_features_from_examples(train_examples, seq_len, tokenizer)
    train_dataset = get_dataset_from_features(train_features)


    if data_path.split('\\')[-1] is 'train.csv':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    return train_dataloader


def evaluate(bert_model, cnn_model, embed_lookup, tokenizer, seq_len, alpha, test_dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        bert_model = bert_model.cuda()
        cnn_model = cnn_model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            test_input_ids, test_input_mask, test_segment_ids, test_label_ids = batch
            test_label = label_list_to_single_label(test_label_ids)

            cnn_test_input_ids = format_embedding_to_batch_cnn(word2idx, tokenizer, test_input_ids)

            test_bert_output = bert_model(test_input_ids, test_input_mask)
            test_cnn_logits = cnn_model(cnn_test_input_ids)

            test_logits = (1 - alpha) * test_bert_output + alpha * test_cnn_logits


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



# embeddings = load_pretrained_vectors(word2idx, "fastText/crawl-300d-2M.vec")
# embeddings = torch.tensor(embeddings)


device = torch.device(type='cuda')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)


basemodel = DistilBERTClass()


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

saved_path = 'C:\Ruibin\Code\BERT_text_classification\data\processed_training_data'

file_name_set = ["symp_diagnosis_relation_training", "complaint_training_data", "combine_complaint_symp"]

text_path = 'C:\Ruibin\Code\BERT_text_classification\data\\' + file_name_set[0] + '.csv'
data_path = os.path.join(saved_path, file_name_set[0])

df_train_path = os.path.join(data_path, 'train.csv')
df_val_path = os.path.join(data_path, 'val.csv')
df_test_path = os.path.join(data_path, 'test.csv')


df_val = pd.read_csv(df_val_path)
df_test = pd.read_csv(df_test_path)

val_test = pd.concat([df_val, df_test])
val_test.to_csv(os.path.join(data_path, 'val_test.csv'), index=False, encoding="utf-8")

df_val_path = os.path.join(data_path, 'val_test.csv')
df_test_path = os.path.join(data_path, 'val_test.csv')


batch_size = 2


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
cnn_learning_rate = 0.25


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

cnn_model = CNN_NLP(pretrained_embedding=embeddings, freeze_embedding=False, vocab_size=vocab_size, embed_dim=cnn_embed_dim,
                        filter_sizes=kernel_sizes, num_filters=[2,2,2], num_classes=4, dropout=0.5)
cnn_model.to(device)

# lr = 3e-5
lr = 1e-6
epochs = 300


bert_optimizer = torch.optim.Adam(basemodel.parameters(), lr=lr)
# cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
cnn_optimizer = optim.Adadelta(cnn_model.parameters(), lr=cnn_learning_rate, rho=0.95)


# criterion = nn.CrossEntropyLoss().cuda()
criterion = nn.BCEWithLogitsLoss().cuda()
basemodel = basemodel.to(device)

training_loss_list = []
total_acc_val_list = []

for i in range(epochs):
    print('-----------EPOCH #{}-----------'.format(i + 1))
    # print('training...')

    total_acc_train = 0
    total_loss_train = 0

    cnn_model.train()
    basemodel.train()

    for batch in tqdm(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch


        bert_output = basemodel(input_ids, input_mask)
        train_label = label_list_to_single_label(label_ids)

        bert_loss = criterion(bert_output, label_ids)

        cnn_input_ids = format_embedding_to_batch_cnn(word2idx, tokenizer, input_ids)

        cnn_logits = cnn_model(cnn_input_ids)
        cnn_loss = criterion(cnn_logits, label_ids)


        loss = alpha*cnn_loss + (1-alpha)*bert_loss

        total_loss_train += loss.item()

        # logits = bert_output
        logits = alpha*cnn_logits + (1-alpha)*bert_output

        acc = (logits.argmax(dim=1) == train_label).sum().item()
        total_acc_train += acc

        alpha = alpha-(cnn_loss-bert_loss)*alpha_lr*alpha     ## 对alpha进行将梯度下降调整
        alpha = alpha.item()     ## tensor中的alpha只取数值

        basemodel.zero_grad()
        cnn_model.zero_grad()
        loss.backward()
        bert_optimizer.step()
        cnn_optimizer.step()


    y_true = []
    y_pred = []
    total_acc_val = 0
    total_loss_val = 0


    cnn_model.eval()
    basemodel.eval()
    print('evaluating...')
    with torch.no_grad():

        for step, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch
            val_label = label_list_to_single_label(val_label_ids)

            cnn_val_input_ids = format_embedding_to_batch_cnn(word2idx, tokenizer, val_input_ids)

            val_bert_output = basemodel(val_input_ids, val_input_mask)
            val_cnn_logits = cnn_model(cnn_val_input_ids)

            val_cnn_loss = criterion(val_cnn_logits, val_label_ids)
            val_bert_loss = criterion(val_bert_output, val_label_ids)

            # val_logits = val_bert_output

            val_logits = (1 - alpha) * val_bert_output + alpha*val_cnn_logits


            # val_loss = val_bert_loss
            val_loss = criterion(val_logits, val_label_ids)
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


evaluate(basemodel, cnn_model, embed_lookup, tokenizer, seq_len, alpha, test_dataloader)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list(range(0, epochs)), training_loss_list, '-r')

ax2 = ax.twinx()
ax2.plot(list(range(0, epochs)), total_acc_val_list)
plt.show()
# torch.save(model.state_dict(), 'bert_output.pkl')




# model.load_state_dict(torch.load('bert_output.pkl'))
# # evaluate(model, test_dataloader)
# evaluate(model, embed_lookup, tokenizer, seq_len, test_dataloader)




