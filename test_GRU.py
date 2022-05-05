import nltk
import string
from nltk import word_tokenize
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

enstop = stopwords.words('english')
punct = string.punctuation

def tokenizer(sent):
    sent = sent.lower()
    tmp = word_tokenize(sent)
    res = []
    for word in tmp:
        if word not in enstop and word not in punct:
            res.append(word)
    return res

import torch
import torch.nn as nn
from torchtext import data
from torchtext import vocab

text_field = data.Field(tokenize=tokenizer, lower=True, include_lengths=True,
                        fix_length=256)
label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.long)
train, valid, test = data.TabularDataset.splits(path='',
                                                train='imdb-train.csv',
                                                validation='imdb-valid.csv',
                                                test='imdb-test.csv',
                                                format='csv', skip_header=True,
                                                fields=[('sentence', text_field), ('label', label_field)])


vec = vocab.Vectors(name='glove.6B.300d.txt')
text_field.build_vocab(train, valid, test, max_size=250000, vectors=vec,
                       unk_init=torch.Tensor.normal_)
label_field.build_vocab(train, valid, test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test), batch_sizes=(64, 64, 64),
                                                               sort_key=lambda x: len(x.sentence),
                                                               sort_within_batch=True,
                                                               repeat=False, shuffle=True,
                                                               device=device)


def train_fun(model, train_iter, dev_iter, num_epoch, opt, criterion, eva,
              out_model_file):
    model.train()
    loss_list = []
    dev_acc = []
    best_dev_acc = 0.
    for epoch in range(num_epoch):
        total_loss = 0.
        for batch in train_iter:
            output = model(batch.sentence)
            loss = criterion(output, batch.label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        loss_list.append(total_loss)
        dev_acc.append(eva(model, dev_iter))
        print(f"Epoch: {epoch+1}/{num_epoch}. Total loss: {total_loss:.3f}. Validation Set Acc: {dev_acc[-1]:.3%}.")
        if dev_acc[-1] > best_dev_acc:
            best_dev_acc = dev_acc[-1]
            # torch.save(model.state_dict(), out_model_file)
    return loss_list, dev_acc


def eva(model, data_iter):
    correct, count = 0, 0
    with torch.no_grad():
        for batch in data_iter:
            pred = model(batch.sentence)
            pred = torch.argmax(pred, dim=-1)
            correct += (pred == batch.label).sum().item()
            count += len(pred)
    return correct / count


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
        text, text_length = x  # text: [seq_len, bs]
        text = text.permute(1, 0)  # text: [bs, seq_len]
        embed_x = self.embed(text)  # embed_x: [bs, seq_len, embed_dim]
        embed_x = embed_x.permute(1, 0, 2)  # embed_x: [seq_len, bs, embed_dim]
        hidden, _ = self.rnn(embed_x, text_length)  # hidden: [2*num_layers, bs, hidden_size]
        hidden = torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1)
        return self.proj(self.dropout(hidden))



embed_size = 300
hidden_size = 300
num_layers = 2
bidirectional = True
out_dim = 2
dropout_rate = 0.2
pretrained_embed = text_field.vocab.vectors
lr = 0.001
num_epoch = 20
freeze = True
use_gru = True
random_embed = False
vocab_size = len(text_field.vocab.stoi)
out_model_file = 'textrnn_gru_freeze.pt'

textrnn_gru_freeze = TextRNN(embed_size, hidden_size, num_layers, bidirectional, out_dim,
                             dropout_rate, pretrained_embed, use_gru=use_gru, freeze=freeze,
                             random_embed=random_embed, vocab_size=None).to(device)
opt = torch.optim.Adam(textrnn_gru_freeze.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
print("Training begin!")
loss_list, dev_acc_list = train_fun(textrnn_gru_freeze, train_iter, valid_iter, num_epoch, opt, criterion,
                                    eva, out_model_file)