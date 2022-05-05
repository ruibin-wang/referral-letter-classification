import tokenizers
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import pandas as pd

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        # self.relu = nn.ReLU()
        self.relu = nn.Sigmoid()

    def forward(self, input_id):

        _, pooled_output = self.bert(input_ids=input_id, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)


        return final_layer


def evaluate(bert_model, text_input):

    test_input_ids = generate_model_input(text_input)
        #
        # for step, batch in enumerate(test_dataloader):
        #     batch = tuple(t.to(device) for t in batch)
        #     test_input_ids, test_input_mask, test_segment_ids, test_label_ids = batch
        #     test_label = label_list_to_single_label(test_label_ids)

    test_logits = bert_model(test_input_ids)
    bert_result = test_logits.argmax(dim=1)

    return bert_result


def generate_model_input(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokens = tokenizer.tokenize(text)
    input_text_tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_text_tokens)
    padding = [0] * (512 - len(input_ids))

    input_ids += padding
    input_ids = pd.array(input_ids)

    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


text = 'We would be grateful if you could review this 56 year old gentleman. He was admitted on the 27-11-18 after falling down the stairs. ...' \
       'Preceding dizziness where he felt he would pass out, but did not lose consciousness. 2 weeks ago states developed sudden onset headache - vice like, ...' \
       'severity at start 8-10. Not positional.States having tunnel vision right eye. On examination had reduced abduction right eye and diplopia. ...' \
       'Case discussed with Dr Cossburn who suggested MRI. The MRI report is as below and suggests neurology review'

bert_model = BertClassifier()
bert_model.load_state_dict(torch.load('C:\Ruibin\Code\intelPA_rwang-main\intelPA\\graph\\bert_output.pkl'))



bert_result = evaluate(bert_model, text)

diseases = ['Epilepsy and recurrent seizures', 'Headache', 'Dorsalgia', 'Cerebral infarction']
labels = {
    'G40': 0,
    'R51': 1,
    'M54': 2,
    'I63': 3

}

disease = diseases[bert_result.item()]

print('You probably have ' + disease)








