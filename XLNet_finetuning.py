import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


import torch
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)



df = pd.read_csv('./data/combine_complaint_symp.csv')
# df = pd.read_csv('./data/complaint_training_data.csv')
# df = pd.read_csv('./data/symp_diagnosis_relation_training.csv')


sentences = df.symptoms.values
sentences = [sentence + " [SEP] [CLS]" for sentence in sentences]

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



num_labels, labels = len(labels_4), labels_4


labels = [labels[label] for label in df['diagnosis']]


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]



MAX_LEN = 128
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)


#Use train_test_split to split our data into train and validation sets for training

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                            random_state=56, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=56, test_size=0.2)


# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)



# Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
batch_size = 16

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top.

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)
model.cuda()


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


# This variable contains all of the hyperparemeter information our training loop needs
optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 80

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch


        b_input_ids = b_input_ids.to(device).long()
        b_input_mask = b_input_mask.to(device).long()
        b_labels = b_labels.to(device).long()


        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))
    # torch.save(model.state_dict(), 'model_without_language_model.ckpt')

    # # Validation

    # # Put model in evaluation mode to evaluate loss on the validation set
    # model.eval()

    # # Tracking variables
    # eval_loss, eval_accuracy = 0, 0
    # nb_eval_steps, nb_eval_examples = 0, 0

    # # Evaluate data for one epoch
    # for batch in validation_dataloader:
    #   # Add batch to GPU
    #   batch = tuple(t.to(device) for t in batch)
    #   # Unpack the inputs from our dataloader
    #   b_input_ids, b_input_mask, b_labels = batch
    #   # Telling the model not to compute or store gradients, saving memory and speeding up validation
    #   with torch.no_grad():
    #     # Forward pass, calculate logit predictions
    #     output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    #     logits = output[0]

    #   # Move logits and labels to CPU
    #   logits = logits.detach().cpu().numpy()
    #   label_ids = b_labels.to('cpu').numpy()

    #   tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    #   eval_accuracy += tmp_eval_accuracy
    #   nb_eval_steps += 1

    # print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))


# torch.save(model.state_dict(), directory_path+'/model_without_language_model.ckpt')
# Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(validation_dataloader):
          batch = tuple(t.to(device) for t in batch)
          # Unpack the inputs from our dataloader
          b_input_ids, b_input_mask, b_labels = batch

          b_input_ids = b_input_ids.to(device).long()
          b_input_mask = b_input_mask.to(device).long()
          b_labels = b_labels.to(device).long()


          # Forward pass
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          # print (outputs)
          prediction = torch.argmax(outputs[0], dim=1)
          total += b_labels.size(0)
          correct+=(prediction==b_labels).sum().item()

    print('Test Accuracy of the model on vla data is: {} %'.format(100 * correct / total))


print('11')