import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import os
import sys

# DATASET = 'spo_top_10_2w'
DATASET = 'nyt_fb_v2'
NUM_LABELS = int(sys.argv[1])
print("=====BERT NUM LABELS=====")
print(NUM_LABELS)
CUDA = '0'
MINI_BATCH_SIZE = 10000

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
sentence_train = json.load(open(DATASET+'/sentence_train.json', 'r'))
sentence_train_label = json.load(open(DATASET+'/sentence_train_label.json', 'r'))
sentence_train_index = json.load(open(DATASET+'/sentence_index.json', 'r'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

print(torch.cuda.get_device_name(0))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
model.cuda()

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentence_train]
MAX_LEN = 128
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(input_ids)
train_index = torch.tensor(sentence_train_index)
train_labels = torch.tensor(sentence_train_label)
train_masks = torch.tensor(attention_masks)

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 16

train_data = TensorDataset(train_inputs, train_index, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

print("Starting LOOP")

for loop in range(5):
    print("Start loop {}, iter 1".format(loop))
    model_1 = nn.Sequential(*list(model.children())[:-2])

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 1

    labels_flat_3 = []
    CLS_temper_data = []
    marker_temper_data = []
    sentence_train_label_2 = []
    input_ids_data_temp = []
    pred_flat_2 = []
    input_ids_data = []
    # trange is a tqdm wrapper around the normal python range

    for _ in trange(epochs, desc="Epoch"):

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_index, b_labels = batch
            # Forward pass
            outputs = model_1(b_input_ids)
            last_hidden_states = outputs[0][11]
            logits = last_hidden_states.detach().cpu().numpy()
            index_ids = b_index.to('cpu').numpy()
            for i in range(len(index_ids)):
                temp1 = logits[i, index_ids[i][0]]
                temp2 = logits[i, index_ids[i][1]]
                temp = np.append(temp1, temp2)
                marker_temper_data.append(temp)
            # Move logits and labels to CPU
            input_ids = b_input_ids.to('cpu').numpy()
            label_ids = b_labels.to('cpu').numpy()
            input_ids = input_ids.tolist()
            input_ids_data_temp.append(input_ids)
            labels_flat_2 = label_ids.tolist()
            labels_flat_3.append(labels_flat_2)

        # KM = MiniBatchKMeans(n_clusters=NUM_LABELS, random_state=0,  batch_size=MINI_BATCH_SIZE)

        import pickle

        with open('BERT_out_'+str(NUM_LABELS), 'wb') as fp:
            print(type(marker_temper_data), len(marker_temper_data), marker_temper_data[0])
            pickle.dump(marker_temper_data, fp) 


        print("==== Dec Start ====")
        import subprocess
        # py_name = "dec-" + str(NUM_LABELS) + ".py"
        py_name = "dec-base.py"
        cmd = "python " + py_name + " --cluster_num " + str(NUM_LABELS)# 
        _, out = subprocess.getstatusoutput(cmd)
        print(out)

        with open ('DEC_out_'+str(NUM_LABELS), 'rb') as fp:
            pred_flat_2 = pickle.load(fp)

        print("==== Dec Finished ====")
        print("Pred_flat_2 length: {}, sample: {}".format(len(pred_flat_2), pred_flat_2[:10]))
        # import ipdb; ipdb.set_trace()

        # KM=KMeans(n_clusters=NUM_LABELS, random_state=0)
        # kmeans = KM.fit(marker_temper_data) # [total, 768 * 2]
        # k_means_label = kmeans.labels_
        # pred_flat_2 = k_means_label.tolist() # [total] 

        for i in range(len(labels_flat_3)):
            for j in range(len(labels_flat_3[i])):
                sentence_train_label_2.append(labels_flat_3[i][j])
        for i in range(len(input_ids_data_temp)):
            for j in range(len(input_ids_data_temp[i])):
                input_ids_data.append(input_ids_data_temp[i][j])

        file_name = DATASET + '_'  + str(NUM_LABELS) + '_fixed_layer_k_means_label_%d.json' % loop
        # file_name = 'TRex_SPO_preprocess_v3_00_fixed_layer_k_means_label_%d.json' % loop
        # with open(file_name, 'a+') as file_object:
        with open(file_name, 'w') as file_object:
            json.dump(pred_flat_2, file_object)
        file_name = DATASET + '_'  + str(NUM_LABELS) +  '_fixed_layer_label_ids_%d.json' % loop
        # file_name = 'TRex_SPO_preprocess_v3_500_fixed_layer_label_ids_%d.json' % loop
        with open(file_name, 'w') as file_object:
            json.dump(labels_flat_3, file_object)

    print("Finish loop {}, iter 1".format(loop))
    
    print("Start loop {}, iter 2".format(loop))
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids_data:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    train_inputs_2 = torch.tensor(input_ids_data)
    train_labels_2 = torch.tensor(pred_flat_2)
    train_true_labels_2 = torch.tensor(sentence_train_label_2)
    train_masks_2 = torch.tensor(attention_masks)

    batch_size = 16

    train_data_2 = TensorDataset(train_inputs_2, train_masks_2, train_labels_2, train_true_labels_2)
    train_sampler_2 = RandomSampler(train_data_2)
    train_dataloader_2 = DataLoader(train_data_2, sampler=train_sampler_2, batch_size=batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    # This variable contains all of the hyperparemeter information our training loop needs
    if (loop <= 2):
        ct = 0
        for child in model.children():
            ct += 1
            if ct < 1:
                for param in child.parameters():
                    param.requires_grad = False

    if (loop > 2):
        ct = 0
        for child in model.children():
            ct += 1
            if ct < 1:
                for param in child.parameters():
                    param.requires_grad = True

    optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=1e-5,
                         warmup=.1)


    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 5
    pseudo_label_ids_data = []
    b_labels_id_data = []
    logits_data = []
    # trange is a tqdm wrapper around the normal python range
    for epoch in trange(epochs, desc="Epoch"):

        #
        # # Set our model to training mode (as opposed to evaluation mode)
        # model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader_2):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_pseudo_labels, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_pseudo_labels)
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

    print("Finsh second loop!")
    print("Start loop {}, iter 2".format(loop))