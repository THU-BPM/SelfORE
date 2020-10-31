import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import random
import numpy as np


class Classifier:
    def __init__(self, k, sentence_path, max_len, batch_size, epoch):
        with open(sentence_path) as f:
            self.sentences = f.readlines()
        self.k = k
        self.epoch = epoch
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.device = self.get_device()
        self.input_ids, self.attention_masks = self.prepare_data()
        # self.entity_idx = self.get_entity_idx()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device

    @staticmethod
    def get_tokenizer():
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        special_tokens_dict = {'additional_special_tokens': [
            '[E1]', '[E2]', '[/E1]', '[/E2]']}  # add special token
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    def get_model(self):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=self.k,
            output_attentions=False,
            output_hidden_states=True,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def prepare_data(self):
        input_ids = []
        attention_masks = []
        for sent in self.sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,                        # Sentence to encode.
                add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                max_length=self.max_len,     # Pad & truncate all sentences.
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',         # Return pytorch tensors.
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        print('Original: ', self.sentences[0])
        print('Token IDs:', input_ids[0])
        return input_ids, attention_masks

    def get_entity_idx(self):
        e1_tks_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        e2_tks_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        entity_idx = []
        for input_id in self.input_ids:
            e1_idx = (input_id == e1_tks_id).nonzero().flatten().tolist()[0]
            e2_idx = (input_id == e2_tks_id).nonzero().flatten().tolist()[0]
            entity_idx.append((e1_idx, e2_idx))
        entity_idx = torch.Tensor(entity_idx)
        return entity_idx

    def get_hidden_state(self):
        self.model.to('cpu')
        outputs = self.model(self.input_ids, self.attention_masks)
        return outputs[1][-1].detach().numpy().flatten().reshape(self.input_ids.shape[0], -1)

    def train(self, labels):
        labels = torch.tensor(labels).long()
        dataset = TensorDataset(self.input_ids, self.attention_masks, labels)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size
        )

        self.validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=self.batch_size
        )
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        epochs = self.epoch
        total_steps = len(self.train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        self.model.cuda()
        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            self.train_epoch()

    def train_epoch(self):
        total_train_loss = 0
        self.model.train()
        for batch in self.train_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            self.model.zero_grad()
            loss, logits, _ = self.model(b_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_input_mask,
                                         labels=b_labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
        avg_train_loss = total_train_loss / len(self.train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("")
        print("Running Validation...")
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in self.validation_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            with torch.no_grad():
                (loss, logits, _) = self.model(b_input_ids,
                                               token_type_ids=None,
                                               attention_mask=b_input_mask,
                                               labels=b_labels)
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / \
            len(self.validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
