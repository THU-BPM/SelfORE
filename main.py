import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer

parser = argparse.ArgumentParser(description='SelfORE source code')
parser.add_argument('--cluster', default='kmeans',
                    help='cluster algorithm, adaptive clustering, kmeans, etc.')
parser.add_argument('--k', default='1000', help='the number of cluster')
parser.add_argument('--loop', default='5', help='the loop number of SelfORE')
parser.add_argument('--token_num_workers', default='10',
                    help='multiprocess tokenize num workers')
parser.add_argument('--bert_max_len', default='128',
                    help='bert embedding max length')
parser.add_argument('--batch_size', default='64', help='batch size')
parser.add_argument('--sentences_path', default='data/sentence_sample',
                    help='sentences of dataset')
args = parser.parse_args()


def get_sentences(sentences_path):
    with open(sentences_path) as f:
        sentences = f.readlines()
    return sentences


def generate_token():
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    special_tokens_dict = {'additional_special_tokens': ['[E1]', '[E2]', '[/E1]', '[/E2]']} # add special token
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def generate_model(tokenizer):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.k,
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    return model


def prepare_data(sentences, tokenizer):
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.bert_max_len,           # Pad & truncate all sentences.
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,   # Construct attn. masks.
            return_tensors='pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_id = encoded_dict['input_ids']
        input_ids.append(input_id)

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    entity_idx = generate_entity_idx(tokenizer, input_ids)
    return input_ids, attention_masks, entity_idx


def generate_entity_idx(tokenizer, input_ids):
    e1_tks_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_tks_id = tokenizer.convert_tokens_to_ids('[E2]')
    entity_idx = []
    for input_id in input_ids:
        e1_idx = (input_id == e1_tks_id).nonzero().flatten().tolist()[0]
        e2_idx = (input_id == e2_tks_id).nonzero().flatten().tolist()[0]
        entity_idx.append((e1_idx, e2_idx))
    entity_idx = torch.Tensor(entity_idx)
    return entity_idx


# Start LOOP
def selfore_loop():
    pass


if __name__ == "__main__":
    sentences = get_sentences(args.sentences_path)
    tokenizer = generate_token()
    model = generate_model(tokenizer)
    input_ids, attention_masks, entity_idx = prepare_data(sentences, tokenizer)
    for i in range(args.loop):
        print("Loop {} start".format(i))
        selfore_loop()
