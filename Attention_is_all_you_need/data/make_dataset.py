import gzip
import json
import base64
import functools

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from indices_blob import blob

def tokenize(tokenizer, data):
    inputs = [doc for doc in data['documents']]
    summary = [summary for summary in data['summary']]

    inputs = tokenizer(inputs, max_length='1024', truncation=True)
    labels = tokenizer(summary, max_length='128', truncation=True)
    inputs['labels'] = labels['input_ids']

    return inputs

def make_dataset():
    xsum = load_dataset('xsum')
    train_indices, validation_indices, test_indices = json.loads(
        gzip.decompress(base64.b64decode(blob)).decode("utf-8"))

    dataset_train = xsum['train'].select(train_indices)
    dataset_validation = xsum['validation'].select(validation_indices)
    dataset_test = xsum['test'].select(test_indices)

    tokenizer_base = AutoTokenizer.from_pretrained('t5-small')
    tokenizer = functools.partial(tokenize, tokenizer_base)

    dataset_train = dataset_train.map(tokenizer, batched=True)
    dataset_validation = dataset_validation.map(tokenizer, batched=True)
    dataset_test = dataset_test.map(tokenizer, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer_base, model="t5-small")
    return dataset_train, dataset_validation, dataset_test, data_collator, tokenizer_base