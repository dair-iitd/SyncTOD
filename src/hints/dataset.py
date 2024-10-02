from collections import Counter
import json
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

from commons.metrics import EntityTypeMetric
from commons.text_process import preprocess_text


def linearize_knowledge_row_major(knowledge, fields):
    """Convert knowledge into a flatten sequecen with special symbols."""
    knowledge_seq = []
    for idx, record in enumerate(knowledge):
        tmp = dict()
        for f in fields:
            tmp[f] = record.get(f, 'none')
        knowledge_seq.append(json.dumps(tmp))

    return "[" + ",".join(knowledge_seq) + "]"


def linearize_knowledge_column_major(knowledge, fields):
    """Convert knowledge into a flatten sequecen with special symbols."""
    col2values = dict()
    for f in fields:
        col2values[f] = []
        for record in knowledge:
            col2values[f].append(record[f])

    return json.dumps(col2values)


def linearize_knowledge(knowledge, fields, kb_format):
    if kb_format == 'row_major':
        return linearize_knowledge_row_major(knowledge, fields)
    elif kb_format == 'col_major':
        return linearize_knowledge_column_major(knowledge, fields)
    else:
        raise NotImplementedError


def collate_fn(batch):
    max_input_length = -1

    for entry in batch:
        if max_input_length < len(entry['input_ids']):
            max_input_length = len(entry['input_ids'])

    assert max_input_length > 0
    
    bs = len(batch)
    input_token_ids = np.zeros((bs,  max_input_length), dtype=np.int64)
    attention_mask = np.zeros((bs,  max_input_length), dtype=np.int64)
    labels = np.zeros((bs,), dtype=np.int64)
    for idx, entry in enumerate(batch):
        in_length = len(entry['input_ids'])
        input_token_ids[idx, :in_length] = entry['input_ids']
        attention_mask[idx, :in_length] = entry['attention_mask']
        labels[idx] = entry['target']

    ret = {
        "input_ids": input_token_ids,
        "attention_mask": attention_mask,
        'labels': labels
    }

    tret = dict()
    for k in ret:
        tret[k] = torch.tensor(ret[k])

    return tret


class BasicDataset(Dataset):
    def __init__(self, data_loc, tokenizer, mode, cfg):
        """
        :param data_loc: str destination location
        :param vocab: BasicVocabulary object of prebuilt-vocabulary
        """
        self.data_loc = data_loc
        self.raw_data = None
        self.data = None
        self.tokenizer = tokenizer
        self.mode = mode
        self.dataset_name = cfg['dataset_name']
        self.kb_key = cfg['model'].get('kb_key', 'kb')
        self.use_kb = cfg['model'].get('use_kb', False)
        self.ctx_length = cfg['model'].get('ctx_length', None)
        self.ctx_format = cfg['model'].get('ctx_format', 'list')

        if self.ctx_format == 'text':
            assert self.ctx_length == 1, f"Only last utterance is supported in text mode"

        self.load_data_from_file(data_loc)

    def load_data_from_file(self, fname):
        """
        Load data from json file.
        :param fname: str location of the data file.
        """
        with open(fname, 'r') as fp:
            self.raw_data = json.load(fp)

        self.process_data()

    def get_input_seq(self, sample):
        context = sample['context']
        if self.ctx_length is not None:
            context = context[-self.ctx_length:]
        kb = sample[self.kb_key]
        if kb is None:
            kb = []

        prompt = '[dialog] '

        if self.ctx_format == 'list':
            temp = []
            for ii, uttr in enumerate(context):
                spk = 'user' if ii % 2 == 0 else 'bot'
                temp.append(f"{spk}: {uttr}")

            prompt += ' '.join(temp)
            # prompt += ' | '.join(temp)
        elif self.ctx_format == 'text':
            text = sample['context_used'].split('<user>')[-1].strip()
            prompt += text

        if len(kb) > 0 and self.use_kb:
            fields = list(kb[0])
            if self.dataset_name == 'CamRest':
                fields = [x for x in fields if x not in ["id", "location", "type"]]

            prompt += ' [kb] ' + linearize_knowledge(kb, fields)

        return prompt

    def process_data(self):
        print('Processing data...')

        # prepare input samples
        samples = []
        for sample in tqdm(self.raw_data):
            input_seq = self.get_input_seq(sample)
            tin = self.tokenizer(input_seq, return_tensors="np")
            hints = sample['hints']
            tsample = {
                'input_seq': input_seq,
                'output': sample['output'],
                'type': sample['type'],
                'input_ids': tin.input_ids[0],
                'attention_mask': tin.attention_mask[0],
                'hints': sample['hints'],
                'target': int(hints['closure'])
            }
            samples.append(tsample)

        self.data = samples
        print(f"Sample type stats", Counter([x['type'] for x in samples]))
        print(f"Sample target stats", Counter([x['target'] for x in samples]))

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def reset_samples(self):
        self.process_data()


def seq2seq_collate_fn(batch):
    max_input_length = -1
    max_output_length = -1
    train_mode = 'labels' in batch[0]

    for entry in batch:
        if max_input_length < len(entry['input_ids']):
            max_input_length = len(entry['input_ids'])

        if train_mode and max_output_length < len(entry['labels']):
            max_output_length = len(entry['labels'])

    assert max_input_length > 0

    bs = len(batch)
    input_token_ids = np.zeros((bs,  max_input_length), dtype=np.int64)
    attention_mask = np.zeros((bs,  max_input_length), dtype=np.int64)

    if train_mode:
        labels = np.ones((bs,  max_output_length), dtype=np.int64) * (-100)

    for idx, entry in enumerate(batch):
        in_length = len(entry['input_ids'])
        input_token_ids[idx, :in_length] = entry['input_ids']
        attention_mask[idx, :in_length] = entry['attention_mask']

        if train_mode:
            out_length = len(entry['labels'])
            labels[idx, :out_length] = entry['labels']

    input_token_ids = torch.tensor(input_token_ids)
    ret = {
        "input_ids": input_token_ids,
        "attention_mask": attention_mask,
    }

    if train_mode:
        labels = torch.tensor(labels)
        ret['labels'] = labels

    for k in ret:
        ret[k] = torch.tensor(ret[k])

    return ret


class Seq2SeqDataset(Dataset):
    def __init__(self, data_loc, tokenizer, mode, cfg):
        """
        :param data_loc: str destination location
        :param vocab: BasicVocabulary object of prebuilt-vocabulary
        """
        self.data_loc = data_loc
        self.raw_data = None
        self.data = None
        self.tokenizer = tokenizer
        self.mode = mode
        self.dataset_name = cfg['dataset_name']
        self.kb_key = cfg['model'].get('kb_key', 'kb')
        self.use_kb = cfg['model'].get('use_kb', False)
        self.ctx_length = cfg['model'].get('ctx_length', None)
        self.dedup_etypes = cfg['model'].get('dedup', True)
        self.kb_format = cfg['model'].get('kb_format', 'row_major')
        self.ctx_format = cfg['model'].get('ctx_format', 'list')
        self.cfg = cfg

        assert self.ctx_length is None, "Only full context is supported!."

        if not self.dedup_etypes:
            print('Deduplication of output is disabled')

        self.load_data_from_file(data_loc)

    def load_data_from_file(self, fname):
        """
        Load data from json file.
        :param fname: str location of the data file.
        """
        with open(fname, 'r') as fp:
            self.raw_data = json.load(fp)

        self.process_data()

    def get_input_seq(self, sample):
        context = sample['context']
        if self.ctx_length is not None:
            context = context[-self.ctx_length:]
        kb = sample[self.kb_key]
        if kb is None:
            kb = []

        if len(kb) > 0:
            prompt = "Based on the [dialog] and [kb], generate entity types to be included in the response:"
        else:
            prompt = "Based on the [dialog], generate entity types to be included in the response:"

        if self.ctx_format == 'list':
            temp = []
            for ii, uttr in enumerate(context):
                spk = 'user' if ii % 2 == 0 else 'bot'
                temp.append(f"{ii + 1}. {spk}: {uttr}")
            prompt += ' [dialog] ' + ' '.join(temp)
        elif self.ctx_format == 'text':
            prompt += ' [dialog] ' + sample['context_used']

        if len(kb) > 0 and self.use_kb:
            fields = list(kb[0])
            if self.dataset_name == 'CamRest':
                fields = [x for x in fields if x not in ["id", "location", "type"]]
            prompt += ' [kb] ' + linearize_knowledge(kb, fields, self.kb_format)

        return prompt

    def process_data(self):
        print('Processing data...')

        # prepare input samples
        samples = []
        for sample in tqdm(self.raw_data):
            input_seq = self.get_input_seq(sample)
            tin = self.tokenizer(input_seq, return_tensors="np")
            tsample = {
                'input_seq': input_seq,
                'input_ids': tin.input_ids[0],
                'attention_mask': tin.attention_mask[0],
            }

            if self.dataset_name == 'SMD':
                # print('cleaning labels.')
                output_etypes = get_cleaned_smd_labels(sample)
            else:
                output_etypes = sample['hints']['entity_types']
            if len(output_etypes) > 0:
                if self.dedup_etypes:
                    output_etypes = sorted(set(output_etypes))
                else:
                    output_etypes = sorted(output_etypes)
                output_seq = ' | '.join(output_etypes)
            else:
                output_seq = '[no entity]'

            tsample['output_seq'] = output_seq
            if self.mode == 'train':
                tsample['labels'] = self.tokenizer(output_seq, return_tensors="np").input_ids[0]

            samples.append(tsample)

        self.data = samples

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def reset_samples(self):
        self.process_data()


def get_cleaned_smd_labels(sample):
    return sample['hints']['entity_types']

