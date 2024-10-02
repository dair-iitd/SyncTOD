import json
import argparse


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def post_process(text):
    text = text.lower()
    text = text.split('\n')[-1]
    if 'assistant:' in text:
        text = text.split('assistant:')[-1].strip()
    if 'system:' in text:
        text = text.split('system:')[-1].strip()

    return text


def process_batch_preds(batch_pred_path, datapath, include_entities=False):
    batch_prediction = load_jsonl(batch_pred_path)
    data = load_json(datapath)

    idx2pred = dict()
    for entry in batch_prediction:
        tmp = entry['custom_id'].split('_')
        uuid = '_'.join(tmp[1:-1])
        turn_id = int(tmp[-1])

        idx2pred[(uuid, turn_id)] = post_process(
            entry['response']['body']['choices'][0]['message']['content']
        )

    predictions = []
    outputs = []
    for entry in data:
        uuid, turn_id = entry['uuid'], entry['turn_id']
        predictions.append(idx2pred[(uuid, turn_id)])
        outputs.append(entry['output'])

    if  not include_entities:
        return predictions, outputs

    entities = []
    for entry in data:
        entities.append([x[1] for x in entry['gold_entities']])

    return predictions, outputs, entities
