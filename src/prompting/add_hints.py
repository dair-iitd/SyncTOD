import os
import json
from tqdm import tqdm
import string
import argparse

from commons.metrics import EntityTypeMetric
from commons.text_process import preprocess_text


def read_args():
    parser = argparse.ArgumentParser(description='Add hints')
    parser.add_argument(
        "-src",
        "--src_path",
        help="Path to src file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-ent",
        "--entity_path",
        help="Path to entity",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        help="Dataset name",
        required=True,
        type=str,
        choices=['MultiWOZ', 'SMD', 'BiTOD']
    )
    parser.add_argument(
        "-hints",
        "--hints",
        help="Hints to add",
        required=True,
        type=str,
        choices=['oracle', 'predicted']
    )
    # Only for predicted
    parser.add_argument(
        "-cpred_path",
        "--closure_pred_path",
        help="Path to closure prediction",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "-epred_path",
        "--etypes_pred_path",
        help="Path to etypes prediction",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "-wl",
        "--word_length",
        help="Expected output size",
        required=False,
        type=int,
        default=None
    )
    args = parser.parse_args()

    return args


def count_words(text):
    # print(text.split())
    return len([x for x in text.split() if x not in string.punctuation])


def compute_hints(sample, entity_metric=None, output=None, word_count=None, etypes=None):
    ret = dict()
    
    if output is None:
        output = sample['output']

    if word_count is None:
        ret['word_count'] = count_words(output)
    else:
        ret['word_count'] = word_count

    if etypes is None:
        if entity_metric.dataset == 'BiTOD':
            ptext = preprocess_text(output, ignore_puncs=['#'])
        else:
            ptext = preprocess_text(output)
        gold_entities = entity_metric.extract_entities(ptext, return_types=True)
        ret['entity_types'] = [x[0] for x in gold_entities]
    else:
        ret['entity_types'] = etypes

    return ret


def insert_oracle_hints(src, tar, entity_metric):
    with open(src, 'r') as fp:
        data = json.load(fp)

    last_uttr_idx = dict()
    for ee in data:
        if ee['uuid'] not in last_uttr_idx:
            last_uttr_idx[ee['uuid']] = -1
        if ee['turn_id'] > last_uttr_idx[ee['uuid']]:
            last_uttr_idx[ee['uuid']] = ee['turn_id']

    for entry in tqdm(data):
        hints = compute_hints(entry, entity_metric)
        hints['closure'] = False
        if entry['turn_id'] == last_uttr_idx[entry['uuid']]:
            hints['closure'] = True
        entry['hints'] = hints

    print('SAVING', len(data))
    with open(tar, 'w') as fp:
        json.dump(data, fp, indent=2)


def insert_predicted_hints(
    src, tar,
    closure_pred_path,
    etypes_pred_path,
    word_length
):
    with open(src, 'r') as fp:
        data = json.load(fp)

    with open(closure_pred_path, 'r') as fp:
        closure_preds = json.load(fp)
        assert len(closure_preds) == len(data), f"Lens {len(closure_preds)} {len(data)}"

    with open(etypes_pred_path, 'r') as fp:
        etypes_preds = json.load(fp)
        assert len(etypes_preds) == len(data), f"Lens {len(etypes_preds)} {len(data)}"

    for ii, entry in enumerate(data):
        etypes = etypes_preds[ii]['prediction']
        hints = compute_hints(entry, output=None, word_count=word_length, etypes=etypes)

        # print(entry['turn_id'], closure_preds[ii]['turn_id'])
        assert entry['uuid'] == closure_preds[ii]['uuid']
        # assert entry['turn_id'] == closure_preds[ii]['turn_id']
        hints['closure'] = True if closure_preds[ii]['closure'] == 1 else False
        entry['predicted_hints'] = hints

    print('SAVING', len(data))
    with open(tar, 'w') as fp:
        json.dump(data, fp, indent=2)


def insert_entities(src, tar, entity_metric):
    with open(src, 'r') as fp:
        data = json.load(fp)

    for entry in tqdm(data):
        gold_entities = entity_metric.extract_entities(
            entry['output'], return_types=True
        )
        entry['output_entities'] = gold_entities

    print('SAVING', len(data))
    with open(tar, 'w') as fp:
        json.dump(data, fp, indent=2)


def main(args):
    if args.dataset == 'BiTOD' and args.hints == 'oracle':
        return
    src_path = args.src_path
    entity_metric = EntityTypeMetric(args.dataset, args.entity_path)

    if args.hints == 'oracle':
        print('Adding oracle hints.')
        insert_oracle_hints(src_path, src_path, entity_metric)

    elif args.hints == 'predicted':
        print('Adding predicted hints')
        assert args.closure_pred_path is not None
        assert args.etypes_pred_path is not None
        assert args.word_length is not None

        insert_predicted_hints(
            src_path, src_path,
            args.closure_pred_path,
            args.etypes_pred_path,
            args.word_length
        )


if __name__ == '__main__':
    args = read_args()
    main(args)
