import os
import json
import re

import numpy as np
import torch

from .dataset import BasicDataset, collate_fn, seq2seq_collate_fn, Seq2SeqDataset
from .utils import read_cli, get_config, override_config, train_tag, val_tag, entity_tag, test_tag
from .trainers import ClosureTrainer, Seq2SeqTrainer

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import TrainingArguments
from transformers import T5ForConditionalGeneration, T5Tokenizer

os.environ["WANDB_DISABLED"] = "true"


def get_best_model_checkpoint(dir_path):
    # Step 1: Collect all the folders of the form "checkpoint-<id>"
    checkpoint_folders = []
    for folder_name in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, folder_name)):
            match = re.match(r'checkpoint-(\d+)', folder_name)
            if match:
                checkpoint_folders.append((int(match.group(1)), folder_name))
    
    if not checkpoint_folders:
        raise ValueError(f"No checkpoint folders found in directory {dir_path}")
    
    # Step 2: Select the "checkpoint-<id>" with maximum id
    max_id, max_checkpoint_folder = max(checkpoint_folders, key=lambda x: x[0])

    # Step 3: Load trainer_state.json from dir/checkpoint-<maxid>
    trainer_state_path = os.path.join(dir_path, max_checkpoint_folder, 'trainer_state.json')
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"trainer_state.json not found in {trainer_state_path}")

    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    # Step 4: Get the value of 'best_model_checkpoint'
    if 'best_model_checkpoint' not in trainer_state:
        raise KeyError("'best_model_checkpoint' key not found in trainer_state.json")

    best_model_checkpoint = trainer_state['best_model_checkpoint']
    
    return best_model_checkpoint


def run_closure(args):
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(args['model']['wildcard'])

    val_file = os.path.join(args['datapath'], test_tag)
    val_dataset = BasicDataset(val_file, tokenizer, mode='infer', cfg=args)

    train_args = TrainingArguments(
        output_dir=args['destpath'],
        overwrite_output_dir=True,
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
    )

    chkpt_path = get_best_model_checkpoint(args['destpath'])
    # chkpt_path = os.path.join(args['destpath'], f"checkpoint-{args['chkpt']}")
    print(f"Loading model from {chkpt_path}")
    model = DebertaV2ForSequenceClassification.from_pretrained(chkpt_path)

    trainer = ClosureTrainer(
        tokenizer=tokenizer,
        model=model, args=train_args,
        # train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # metrics = trainer.evaluate(train_dataset)
    metrics = trainer.evaluate(
        val_dataset, save_results=True,
        result_path=args['result_path']
    )
    print(metrics)


def run_etypes(args):
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

    chkpt_path = get_best_model_checkpoint(args['destpath'])
    # chkpt_path = os.path.join(args['destpath'], f"checkpoint-{args['chkpt']}")
    print(f"Loading model from {chkpt_path}")
    model = T5ForConditionalGeneration.from_pretrained(chkpt_path)
    tokenizer = T5Tokenizer.from_pretrained(args['model']['wildcard'], model_max_length=args['model'].get('model_max_length'))

    print(args['datapath'])
    val_file = os.path.join(args['datapath'], test_tag)
    val_dataset = Seq2SeqDataset(
        val_file, tokenizer, mode='infer', cfg=args
    )

    train_args = TrainingArguments(
        output_dir=args['destpath'],
        overwrite_output_dir=True,
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
    )

    trainer = Seq2SeqTrainer(
        cfg=args,
        tokenizer=tokenizer,
        model=model, args=train_args,
        # train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=seq2seq_collate_fn,
    )
    
    # metrics = trainer.evaluate(train_dataset)
    metrics = trainer.evaluate(
        val_dataset, save_results=True,
        result_path=args['result_path']
    )
    print(metrics)


if __name__ == "__main__":
    cargs = read_cli()

    # model_path = cargs['model_path']
    cfg_path = cargs['config']
    args = get_config(cfg_path)
    # args['destpath'] = model_path
    if cargs['datapath'] is not None:
        args['datapath'] = cargs['datapath']
    else:
        del cargs['datapath']

    if cargs['batch_size'] is not None:
        args['dev']['per_device_eval_batch_size'] = cargs['batch_size']
    else:
        del cargs['batch_size']

    args.update(cargs)
    if args['hint'] == 'closure':
        run_closure(args)
    elif args['hint'] == 'entity_types':
        # args['model']['dedup'] = True
        run_etypes(args)
