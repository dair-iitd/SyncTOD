import os
import numpy as np
import torch

from transformers import IntervalStrategy, TrainingArguments

import wandb
from .trainers import ClosureTrainer, Seq2SeqTrainer
from .dataset import BasicDataset, collate_fn, Seq2SeqDataset, seq2seq_collate_fn
from .utils import read_cli, get_config, override_config, train_tag, val_tag, entity_tag

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration


def run_closure(args, report_to='tensorboard'):
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        args['model']['wildcard'],
        model_max_length=args['model']['max_input_length']
    )
    tokenizer.save_pretrained(args['destpath'])

    train_file = os.path.join(args['datapath'], train_tag)
    train_dataset = BasicDataset(train_file, tokenizer, mode='train', cfg=args)

    val_file = os.path.join(args['datapath'], val_tag)
    val_dataset = BasicDataset(val_file, tokenizer, mode='infer', cfg=args)
    print('-->', train_dataset[1]['input_seq'])
    print('-->', val_dataset[1]['input_seq'])

    model = DebertaV2ForSequenceClassification.from_pretrained(args['model']['wildcard'], num_labels=2)
    if args['train'].get('freeze_encoder', False):
        print('Freezeing encoder')
        for param in model.deberta.parameters():
            param.requires_grad = False

    # 4. Setup trainer arguments
    train_args = TrainingArguments(
        output_dir=args['destpath'],
        overwrite_output_dir=True,
        remove_unused_columns=False,
        log_level='info',
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
        num_train_epochs=args['train']['num_epochs'],
        learning_rate=args['train']['learning_rate'],
        max_steps=args['train'].get('max_steps', -1),
        # save_strategy='epoch',
        save_strategy=args['train'].get('save_strategy', IntervalStrategy.STEPS),
        save_steps=args['train'].get('save_eval_steps', 100),
        seed=args['train']['seed'],
        fp16=args['train']['fp16'],
        tf32=args['train'].get('tf32', False),
        gradient_checkpointing=args['train'].get('gradient_checkpointing', False),
        # evaluation_strategy="epoch",
        evaluation_strategy=args['train'].get('evaluation_strategy', IntervalStrategy.STEPS),
        eval_steps=args['train'].get('save_eval_steps', 100),
        gradient_accumulation_steps=args['train']['gradient_accumulation_steps'],
        logging_steps=1,
        ddp_find_unused_parameters=False,
        save_total_limit=args['train'].get('save_total_limit', 5),
        load_best_model_at_end=True,
        metric_for_best_model=args['train']['metric_for_best_model'],
        greater_is_better=args['train']['greater_is_better'],
        report_to=report_to,
        run_name=args['experiment_name'],
        warmup_ratio=args['train'].get('warmup_ratio', 0.0),
        dataloader_drop_last=True
    )

    trainer = ClosureTrainer(
        model=model, args=train_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=collate_fn
    )
    trainer.train(resume_from_checkpoint=False)


def run_etypes(args, report_to='tensorboard'):
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = T5ForConditionalGeneration.from_pretrained(args['model']['wildcard'])
    tokenizer = T5Tokenizer.from_pretrained(args['model']['wildcard'], model_max_length=args['model'].get('model_max_length'))

    train_file = os.path.join(args['datapath'], train_tag)
    train_dataset = Seq2SeqDataset(
        train_file, tokenizer, mode='train', cfg=args
    )

    val_file = os.path.join(args['datapath'], val_tag)
    val_dataset = Seq2SeqDataset(
        val_file, tokenizer, mode='infer', cfg=args
    )
    print(train_dataset[0]['input_seq'])
    print(train_dataset[0]['output_seq'])

    train_args = TrainingArguments(
        output_dir=args['destpath'],
        overwrite_output_dir=True,
        remove_unused_columns=False,
        log_level='warning',
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
        num_train_epochs=args['train']['num_epochs'],
        learning_rate=args['train']['learning_rate'],
        max_steps=args['train'].get('max_steps', -1),
        # save_strategy='epoch',
        save_strategy=args['train'].get('save_strategy', IntervalStrategy.STEPS),
        save_steps=args['train'].get('save_eval_steps', 100),
        seed=args['train']['seed'],
        fp16=args['train']['fp16'],
        tf32=args['train'].get('tf32', False),
        gradient_checkpointing=args['train'].get('gradient_checkpointing', False),
        # evaluation_strategy="epoch",
        evaluation_strategy=args['train'].get('evaluation_strategy', IntervalStrategy.STEPS),
        eval_steps=args['train'].get('save_eval_steps', 100),
        gradient_accumulation_steps=args['train']['gradient_accumulation_steps'],
        logging_steps=10,
        ddp_find_unused_parameters=False,
        save_total_limit=args['train'].get('save_total_limit', 5),
        load_best_model_at_end=True,
        metric_for_best_model=args['train']['metric_for_best_model'],
        greater_is_better=args['train']['greater_is_better'],
        report_to=report_to,
        run_name=args['experiment_name'],
        warmup_ratio=args['train'].get('warmup_ratio', 0.0),
        dataloader_drop_last=True,
        # lr_scheduler_type="constant",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=train_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=seq2seq_collate_fn,
        cfg=args, tokenizer=tokenizer
    )
    trainer.train(resume_from_checkpoint=args['train']['resume_training'])


if __name__ == "__main__":
    cargs = read_cli()
    args = get_config(cargs['config'])
    args = override_config(args, cargs)

    local_rank = os.environ.get('LOCAL_RANK', '')
    report_to = 'tensorboard'
    if args.get('use_wandb', False):
        import wandb
        wandb.init(
            group=args['experiment_name'],
            name=args['experiment_name'] + local_rank,
            resume='allow'
        )
        report_to='wandb'

    if args['hint'] == 'closure':
        run_closure(args, report_to=report_to)
    elif args['hint'] == 'entity_types':
        run_etypes(args, report_to=report_to)
