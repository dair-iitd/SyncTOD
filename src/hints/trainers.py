from collections import Counter
import json
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from transformers import Trainer

try:
    import wandb
except:
    wandb = None

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ClosureTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_evaluation(self, dataset):
        local_rank = self.args.local_rank
        world_size = max(self.args.world_size, 1)

        model = self._wrap_model(self.model, training=False)
        model.eval()

        if type(model) == DataParallel:
            model = model.module

        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            sampler=sampler,
        )
        model.eval()
        if world_size > 1:
            sampler.set_epoch(0)
        loader = tqdm(dataloader, desc='Evaluation...') if local_rank in [-1, 0] else dataloader
        logits = []
        losses = []
        for inputs in loader:
            batch = dict([(k, v.to(self.model.device)) for k, v in inputs.items()])
            with torch.no_grad():
                ret = model(**batch)
                loss, tlogits = ret.loss, ret.logits
                tlogits = tlogits.cpu().numpy()
                tloss = loss.item()
            logits.extend(tlogits)
            losses.append(tloss)
        model.train()

        if world_size > 1:
            all_logits = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_logits, logits)
            all_losses = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_losses, logits)
        else:
            all_logits = [logits]
            all_losses = [losses]

        final_logits = []
        final_losses = []
        for ii in range(len(logits)):
            for resps in all_logits:
                final_logits.append(resps[ii])
            for losses in all_losses:
                final_losses.append(losses)

        final_logits = final_logits[:len(dataset)]
        final_loss = np.mean(final_losses)

        return np.array(final_logits), final_loss

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        return_logits=False,
        save_results=False,
        result_path=None,
    ):
        print(f'Running evaluation......{self.args.local_rank}')
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        logits, final_loss = self.run_evaluation(eval_dataset)
        tmetrics = {
            'loss_val': float(final_loss),
        }
        predictions = np.argmax(logits, axis=1)
        labels = [eval_dataset[ii]['target'] for ii in range(len(eval_dataset))]
        acc = accuracy_score(labels, predictions)
        tmetrics['accuracy'] = acc
        all_predictions = [int(x) for x in predictions]
        print('Predictions', Counter(predictions))

        metrics = dict()
        for key in tmetrics:
            metrics[f"{metric_key_prefix}_{key}"] = tmetrics[key]
        if wandb is not None and wandb.run is not None:
            wandb.log(metrics)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        print(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        if save_results:
            save_data = []
            for ii in range(len(eval_dataset)):
                obj = dict()
                obj['uuid'] = eval_dataset.raw_data[ii]['uuid']
                obj['turn_id'] = eval_dataset.raw_data[ii]['turn_id']
                obj['closure'] = all_predictions[ii]
                save_data.append(obj)

            tname = 'results.json'
            if result_path is not None:
                tname = result_path

            with open(tname, 'w') as fp:
                json.dump(save_data, fp, indent=2)

        return metrics


def get_indexed_entity_types(entity_types):
    cnts = Counter(entity_types).most_common()
    ret = set()
    for et, cnt in cnts:
        for jj in range(cnt):
            ret.add(f"{et}_{jj}")

    return ret


class Seq2SeqTrainer(Trainer):
    def __init__(self, cfg, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dedup_etypes = cfg['model'].get('dedup', True)

    def decode_responses(self, outputs):
        tokenizer = self.tokenizer

        preds = []
        responses = tokenizer.batch_decode(
            outputs, clean_up_tokenization_spaces=False
        )
        for resp in responses:
            preds.append(
                resp.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
            )

        return preds

    def run_evaluation(self, dataset):
        local_rank = self.args.local_rank
        world_size = max(self.args.world_size, 1)

        max_new_tokens = self.cfg['dev']['max_resp_length']
        # pred_end = self.vocab.eos_token_idx
        model = self._wrap_model(self.model, training=False)
        model.eval()

        if type(model) == DataParallel:
            model = model.module

        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            sampler=sampler,
        )
        model.eval()
        if world_size > 1:
            sampler.set_epoch(0)
        loader = tqdm(dataloader, desc='Evaluation...') if local_rank in [-1, 0] else dataloader
        responses = []
        for inputs in loader:
            batch = dict([(k, v.to(self.model.device)) for k, v in inputs.items()])
            if self.cfg['dev'].get('sample', False):
                outputs = model.generate(
                    **batch, use_cache=True,
                    # eos_token_id=pred_end,
                    # pad_token_id=pred_end,
                    max_new_tokens=max_new_tokens,
                    do_sample=True, min_length=1,
                    length_penalty=5.0,
                    temperature=self.cfg['dev'].get('temperature', 0.85),
                    top_k=self.cfg['dev'].get('top_k', 8),
                    top_p=self.cfg['dev'].get('top_p', 0.9),
                )
            else:
                with torch.no_grad():
                    outputs = model.generate(
                        **batch, use_cache=True,
                        length_penalty=1.0,
                        # eos_token_id=pred_end,
                        # pad_token_id=pred_end,
                        max_new_tokens=max_new_tokens, do_sample=False,
                        num_beams=self.cfg['dev'].get('num_beams', 1),
                        # early_stopping="never"
                    )

            outputs = outputs.to('cpu')
            responses.extend(self.decode_responses(outputs))

        model.train()
        if world_size > 1:
            all_responses = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_responses, responses)
        else:
            all_responses = [responses]

        final_responses = []
        for ii in range(len(responses)):
            for resps in all_responses:
                final_responses.append(resps[ii])
        final_responses = final_responses[:len(dataset)]

        return final_responses

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        save_results=False,
        result_path=None,
    ):
        print(f'Running evaluation......{self.args.local_rank}')
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        responses = self.run_evaluation(eval_dataset)

        tp, fn, fp = 0, 0, 0
        corr = 0
        for ii in range(len(responses)):
            gold = eval_dataset[ii]['output_seq']
            pred = responses[ii]

            if '[no entity]' in pred:
                pred = '[no entity]'

            if self.dedup_etypes:
                ents = [x.strip() for x in gold.split('|')]
                gold_types = set([x for x in ents if len(x) > 0])
                ents = [x.strip() for x in pred.split('|')]
                pred_types = set([x for x in ents if len(x) > 0])

            else:
                ents = [x.strip() for x in gold.split('|')]
                gold_types = get_indexed_entity_types(ents)
                ents = [x.strip() for x in pred.split('|')]
                pred_types = get_indexed_entity_types(ents)

            inter = gold_types.intersection(pred_types)
            ttp = len(inter)
            tfp = len(pred_types - gold_types)
            tfn = len(gold_types - pred_types)

            if tfp == 0 and tfn == 0:
                corr += 1

            tp += ttp
            fp += tfp
            fn += tfn

        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
        acc = corr / len(eval_dataset)

        metrics = dict()
        metrics[f"{metric_key_prefix}_f1"] = f1
        metrics[f"{metric_key_prefix}_prec"] = prec
        metrics[f"{metric_key_prefix}_rec"] = rec
        metrics[f"{metric_key_prefix}_acc"] = acc

        if wandb is not None and wandb.run is not None:
            wandb.log(metrics)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        print(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        if save_results and self.args.local_rank in [-1, 0]:
            ret = []
            for ii, resp in enumerate(responses):
                ret.append(dict())
                ret[-1]['uuid'] = eval_dataset.raw_data[ii]['uuid']
                ret[-1]['turn_id'] = eval_dataset.raw_data[ii]['turn_id']
                tetypes = resp.split('|')
                tetypes = [x.strip() for x in tetypes if '[no entity]' != x]
                ret[-1]['prediction'] = tetypes

            tname = 'results.json'
            if result_path is not None:
                tname = result_path
            with open(tname, 'w') as fp:
                json.dump(ret, fp, indent=2)

        return metrics
