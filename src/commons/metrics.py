#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# From https://github.com/PaddlePaddle/Knover/blob/develop/projects/Q-TOD/evaluate.py

"""Evaluate generated response."""

import argparse
from copy import deepcopy
import json
import re
import os
import numpy as np
import tempfile, subprocess

from .text_process import preprocess_text
from .batch_utils import process_batch_preds

def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    multi_bleu_path = os.path.abspath("./commons/multi-bleu.perl")

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

     # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["SMD", "MultiWOZ", "BiTOD"], required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--entity_file", type=str, required=True)
    
    args = parser.parse_args()
    return args


def compute_metrics(preds, refs, dataset, entity_file):
    bleu_res = moses_multi_bleu(preds, refs)
    # entity_metric = EntityMetric(args)
    entity_metric = EntityMetric(dataset, entity_file)
    entity_res, precision, recall = entity_metric.evaluate(preds, refs)
    results = {
        "bleu": bleu_res,
        "entity_f1": entity_res,
        "entity_precision": precision,
        "entity_recall": recall
    }

    return results


def evaluate(args):
    """Main evaluation function."""
    preds, refs = process_batch_preds(args.pred_path, args.data_path)
    preds = [preprocess_text(x) for x in preds]
    refs = [preprocess_text(x) for x in refs]

    assert len(preds) == len(refs), f"{len(preds)} != {len(refs)}"
    results = compute_metrics(preds, refs, args.dataset, args.entity_file)

    print(json.dumps(results, indent=2))


class EntityMetric(object):
    """Entity Metric for Response."""

    # def __init__(self, args):
    #     self.dataset = args.dataset
    #     self.entities = self._load_entities(args.entity_file)
    def __init__(self, dataset, entity_file):
        self.dataset = dataset
        self.entities = self._load_entities(entity_file)

    def evaluate(self, preds, refs):
        extracted_preds_entities = []
        extracted_refs_entities = []
        for pred, ref in zip(preds, refs):
            pred_entities = self._extract_entities(pred)
            ref_entities = self._extract_entities(ref)
            extracted_preds_entities.append(pred_entities)
            extracted_refs_entities.append(ref_entities)
        entity_f1, precision, recall = self._compute_entity_f1(extracted_preds_entities, extracted_refs_entities)
        return entity_f1, precision, recall

    def _load_entities(self, entities_file):
        with open(entities_file, "r") as fin:
            raw_entities = json.load(fin)
        entities = set()

        if self.dataset == "SMD":
            for slot, values in raw_entities.items():
                for val in values:
                    if slot == "poi":
                        entities.add(val["address"])
                        entities.add(val["poi"])
                        entities.add(val["type"])
                    elif slot == "distance":
                        entities.add(f"{val} miles")
                    elif slot == "temperature":
                        entities.add(f"{val}f")
                    else:
                        entities.add(val)

            # add missing entities
            missed_entities = ["yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist",
                               "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                               "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "jill", "jack", " hr "]
            for missed_entity in missed_entities:
                entities.add(missed_entity)
            # special handle of "hr"
            entities.remove("hr")

        else:
            for slot, values in raw_entities.items():
                for val in values:
                    if self.dataset == "MultiWOZ" and slot == "choice":
                        val = f"choice-{val}"
                    entities.add(val)

        processed_entities = []
        for val in sorted(entities):
            processed_entities.append(val.lower())
        processed_entities.sort(key=lambda x: len(x), reverse=True)
        return processed_entities

    def _extract_entities(self, response):
        def _is_sub_str(str_list, sub_str):
            for str_item in str_list:
                if sub_str in str_item:
                    return True
            return False

        response = f" {response} ".lower()
        extracted_entities = []

        if self.dataset == "SMD":
            # preprocess response
            for h in range(0, 13):
                response = response.replace(f"{h} am", f"{h}am")
                response = response.replace(f"{h} pm", f"{h}pm")
            for low_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for high_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    response = response.replace(f"{low_temp}-{high_temp}f", f"{low_temp}f-{high_temp}f")

        for entity in sorted(self.entities, key=lambda x: -len(x)):
            if self.dataset == "MultiWOZ":
                success_tag = False
                if entity.startswith("choice-"):
                    entity = entity[7:]
                    if entity == "many":
                        if entity in re.sub(r"(many (other types|food types|cuisines)|how many)", " ", response):
                            success_tag = True
                    elif entity == "all":
                        if re.search(r"all (of the|expensive|moderate|cheap)", response):
                            success_tag = True
                    elif entity == "to":
                        success_tag = False
                    else:
                        if re.search(f"(there are|there is|found|have about|have)( only|) {entity}", response):
                            success_tag = True
                elif entity == "centre":
                    if entity in response.replace("cambridge towninfo centre", " "):
                        success_tag = True
                elif entity == "free":
                    if re.search(r"free (parking|internet|wifi)", response):
                        success_tag = True
                elif entity in response or entity.lower() in response.lower():
                    success_tag = True

                if success_tag:
                    extracted_entities.append(entity)
                    response = response.replace(entity, " ")

            else:
                if entity in response and not _is_sub_str(extracted_entities, entity):
                    extracted_entities.append(entity)

        return extracted_entities

    def _compute_entity_f1(self, preds, refs):
        """Compute Entity-F1."""
        def _count(pred, ref):
            tp, fp, fn = 0, 0, 0
            if len(ref) != 0:
                for g in ref:
                    if g in pred:
                        tp += 1
                    else:
                        fn += 1
                for p in set(pred):
                    if p not in ref:
                        fp += 1
            return tp, fp, fn

        tp_all, fp_all, fn_all = 0, 0, 0
        for pred, ref in zip(preds, refs):
            tp, fp, fn = _count(pred, ref)
            tp_all += tp
            fp_all += fp
            fn_all += fn

        precision = tp_all / float(tp_all + fp_all) if (tp_all + fp_all) != 0 else 0
        recall = tp_all / float(tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        print('Total', tp_all + fn_all)
        return f1, precision, recall


class EntityTypeMetric(object):
    """Entity Metric for Response."""

    def __init__(self, dataset_name, entities_file):
        self.dataset = dataset_name
        self.entities = self._load_entities(entities_file)

    def evaluate(self, preds, refs):
        extracted_preds_entities = []
        extracted_refs_entities = []
        for pred, ref in zip(preds, refs):
            pred_entities = self.extract_entities(pred, return_types=True)
            ref_entities = self.extract_entities(ref, return_types=True)
            extracted_preds_entities.append(pred_entities)
            extracted_refs_entities.append(ref_entities)
        entity_f1 = self._compute_entity_f1(extracted_preds_entities, extracted_refs_entities)
        return entity_f1

    def _load_entities(self, entities_file):
        with open(entities_file, "r") as fin:
            raw_entities = json.load(fin)
        entities = set()

        if self.dataset == "SMD":
            for slot, values in raw_entities.items():
                for val in values:
                    if slot == "poi":
                        entities.add(('address', val["address"]))
                        entities.add(('poi', val["poi"]))
                        entities.add(('type', val["type"]))
                    elif slot == "distance":
                        entities.add(('distance', f"{val} miles"))
                    elif slot == "temperature":
                        entities.add(('temperature', f"{val}f"))
                    else:
                        entities.add((slot, val))

            # add missing entities
            missed_entities = [
                "yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist",
                               "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                               "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "jill", "jack", " hr "
            ]
            missed_entities = [
                ('event', 'yoga'), ('event', 'tennis'), ('event', 'swimming'),
                ('event', 'football'), ('event', ' lab '), ('event', 'doctor'),
                ('event', 'optometrist'), ('event', 'dentist'), ('date', '1st'),
                ('date', '2nd'), ('date', '3rd'), ('date', '4th'),
                ('date', '5th'), ('date', '6th'), ('date', '7th'),
                ('date', '8th'), ('date', '9th'), ('date', '10th'),
                ('date', '11th'), ('date', '12th'), ('date', '13th'),
                ('date', '14th'), ('date', '15th'), ('date', '16th'),
                ('date', '17th'), ('date', '18th'), ('date', '19th'),
                ('date', '20th'), ('party', 'jill'), ('party', 'jack'),
                ('party', ' hr ')
            ]

            for missed_entity in missed_entities:
                entities.add(missed_entity)
            # special handle of "hr"
            entities.remove(("party", "hr"))

        else:
            for slot, values in raw_entities.items():
                for val in values:
                    if self.dataset == "MultiWOZ" and slot == "choice":
                        val = f"choice-{val}"
                    entities.add((slot, val))

        processed_entities = []
        for etype, val in sorted(entities, key=lambda x: x[0]):
            processed_entities.append((etype, val.lower()))
        processed_entities.sort(key=lambda x: len(x[1]), reverse=True)
        return processed_entities

    def extract_entities(self, response, return_types=False):
        def _is_sub_str(str_list, sub_str):
            for str_item in str_list:
                if sub_str in str_item:
                    return True
            return False

        response = f" {response} ".lower()
        extracted_entities = []

        if self.dataset == "SMD":
            # preprocess response
            for h in range(0, 13):
                response = response.replace(f"{h} am", f"{h}am")
                response = response.replace(f"{h} pm", f"{h}pm")
            for low_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for high_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    response = response.replace(f"{low_temp}-{high_temp}f", f"{low_temp}f-{high_temp}f")

        for etype, entity in sorted(self.entities, key=lambda x: -len(x[1])):
            if self.dataset == "MultiWOZ":
                # print('WARNING.... Not tested for MultiWOZ')
                success_tag = False
                if entity.startswith("choice-"):
                    entity = entity[7:]
                    if entity == "many":
                        if entity in re.sub(r"(many (other types|food types|cuisines)|how many)", " ", response):
                            success_tag = True
                    elif entity == "all":
                        if re.search(r"all (of the|expensive|moderate|cheap)", response):
                            success_tag = True
                    elif entity == "to":
                        success_tag = False
                    else:
                        if re.search(f"(there are|there is|found|have about|have)( only|) {entity}", response):
                            success_tag = True
                elif entity == "centre":
                    if entity in response.replace("cambridge towninfo centre", " "):
                        success_tag = True
                elif entity == "free":
                    if re.search(r"free (parking|internet|wifi)", response):
                        success_tag = True
                elif entity in response or entity.lower() in response.lower():
                    success_tag = True

                if success_tag:
                    extracted_entities.append((etype, entity))
                    response = response.replace(entity, " ")

            else:
                if entity in response and not _is_sub_str(extracted_entities, entity):
                    extracted_entities.append((etype, entity))

        if return_types:
            return extracted_entities

        return [x[1] for x in extracted_entities]

    def _compute_entity_f1(self, preds, refs):
        """Compute Entity-F1."""
        def _count(pred, ref):
            tp, fp, fn = 0, 0, 0
            if len(ref) != 0:
                for g in ref:
                    if g in pred:
                        tp += 1
                    else:
                        fn += 1
                for p in set(pred):
                    if p not in ref:
                        fp += 1
            return tp, fp, fn

        tp_all, fp_all, fn_all = 0, 0, 0
        for pred, ref in zip(preds, refs):
            tp, fp, fn = _count(pred, ref)
            tp_all += tp
            fp_all += fp
            fn_all += fn

        precision = tp_all / float(tp_all + fp_all) if (tp_all + fp_all) != 0 else 0
        recall = tp_all / float(tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        
        return f1


class EntityMetricBiTOD(object):
    """Entity Metric for Response."""

    def __init__(self, dataset_name, entities_file):
        self.dataset = dataset_name
        self.entities = self._load_entities(entities_file)

    def evaluate(self, preds, gold_entities):
        extracted_preds_entities = [self.extract_entities(x) for x in preds]
        extracted_refs_entities = gold_entities
        entity_f1 = self._compute_entity_f1(extracted_preds_entities, extracted_refs_entities)
        return entity_f1

    def _load_entities(self, entities_file):
        with open(entities_file, "r") as fin:
            raw_entities = json.load(fin)
        entities = set()

        if self.dataset == "SMD":
            for slot, values in raw_entities.items():
                for val in values:
                    if slot == "poi":
                        entities.add(('address', val["address"]))
                        entities.add(('poi', val["poi"]))
                        entities.add(('type', val["type"]))
                    elif slot == "distance":
                        entities.add(('distance', f"{val} miles"))
                    elif slot == "temperature":
                        entities.add(('temperature', f"{val}f"))
                    else:
                        entities.add((slot, val))

            # add missing entities
            missed_entities = [
                "yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist",
                               "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                               "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "jill", "jack", " hr "
            ]
            missed_entities = [
                ('event', 'yoga'), ('event', 'tennis'), ('event', 'swimming'),
                ('event', 'football'), ('event', ' lab '), ('event', 'doctor'),
                ('event', 'optometrist'), ('event', 'dentist'), ('date', '1st'),
                ('date', '2nd'), ('date', '3rd'), ('date', '4th'),
                ('date', '5th'), ('date', '6th'), ('date', '7th'),
                ('date', '8th'), ('date', '9th'), ('date', '10th'),
                ('date', '11th'), ('date', '12th'), ('date', '13th'),
                ('date', '14th'), ('date', '15th'), ('date', '16th'),
                ('date', '17th'), ('date', '18th'), ('date', '19th'),
                ('date', '20th'), ('party', 'jill'), ('party', 'jack'),
                ('party', ' hr ')
            ]

            for missed_entity in missed_entities:
                entities.add(missed_entity)
            # special handle of "hr"
            entities.remove(("party", "hr"))

        else:
            for slot, values in raw_entities.items():
                for val in values:
                    if self.dataset == "MultiWOZ" and slot == "choice":
                        val = f"choice-{val}"
                    entities.add((slot, val))

        processed_entities = []
        for etype, val in sorted(entities, key=lambda x: x[0]):
            processed_entities.append((etype, val.lower()))
        processed_entities.sort(key=lambda x: len(x[1]), reverse=True)

        if self.dataset == 'BiTOD':
            etarr = [
                'rating', 'number_of_rooms', 'number_of_nights',
                'stars', 'time', 'start_day', 'day', 'number_of_people'
            ]
            part1 = [x for x in processed_entities if x[0] not in etarr]
            part2 = [x for x in processed_entities if x[0] in etarr]
            new_part2 = []
            for et in etarr:
                new_part2.extend(sorted([x for x in part2 if x[0] == et], key=lambda x: -len(x[1])))
            processed_entities = part1 + new_part2
            # rating, number_of_rooms, number_of_nights, stars, time, start_day, day

        return processed_entities

    def extract_entities(self, response, return_types=False):
        def _is_sub_str(str_list, sub_str):
            for str_item in str_list:
                if sub_str in str_item:
                    return True
            return False

        if self.dataset == 'BiTOD':
            response = response.replace('_', ' ')
            response = ' '.join(response.split())

        response = f" {response} ".lower()
        response = response.replace('may i', 'MAY I')
        extracted_entities = []

        if self.dataset == "SMD":
            # preprocess response
            for h in range(0, 13):
                response = response.replace(f"{h} am", f"{h}am")
                response = response.replace(f"{h} pm", f"{h}pm")
            for low_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for high_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    response = response.replace(f"{low_temp}-{high_temp}f", f"{low_temp}f-{high_temp}f")

        i_data = self.entities
        if self.dataset == 'BiTOD':
            i_data = self.entities
        else:
            i_data = sorted(self.entities, key=lambda x: -len(x[1]))
        # for etype, entity in sorted(self.entities, key=lambda x: -len(x[1])):
        for etype, entity in i_data:
            if self.dataset == "MultiWOZ":
                # print('WARNING.... Not tested for MultiWOZ')
                success_tag = False
                if entity.startswith("choice-"):
                    entity = entity[7:]
                    if entity == "many":
                        if entity in re.sub(r"(many (other types|food types|cuisines)|how many)", " ", response):
                            success_tag = True
                    elif entity == "all":
                        if re.search(r"all (of the|expensive|moderate|cheap)", response):
                            success_tag = True
                    elif entity == "to":
                        success_tag = False
                    else:
                        if re.search(f"(there are|there is|found|have about|have)( only|) {entity}", response):
                            success_tag = True
                elif entity == "centre":
                    if entity in response.replace("cambridge towninfo centre", " "):
                        success_tag = True
                elif entity == "free":
                    if re.search(r"free (parking|internet|wifi)", response):
                        success_tag = True
                elif entity in response or entity.lower() in response.lower():
                    success_tag = True

                if success_tag:
                    extracted_entities.append((etype, entity))
                    response = response.replace(entity, " ")

            elif self.dataset == 'BiTOD':
                # rating, number_of_rooms, number_of_nights, stars, time, start_day, day
                patterns = []
                if etype == 'rating':
                    patterns = [f"rating is {entity}", f"rated {entity}", f"rating of {entity}"]
                elif etype == 'number_of_rooms':
                    patterns = [f"{entity} room", f"{entity} rooms", f"room {entity}", f"rooms {entity}"]
                elif etype == 'number_of_nights':
                    patterns = [f"{entity} night", f"{entity} day"]
                elif etype == 'stars':
                    patterns = [f"{entity} star", f"star rating is {entity}", f"stars rating is {entity}"]
                elif etype == 'time':
                    if len(entity) <= 2:
                        patterns = [f"{entity} am", f"{entity} pm"]
                    else:
                        patterns = [deepcopy(entity)]
                else:
                    patterns = [deepcopy(entity)]

                for pattern in patterns:
                    # tentity = deepcopy(entity)
                    # tentity = tentity.replace('_', ' ')
                    # pattern = f" {tentity} "

                    pattern = pattern.replace('_', ' ')
                    pattern = f" {pattern} "
                    if pattern in response and not _is_sub_str([x[1] for x in extracted_entities], entity):
                        extracted_entities.append((etype, entity))
                        break

            else:
                if entity in response and not _is_sub_str([x[1] for x in extracted_entities], entity):
                    extracted_entities.append((etype, entity))

        if return_types:
            return extracted_entities

        return [x[1] for x in extracted_entities]

    def _compute_entity_f1(self, preds, refs):
        """Compute Entity-F1."""
        def _count(pred, ref):
            tp, fp, fn = 0, 0, 0
            if len(ref) != 0:
                for g in ref:
                    if g in pred:
                        tp += 1
                    else:
                        fn += 1
                for p in set(pred):
                    if p not in ref:
                        fp += 1
            return tp, fp, fn

        tp_all, fp_all, fn_all = 0, 0, 0
        for pred, ref in zip(preds, refs):
            tp, fp, fn = _count(pred, ref)
            tp_all += tp
            fp_all += fp
            fn_all += fn

        precision = tp_all / float(tp_all + fp_all) if (tp_all + fp_all) != 0 else 0
        recall = tp_all / float(tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0

        return f1, precision, recall


def evaluate_bitod(args):
    """Main evaluation function."""
    preds, refs, gold_entities = process_batch_preds(args.pred_path, args.data_path, include_entities=True)
    gold_entities = [[x for x in y if x != 'a'] for y in gold_entities]

    preds = [x.replace('_', ' ') for x in preds]
    preds = [preprocess_text(x, ignore_puncs=['#']) for x in preds]
    refs = [x.replace('_', ' ') for x in refs]
    refs = [preprocess_text(x, ignore_puncs=['#']) for x in refs]

    assert len(preds) == len(refs), f"{len(preds)} != {len(refs)}"

    bleu_res = moses_multi_bleu(preds, refs)
    entity_metric = EntityMetricBiTOD('BiTOD', args.entity_file)
    entity_res, precision, recall = entity_metric.evaluate(preds, gold_entities)
    results = {
        "bleu": bleu_res,
        "entity_f1": entity_res,
        "entity_precision": precision,
        "entity_recall": recall
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    args = setup_args()
    if args.dataset == 'BiTOD':
        evaluate_bitod(args)
    else:
        evaluate(args)
