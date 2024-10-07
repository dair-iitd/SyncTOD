import json
import argparse
from copy import deepcopy
import numpy as np
import pandas as pd
import tiktoken

from tqdm import trange
import re
from .retriever import SentenceBertRetriever, KBRetriever, retrieve_kb

from commons.metrics import EntityTypeMetric
from commons.text_process import preprocess_text
MULTIWOZ_PROMPT_VERSION = 'V6'
np.random.seed(42)
THRESH = 3696


def read_cli():
    parser = argparse.ArgumentParser(description='Generate model.')
    parser.add_argument(
        "-ds",
        "--dataset",
        help="Dataset",
        required=True,
        type=str,
        choices=['MultiWOZ', 'SMD', 'BiTOD']
    )
    parser.add_argument(
        "-idx",
        "--index_path",
        help="Path to index",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-src",
        "--src_path",
        help="Path to source",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tar",
        "--tar_path",
        help="Path to target",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-mode",
        "--mode",
        help="Mode",
        required=True,
        type=str,
        choices=['oracle', 'manual', 'predicted']
    )
    parser.add_argument(
        "-mkey",
        "--manual_hint_key",
        help="manual_hint_key",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "-hr",
        "--hint_ranking",
        help="Use Hint ranking",
        required=False,
        type=str,
        default='False',
        choices=['True', 'False']
    )
    parser.add_argument(
        "-bs_tar",
        "--booking_status_index_tar",
        help="Use booking status index tar",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "-eg",
        "--entity_generation",
        help="Entity Generation",
        required=False,
        type=str,
        default='False',
        choices=['True', 'False']
    )
    parser.add_argument(
        "-ef",
        "--entity_file",
        help="Entity File",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "-ablation",
        "--ablation",
        help="Ablation",
        required=False,
        type=str,
        default='none'
    )
    parser.add_argument(
        "-noret",
        "--noret",
        help="No retrieval",
        required=False,
        type=str,
        default='False'
    )

    args = vars(parser.parse_args())
    args['hint_ranking'] = args['hint_ranking'] == 'True'
    args['entity_generation'] = args['entity_generation'] == 'True'
    args['noret'] = args['noret'] == 'True'

    if args['dataset'] != 'MultiWOZ':
        print(f'Disabling booking status for {args["dataset"]}')
        args['booking_status'] = False

    return args


def multiwoz_prompt(orig_sample, exemplars, booking_status=False, etype_metric=None, ablation='none'):
    all_cols = ['name', 'address', 'phone', 'food', 'area', 'postcode', 'price range', 'type', 'reference number', 'stars', 'choice']

    instructions = """Henceforth, assume that you are a customer support expert. I will give you an incomplete dialog between a user and a customer service representative. As an expert, you must suggest the most appropriate follow-up response to the dialog. Ensure you also include correct information (entities) from the given database. Entities can be of the following types - 
1. name - name of a place (restaurant, hotel or attraction)
2. address - address of the place
3. phone - phone number of the place
4. food - the type of food a restaurant serves
5. area - a region of the city, e.g. centre, north, south, east, west
6. postcode - postcode of the place
7. price range - price range of the place, e.g. cheap, moderate, expensive
8. type - the type of a place, e.g. restaurant, hotel, gusesthouse, attraction
9. reference number - reference code for booking, e.g. 542j9wog
10. stars - star rating of the hotel, e.g. 3 stars
11. choice - number of available choices that match user's requirements, e.g. many, few, several, 10

"""

    if ablation == 'all_rules':
        instructions += """Here are the examples -"""
    else:
        instructions += """As an expert, you are very strict about following rules. Make sure that the follow-up response you write follows all the given rules. Here are the examples -"""

    prompt = [instructions]
    prompt.append('')
    exemplars = exemplars + [orig_sample]

    for ii, tee in enumerate(exemplars):
        sample = deepcopy(tee)
        prompt.append(f'[example {ii + 1}]')
        prompt.append(f'[database {ii + 1}]')

        cols = []
        for kk in sample['kb']:
            cols.extend(list(kk))
        cols = set(cols)
        cols = [cc for cc in all_cols if cc in cols]

        final_cols = []
        for col in cols:
            flag = False
            for row in sample['kb']:
                vv = row.get(col, None)
                if vv is not None and vv != '-':
                    flag = True

            if flag:
                final_cols.append(col)
        cols = final_cols

        tmp_kb = []
        for row in sample['kb']:
            tmp_row = dict()
            for c in cols:
                tmp_row[c] = row.get(c, '-')
            tmp_kb.append(tmp_row)

        index_col = 'name'
        kb_df = pd.DataFrame(tmp_kb)
        indices = kb_df[index_col].tolist()
        kb_df[index_col] = dedup_index(indices)
        kb_df = kb_df.set_index(index_col)
        tmp_text = kb_df.to_json(orient='index', indent=2)
        prompt.append(tmp_text)
        prompt.append('')

        if ablation not in ['all_rules']:
            prompt.append(f'[rules {ii + 1}]')
            hints = sample['hints']
            prompt.append(f'The response must be {hints["word_count"]} words or shorter.')
            if hints['closure']:
                prompt.append('The response must close the dialog.')
            else:
                prompt.append('The response must not close the dialog.')

            tetypes = sorted(hints['entity_types'])
            tetypes = [cc for cc in tetypes if cc in all_cols]
            ntetypes = [cc for cc in all_cols if (cc not in tetypes) and (cc not in ['internet', 'parking'])]
            if len(hints['entity_types']) > 0:
                prompt.append(f"The response must only include entities of type - {', '.join(tetypes)}.")
            prompt.append(f"The response must not include any entities of type - {', '.join(ntetypes)}.")

            prompt.append('')

        prompt.append(f'[dialog history {ii + 1}]')
        for jj, uttr in enumerate(sample['context']):
            spk = 'user' if jj % 2 == 0 else 'assistant'
            prompt.append(f"{spk}: {uttr}")
        prompt.append('')

        prompt.append(f"[follow-up response {ii + 1}]")
        if booking_status:
            bstatus = sample['booking_status']
            if bstatus == 'successful':
                prompt.append("Note that user's booking is successful.")
            if bstatus == 'failed':
                prompt.append("Note that user's booking is unsuccessful.")

        # if ablation == 'all_rules':
        if ablation in ['all_rules']:
            # prompt.append(f"As an expert, you must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}] and write the response.")
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}] and write the response.")
            if ii == len(exemplars) - 1:
                prompt.append(f"assistant:")
            else:
                prompt.append(f"assistant: {sample['output']}")
                prompt.append("")
        elif etype_metric is None:
            prompt.append(f"As an expert, you must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}], follow all the [rules {ii + 1}] and write the response.")
            if ii == len(exemplars) - 1:
                prompt.append(f"assistant:")
            else:
                prompt.append(f"assistant: {sample['output']}")
                prompt.append("")
        else:
            prompt.append(f"Let's think step-by-step.")
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}], follow all the [rules {ii + 1}] and write the response.")
            prompt.append(f"I will include entities of type {tetypes} in my response.")

            if ii == len(exemplars) - 1:
                prompt.append(f"I will include these entities -")
            else:
                tmp = etype_metric.extract_entities(
                    preprocess_text(sample['output']), return_types=True
                )
                for hh in range(len(tmp)):
                    if tmp[hh][0] == 'ref':
                        tmp[hh] = ('reference number', tmp[hh][1])
                    elif tmp[hh][0] == 'pricerange':
                        tmp[hh] = ('price range', tmp[hh][1])

                tents = []
                for cc in sorted(set(tetypes)):
                    for ee in tmp:
                        if ee[0] == cc:
                            tents.append(ee)

                prompt.append(f"I will include these entities - {tents}")
                prompt.append(f"assistant: {sample['output']}")
                prompt.append("")

    prompt = '\n'.join(prompt)

    return prompt


def smd_prompt_without_kb(orig_sample, exemplars, etype_metric=None, ablation='none'):
    all_cols = [
        # navigate ['poi', 'address', 'poi_type', 'traffic_info', 'distance']
        'poi', 'address', 'poi type', 'traffic info', 'distance',
        # schedule
        'event', 'date', 'time', 'party', 'agenda', 'room',
        # weather
        'location', 'weather attribute', 'temperature',
        # common
        'weekly time'
    ]

    instructions = """Henceforth, assume that you are an expert in in-car infotainment. I will give you an incomplete dialog between a user and an in-car infotainment system. As an expert, you must suggest the most appropriate follow-up response to the dialog. Ensure you also include correct information (entities) from the given dialog history. Entities can be of the following types - 
1. poi - name of a point of interest, e.g., home, starbucks, pizza chicago, etc.
2. address - address of a poi, e.g, 783 arcadia pl.
3. poi type - the type of a poi, e.g., tea or coffee place, hospital, shopping center, etc.
4. traffic info - traffic status on the way to a poi, e.g., heavy traffic, no traffic, road block nearby, etc. 
5. distance - distance of a poi from the user's current location, e.g., 2 miles, 4 miles, etc.
6. event - an event in the user's calendar
7. date - date in a month like the 1st or the 4th or day of a week like monday, wednesday.
8. time - the time on which an event is scheduled
9. party - party attending an event, e.g., tom, boss, brother, executive team, etc.
10. agenda - agenda associated with an event, e.g., discuss dress code, go over budget, etc.
11. room - meeting place of an event, e.g., conference room 100, etc.
12. location - a location for which the user may request the weather information, e.g, boston, los angeles, etc.
13. weather attribute - weather description in a location, e.g., cloudy, warm, hot, overcast etc.
14. temperature - the in a location, e.g., 60f, 100f, etc.
15. weekly time - temporal indicators like today, tomorrow, next week etc.

"""
    if ablation == 'all_rules':
        instructions += """Here are the examples -"""
    else:
        instructions += """As an expert, you are very strict about following rules. Make sure that the follow-up response you write follows all the given rules. Here are the examples -"""

    prompt = [instructions]
    prompt.append('')
    exemplars = exemplars + [orig_sample]

    for ii, tee in enumerate(exemplars):
        sample = deepcopy(tee)
        prompt.append(f'[example {ii + 1}]')
        if ablation not in ['all_rules']:
            prompt.append(f'[rules {ii + 1}]')
            hints = sample['hints']
            prompt.append(f'The response must be {hints["word_count"]} words or shorter.')
            if hints['closure']:
                prompt.append('The response must close the dialog.')
            else:
                prompt.append('The response must not close the dialog.')

            # tetypes = sorted(set(hints['entity_types']))
            tetypes = sorted(hints['entity_types'])
            tetypes = [cc for cc in tetypes if cc in all_cols]
            # tetypes = [cc for cc in all_cols if cc in tetypes]
            ntetypes = [cc for cc in all_cols if (cc not in tetypes) and (cc not in ['internet', 'parking'])]
            if len(hints['entity_types']) > 0:
                prompt.append(f"The response must only include entities of type - {', '.join(tetypes)}.")
            prompt.append(f"The response must not include any entities of type - {', '.join(ntetypes)}.")

            prompt.append('')

        prompt.append(f'[dialog history {ii + 1}]')
        for jj, uttr in enumerate(sample['context']):
            spk = 'user' if jj % 2 == 0 else 'system'
            prompt.append(f"{spk}: {uttr}")
        prompt.append('')

        prompt.append(f"[follow-up response {ii + 1}]")
        if ablation == 'all_rules':
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}] and write the response.")
            if ii == len(exemplars) - 1:
                prompt.append(f"system:")
            else:
                prompt.append(f"system: {sample['output']}")
                prompt.append("")

        elif etype_metric is None:
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], follow all the [rules {ii + 1}] and write the response.")
            if ii == len(exemplars) - 1:
                prompt.append(f"system:")
            else:
                prompt.append(f"system: {sample['output']}")
                prompt.append("")
        else:
            prompt.append(f"Let's think step-by-step.")
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], follow all the [rules {ii + 1}] and write the response.")
            prompt.append(f"I will include entities of type {tetypes} in my response.")

            if ii == len(exemplars) - 1:
                prompt.append(f"I will include these entities -")
            else:
                tmp = etype_metric.extract_entities(
                    preprocess_text(sample['output']), return_types=True
                )
                for hh in range(len(tmp)):
                    if tmp[hh][0] == 'poi_type':
                        tmp[hh] = ('poi type', tmp[hh][1])
                    elif tmp[hh][0] == 'traffic_info':
                        tmp[hh] = ('traffic info', tmp[hh][1])
                    elif tmp[hh][0] == 'weekly_time':
                        tmp[hh] = ('weekly time', tmp[hh][1])
                    elif tmp[hh][0] == 'weather_attribute':
                        tmp[hh] = ('weather attribute', tmp[hh][1])
                    
                tents = []
                for cc in sorted(set(tetypes)):
                    for ee in tmp:
                        if ee[0] == cc:
                            tents.append(ee)

                prompt.append(f"I will include these entities - {tents}")
                prompt.append(f"system: {sample['output']}")
                prompt.append("")

    prompt = '\n'.join(prompt)

    return prompt


def dedup_index(indices):
    val2cnt = dict()
    ret = []
    for val in indices:
        tmp = val2cnt.get(val, 0)
        if tmp == 0:
            # This is the first time we see this entry
            ret.append(val)
            val2cnt[val] = tmp + 1
        else:
            ret.append(f"{val} {tmp + 1}")
            val2cnt[val] = tmp + 1

    return ret


def smd_prompt(orig_sample, exemplars, etype_metric=None, ablation='none'):
    use_kb = len(orig_sample['kb']) > 0
    for ee in exemplars:
        if len(ee['kb']) > 0:
            use_kb = True

    if not use_kb:
        return smd_prompt_without_kb(orig_sample, exemplars, etype_metric, ablation)

    all_cols = [
        # navigate
        'poi', 'address', 'poi type', 'traffic info', 'distance',
        # schedule
        'event', 'date', 'time', 'party', 'agenda', 'room',
        # weather
        'location', 'weather attribute', 'temperature',
        # common
        'weekly time'
    ]
    add_cols = [
        'temperature low', 'temperature high',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'today'
    ]

    instructions = """Henceforth, assume that you are an expert in in-car infotainment. I will give you an incomplete dialog between a user and an in-car infotainment system. As an expert, you must suggest the most appropriate follow-up response to the dialog. Ensure you also include correct information (entities) from the given database. Entities can be of the following types - 
1. poi - name of a point of interest, e.g., home, starbucks, pizza chicago, etc.
2. address - address of a poi, e.g, 783 arcadia pl.
3. poi type - the type of a poi, e.g., tea or coffee place, hospital, shopping center, etc.
4. traffic info - traffic status on the way to a poi, e.g., heavy traffic, no traffic, road block nearby, etc. 
5. distance - distance of a poi from the user's current location, e.g., 2 miles, 4 miles, etc.
6. event - an event in the user's calendar
7. date - date in a month like the 1st or the 4th or day of a week like monday, wednesday.
8. time - the time on which an event is scheduled
9. party - party attending an event, e.g., tom, boss, brother, executive team, etc.
10. agenda - agenda associated with an event, e.g., discuss dress code, go over budget, etc.
11. room - meeting place of an event, e.g., conference room 100, etc.
12. location - a location for which the user may request the weather information, e.g, boston, los angeles, etc.
13. weather attribute - weather description in a location, e.g., cloudy, warm, hot, overcast etc.
14. temperature - the in a location, e.g., 60f, 100f, etc.
15. weekly time - temporal indicators like today, tomorrow, next week etc.

"""
    if ablation == 'all_rules':
        instructions += """Here are the examples -"""
    else:
        instructions += """As an expert, you are very strict about following rules. Make sure that the follow-up response you write follows all the given rules. Here are the examples -"""

    prompt = [instructions]
    prompt.append('')
    exemplars = exemplars + [orig_sample]

    for ii, tee in enumerate(exemplars):
        sample = deepcopy(tee)
        prompt.append(f'[example {ii + 1}]')
        prompt.append(f'[database {ii + 1}]')

        sample_kb = sample['kb']
        today = sample.get('today', None)

        cols = []
        for kk in sample_kb:
            cols.extend(list(kk))
        cols = set(cols)
        cols = [cc for cc in all_cols + add_cols if cc in cols]

        final_cols = []
        for col in cols:
            flag = False
            for row in sample_kb:
                vv = row.get(col, None)
                if vv is not None and vv != '-':
                    flag = True

            if flag:
                final_cols.append(col)
        cols = final_cols

        tmp_kb = []
        for row in sample_kb:
            tmp_row = dict()
            for c in cols:
                tmp_row[c] = row.get(c, '-')
            tmp_kb.append(tmp_row)

        assert sample['type'] in ['weather', 'navigate', 'schedule']
        if sample['type'] == 'weather':
            index_col = 'location'
        elif sample['type'] == 'navigate':
            index_col = 'poi'
        elif sample['type'] == 'schedule':
            index_col = 'event'

        if len(tmp_kb) > 0:
            kb_df = pd.DataFrame(tmp_kb)
            indices = kb_df[index_col].tolist()
            kb_df[index_col] = dedup_index(indices)
            kb_df = kb_df.set_index(index_col)

            tmp_text = kb_df.to_json(orient='index', indent=2)
        else:
            tmp_text = '{}'

        prompt.append(tmp_text)
        if today is not None:
            prompt.append(f"Note: You can assume that today is {today}.")
        prompt.append('')

        if ablation not in ['all_rules']:
            prompt.append(f'[rules {ii + 1}]')
            hints = sample['hints']
            prompt.append(f'The response must be {hints["word_count"]} words or shorter.')
            if hints['closure']:
                prompt.append('The response must close the dialog.')
            else:
                prompt.append('The response must not close the dialog.')

            tetypes = sorted(hints['entity_types'])
            tetypes = [cc for cc in tetypes if cc in all_cols]
            ntetypes = [cc for cc in all_cols if (cc not in tetypes) and (cc not in ['internet', 'parking'])]
            if len(hints['entity_types']) > 0:
                prompt.append(f"The response must only include entities of type - {', '.join(tetypes)}.")
            prompt.append(f"The response must not include any entities of type - {', '.join(ntetypes)}.")

            prompt.append('')

        prompt.append(f'[dialog history {ii + 1}]')
        for jj, uttr in enumerate(sample['context']):
            spk = 'user' if jj % 2 == 0 else 'system'
            prompt.append(f"{spk}: {uttr}")
        prompt.append('')
        prompt.append(f"[follow-up response {ii + 1}]")

        if ablation == 'all_rules':
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}] and write the response.")
            if ii == len(exemplars) - 1:
                prompt.append(f"system:")
            else:
                prompt.append(f"system: {sample['output']}")
                prompt.append("")

        elif etype_metric is None:
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}], follow all the [rules {ii + 1}] and write the response.")
            if ii == len(exemplars) - 1:
                prompt.append(f"system:")
            else:
                prompt.append(f"system: {sample['output']}")
                prompt.append("")
        else:
            prompt.append(f"Let's think step-by-step.")
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}], follow all the [rules {ii + 1}] and write the response.")
            prompt.append(f"I will include entities of type {tetypes} in my response.")

            if ii == len(exemplars) - 1:
                prompt.append(f"I will include these entities -")
            else:
                tmp = etype_metric.extract_entities(
                    preprocess_text(sample['output']), return_types=True
                )
                for hh in range(len(tmp)):
                    if tmp[hh][0] == 'poi_type':
                        tmp[hh] = ('poi type', tmp[hh][1])
                    elif tmp[hh][0] == 'traffic_info':
                        tmp[hh] = ('traffic info', tmp[hh][1])
                    elif tmp[hh][0] == 'weekly_time':
                        tmp[hh] = ('weekly time', tmp[hh][1])
                    elif tmp[hh][0] == 'weather_attribute':
                        tmp[hh] = ('weather attribute', tmp[hh][1])
                    
                tents = []
                for cc in sorted(set(tetypes)):
                    for ee in tmp:
                        if ee[0] == cc:
                            tents.append(ee)

                tents = sorted(tents)
                prompt.append(f"I will include these entities - {tents}")
                prompt.append(f"system: {sample['output']}")
                prompt.append("")

    prompt = '\n'.join(prompt)

    return prompt


def fix_multiwoz_typos(sample):
    for row in sample['kb']:
        if 'interent' in row:
            row['internet'] = row['interent']
            del row['interent']
        if 'ref' in row:
            row['reference number'] = row['ref']
            del row['ref']
        if 'pricerange' in row:
            row['price range'] = row['pricerange']
            del row['pricerange']
            
    for ii in range(len(sample['hints']['entity_types'])):
        if sample['hints']['entity_types'][ii] == 'ref':
            sample['hints']['entity_types'][ii] = 'reference number'
        elif sample['hints']['entity_types'][ii] == 'pricerange':
            sample['hints']['entity_types'][ii] = 'price range'

    for jj in range(len(sample['kb'])):
        for k in sample['kb'][jj]:
            sample['kb'][jj][k] = re.sub('_', ' ', sample['kb'][jj][k])


def fix_booking_status(sample):
    bstatus = sample['booking_status']
    if bstatus == 'dont_care':
        return

    if bstatus == 'successful' and 'ref' in sample['hints']:
        sample['hints'].remove('ref')


def fix_smd_typos(sample):
    for row in sample['kb']:
        if 'poi_type' in row:
            row['poi type'] = row['poi_type']
            del row['poi_type']
        if 'traffic_info' in row:
            row['traffic info'] = row['traffic_info']
            del row['traffic_info']

    for ii in range(len(sample['hints']['entity_types'])):
        if sample['hints']['entity_types'][ii] == 'poi_type':
            sample['hints']['entity_types'][ii] = 'poi type'
        elif sample['hints']['entity_types'][ii] == 'traffic_info':
            sample['hints']['entity_types'][ii] = 'traffic info'
        elif sample['hints']['entity_types'][ii] == 'weekly_time':
            sample['hints']['entity_types'][ii] = 'weekly time'
        elif sample['hints']['entity_types'][ii] == 'weather_attribute':
            sample['hints']['entity_types'][ii] = 'weather attribute'

    for jj in range(len(sample['kb'])):
        for k in sample['kb'][jj]:
            sample['kb'][jj][k] = re.sub('_', ' ', sample['kb'][jj][k])


def print_estimated_costs(data, gpt35_enc):
    total_inp_characters = 0
    total_output_characters = 0
    total_inp_tokens = 0
    total_output_tokens = 0
    for ee in data:
        total_inp_characters += len(ee['prompt'])
        total_output_characters += 80
        total_inp_tokens += len(gpt35_enc.encode(ee['prompt']))
        total_output_tokens += 120

    input_cost = (total_inp_characters / 1000) * 0.0005
    output_cost = (total_output_characters / 1000) * 0.0005
    print('text-bison Cost Estimation')
    print(f'Input: {round(input_cost, 2)}$')
    print(f'Output: {round(output_cost, 2)}$')
    print(f'Total: {round(input_cost + output_cost, 2)}$')
    print()

    input_cost = (total_inp_tokens / 1000) * 0.03
    output_cost = (total_output_tokens / 1000) * 0.06
    print('GPT4 Cost Estimation')
    print(f'Input: {round(input_cost, 2)}$')
    print(f'Output: {round(output_cost, 2)}$')
    print(f'Total: {round(input_cost + output_cost, 2)}$')
    print()

    input_cost = (total_inp_tokens / 1000) * 0.0015
    output_cost = (total_output_tokens / 1000) * 0.002
    print('ChatGPT Cost Estimation')
    print(f'Input: {round(input_cost, 2)}$')
    print(f'Output: {round(output_cost, 2)}$')
    print(f'Total: {round(input_cost + output_cost, 2)}$')
    print()


def get_multiwoz_prompts(args):
    args['booking_status'] = True

    with open(args['src_path'], 'r') as fp:
        data = json.load(fp)
    print('Number of samples', len(data))

    with open(args['index_path'], 'r') as fp:
        index_data = json.load(fp)
    print('Number of index samples', len(index_data))

    ablation = args['ablation']
    print('Ablation:', ablation)

    if ablation == "retriever" or args['noret'] == True:
        index_idxs = np.random.randint(0, len(index_data), size=2)
        print('Using fixed indices in samples', index_idxs)
    else:
        retriever = SentenceBertRetriever(
            index_data, dataset=args['dataset'], hint_ranking=args['hint_ranking']
        )
    kb_retriever = KBRetriever(dataset=args['dataset'])

    if args['mode'] == 'manual':
        print('Mode set to manual.')
        assert args['manual_hint_key'] is not None, f"Manual mode requires a key"
    elif args['mode'] == 'oracle':
        print('Using gold hints.')
    else:
        print('Using predicted hints.')

    etype_metric = None
    if args['entity_generation']:
        assert args['entity_file'] is not None
        print('Enabling entity generation...')
        etype_metric = EntityTypeMetric(args['dataset'], args['entity_file'])

    num_exemplars = 2
    prompt_samples = []
    gpt35_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    kb_prune_count = 0
    for ii in trange(len(data)):
        text = data[ii]['context'][-1]
        if ablation == "retriever" or args['noret'] == True:
            exemplars = [index_data[jj] for jj in index_idxs]

        elif args['hint_ranking'] and args['mode'] == 'manual':
            exemplars = retriever.search_top_k(
                text, num_exemplars, uuid=data[ii]['uuid'], etype=data[ii]['type'],
                hints=data[ii][args['manual_hint_key']]
            )[::-1]

        elif args['hint_ranking'] and 'predicted_hints' in data[ii]:
            exemplars = retriever.search_top_k(
                text, num_exemplars, uuid=data[ii]['uuid'], etype=data[ii]['type'],
                hints=data[ii]['predicted_hints']
            )[::-1]
        else:
            exemplars = retriever.search_top_k(
                text, num_exemplars, uuid=data[ii]['uuid'], etype=data[ii]['type'],
            )[::-1]
        sample = deepcopy(data[ii])

        if args['mode'] == 'oracle':
            pass
        elif args['mode'] == 'predicted':
            sample['hints'] = deepcopy(sample['predicted_hints'])
        elif args['mode'] == 'manual':
            sample['hints'] = deepcopy(sample.get(args['manual_hint_key'], None))
            assert sample['hints'] is not None
        else:
            raise NotImplementedError

        if args['booking_status']:
            fix_booking_status(sample)
            for jj in range(len(exemplars)):
                fix_booking_status(exemplars[jj])

        fix_multiwoz_typos(sample)
        for jj in range(len(exemplars)):
            fix_multiwoz_typos(exemplars[jj])

        prompt = multiwoz_prompt(sample, deepcopy(exemplars), args['booking_status'], etype_metric, ablation=ablation)

        if len(gpt35_enc.encode(prompt)) > THRESH:
            retrieve_kb(sample, kb_retriever)
            for jj in range(len(exemplars)):
                retrieve_kb(exemplars[jj], kb_retriever)
            prompt = multiwoz_prompt(sample, deepcopy(exemplars), args['booking_status'], etype_metric, ablation=ablation)
            kb_prune_count += 1

        sample['prompt'] = prompt
        sample['prompt_version'] = MULTIWOZ_PROMPT_VERSION
        prompt_samples.append(sample)

    print('Samples with pruned KB', kb_prune_count)
    print_estimated_costs(prompt_samples, gpt35_enc)

    print(f'Saving prompts in {args["tar_path"]}')
    with open(args["tar_path"], 'w') as fp:
        json.dump(prompt_samples, fp, indent=2)


def get_smd_prompts(args):
    with open(args['src_path'], 'r') as fp:
        data = json.load(fp)
    print('Number of samples', len(data))
    ablation = args['ablation']
    print('Ablation:', ablation)

    with open(args['index_path'], 'r') as fp:
        index_data = json.load(fp)
    print('Number of index samples', len(index_data))

    if ablation == "retriever" or args['noret'] == True:
        index_idxs = np.random.randint(0, len(index_data), size=2)
        print('Using fixed indices in samples', index_idxs)
    else:
        retriever = SentenceBertRetriever(
            index_data, dataset=args['dataset'], hint_ranking=args['hint_ranking']
        )

    kb_retriever = KBRetriever(dataset=args['dataset'])

    if args['mode'] == 'manual':
        print('Mode set to manual.')
        assert args['manual_hint_key'] is not None, f"Manual mode requires a key"
    elif args['mode'] == 'oracle':
        print('Using gold hints.')
    else:
        print('Using predicted hints.')

    etype_metric = None
    if args['entity_generation']:
        assert args['entity_file'] is not None
        print('Enabling entity generation...' , args['dataset'])
        etype_metric = EntityTypeMetric(args['dataset'], args['entity_file'])

    num_exemplars = 2
    prompt_samples = []
    gpt35_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    kb_prune_count = 0
    for ii in trange(len(data)):
        text = ' '.join(data[ii]['context'])
        if ablation == "retriever" or args['noret'] == True:
            exemplars = [index_data[jj] for jj in index_idxs]
        elif args['hint_ranking'] and 'predicted_hints' in data[ii]:
            exemplars = retriever.search_top_k(
                text, num_exemplars, uuid=data[ii]['uuid'], etype=data[ii]['type'],
                hints=data[ii]['predicted_hints']
            )[::-1]
        else:
            exemplars = retriever.search_top_k(
                text, num_exemplars, uuid=data[ii]['uuid'], etype=data[ii]['type'],
            )[::-1]
        sample = deepcopy(data[ii])

        if args['mode'] == 'oracle':
            pass
        elif args['mode'] == 'predicted':
            sample['hints'] = deepcopy(sample['predicted_hints'])
        elif args['mode'] == 'manual':
            sample['hints'] = deepcopy(sample.get(args['manual_hint_key'], None))
            assert sample['hints'] is not None
        else:
            raise NotImplementedError

        fix_smd_typos(sample)
        for jj in range(len(exemplars)):
            fix_smd_typos(exemplars[jj])

        prompt = smd_prompt(sample, deepcopy(exemplars), etype_metric, ablation=ablation)

        if len(gpt35_enc.encode(prompt)) > THRESH:
            retrieve_kb(sample, kb_retriever)
            for jj in range(len(exemplars)):
                retrieve_kb(exemplars[jj], kb_retriever)
            prompt = smd_prompt(sample, deepcopy(exemplars), etype_metric, ablation=ablation)
            kb_prune_count += 1

        sample['prompt'] = prompt
        sample['prompt_version'] = MULTIWOZ_PROMPT_VERSION
        prompt_samples.append(sample)

    print('Samples with pruned KB', kb_prune_count)
    print_estimated_costs(prompt_samples, gpt35_enc)

    print(f'Saving prompts in {args["tar_path"]}')
    with open(args["tar_path"], 'w') as fp:
        json.dump(prompt_samples, fp, indent=2)


def fix_bitod_typos(sample, flag='gold'):
    for row in sample['kb']:
        if 'ref_number' in row:
            row['reference number'] = row['ref_number']
            del row['ref_number']
            
    for ii in range(len(sample['hints']['entity_types'])):
        if sample['hints']['entity_types'][ii] == 'ref_number':
            sample['hints']['entity_types'][ii] = 'reference number'

    tkb = []
    for jj in range(len(sample['kb'])):
        tentry = dict()
        for k in sample['kb'][jj]:
            tentry[k.replace('_', ' ')] = re.sub('_', ' ', sample['kb'][jj][k])
        tkb.append(tentry)
        # for k in sample['kb'][jj]:
        #     sample['kb'][jj][k] = re.sub('_', ' ', sample['kb'][jj][k])
    sample['kb'] = tkb

    for jj in range(len(sample['context'])):
        sample['context'][jj] = re.sub('_', ' ', sample['context'][jj])

    if flag == 'gold':
        for jj in range(len(sample['gold_entities'])):
            if sample['gold_entities'][jj][0] == 'ref_number':
                sample['gold_entities'][jj] = [
                    'reference number',
                    sample['gold_entities'][jj][1].replace('_', ' ')
                ]

            else:
                sample['gold_entities'][jj] = [
                    sample['gold_entities'][jj][0].replace('_', ' '),
                    sample['gold_entities'][jj][1].replace('_', ' ')
                ]

        sample['output'] = sample['output'].replace('_', ' ')

    sample['hints']['entity_types'] = [
        x.replace('_', ' ')
        for x in sample['hints']['entity_types']
    ]


def bitod_prompt(orig_sample, exemplars, etype_metric=None, ablation='none'):
    all_cols = [
        'name', 'address', 'phone number', 'location', 'rating', 'price level', 'reference number',
        'stars', 'price per night', 'number of rooms', 'number of nights', 'user name', 'start month', 'start day',
        'cuisine', 'dietary restrictions', 'number of people', 'month', 'day', 'time', 'type'
    ]

    instructions = """Henceforth, assume that you are a customer support expert. I will give you an incomplete dialog between a user and a customer service representative. As an expert, you must suggest the most appropriate follow-up response to the dialog. Ensure you also include correct information (entities) from the given database. Entities can be of the following types - 
1. name - name of a place (restaurant, hotel or attraction)
2. address - address of the place
3. phone number - phone number of the place
4. location - a part of the city e.g. canal road, central district
5. rating - user rating of the place out of 10 e.g. 8, 9
6. price level - price range of the place, e.g. cheap, moderate, expensive
7. reference number - reference code for booking, e.g. 542j9wog
8. stars - star rating of the hotel, e.g. 3 stars
9. price per night - hotel charges per night e.g. 512, 600, etc.
10. number of rooms - number of rooms to book for the customer e.g. 1, 2
11. number of nights - number of nights the customer wants to book the hotel e.g. 2, 3
12. user name - name of the user e.g. Jack, Henry
13. start month - starting month of the booking e.g. July, May, etc.
14. start day - starting day of the booking e.g. 12, 30 etc.
15. cuisine - the cuisine of a restaurant, e.g. thai, chinese, etc.
16. dietary restrictions - dietary restrictions that the restaurant facilitates e.g. vegan, gluten free
17. number of people - number of people to reserve a restaurant for e.g. 2, 10
18. month - a month of the year e.g. january, february, etc.
19. day - a day of the month/week e.g. 12, 17, monday, etc.
20. time - a time of the daay e.g. 1200, 1330, 930, etc.
21. type - type of an attraction e.g. zoos and aquariums, shopping, etc.

"""

    if ablation == 'all_rules':
        instructions += """Here are the examples -"""
    else:
        instructions += """As an expert, you are very strict about following rules. Make sure that the follow-up response you write follows all the given rules. Here are the examples -"""

    prompt = [instructions]
    prompt.append('')
    exemplars = exemplars + [orig_sample]

    for ii, tee in enumerate(exemplars):
        sample = deepcopy(tee)
        prompt.append(f'[example {ii + 1}]')
        prompt.append(f'[database {ii + 1}]')

        cols = []
        for kk in sample['kb']:
            cols.extend(list(kk))
        cols = set(cols)
        cols = [cc for cc in all_cols if cc in cols]

        final_cols = []
        for col in cols:
            flag = False
            for row in sample['kb']:
                vv = row.get(col, None)
                if vv is not None and vv != '-':
                    flag = True

            if flag:
                final_cols.append(col)
        cols = final_cols

        tmp_kb = []
        for row in sample['kb']:
            tmp_row = dict()
            for c in cols:
                tmp_row[c] = row.get(c, '-')
            tmp_kb.append(tmp_row)
        assert len(tmp_kb) != 0

        index_col = 'name'
        kb_df = pd.DataFrame(tmp_kb)
        indices = kb_df[index_col].tolist()
        kb_df[index_col] = dedup_index(indices)
        kb_df = kb_df.set_index(index_col)
        tmp_text = kb_df.to_json(orient='index', indent=2)
        prompt.append(tmp_text)
        prompt.append('')

        if ablation not in ['all_rules']:
            prompt.append(f'[rules {ii + 1}]')
            hints = sample['hints']
            prompt.append(f'The response must be {hints["word_count"]} words or shorter.')
            if hints['closure']:
                prompt.append('The response must close the dialog.')
            else:
                prompt.append('The response must not close the dialog.')

            tetypes = sorted(hints['entity_types'])
            tetypes = [cc for cc in tetypes if cc in all_cols]
            ntetypes = [cc for cc in all_cols if cc not in tetypes]
            if len(hints['entity_types']) > 0:
                prompt.append(f"The response must only include entities of type - {', '.join(tetypes)}.")
            prompt.append(f"The response must not include any entities of type - {', '.join(ntetypes)}.")

            prompt.append('')

        prompt.append(f'[dialog history {ii + 1}]')
        for jj, uttr in enumerate(sample['context']):
            spk = 'user' if jj % 2 == 0 else 'assistant'
            prompt.append(f"{spk}: {uttr}")
        prompt.append('')

        prompt.append(f"[follow-up response {ii + 1}]")

        # if ablation == 'all_rules':
        if ablation in ['all_rules']:
            # prompt.append(f"As an expert, you must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}] and write the response.")
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}] and write the response.")
            if ii == len(exemplars) - 1:
                prompt.append(f"assistant:")
            else:
                prompt.append(f"assistant: {sample['output']}")
                prompt.append("")
        elif etype_metric is None:
            prompt.append(f"As an expert, you must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}], follow all the [rules {ii + 1}] and write the response.")
            if ii == len(exemplars) - 1:
                prompt.append(f"assistant:")
            else:
                prompt.append(f"assistant: {sample['output']}")
                prompt.append("")
        else:
            prompt.append(f"Let's think step-by-step.")
            prompt.append(f"As an expert, I must understand the user's requirements from [dialog history {ii + 1}], identify the relevant information from the [database {ii + 1}], follow all the [rules {ii + 1}] and write the response.")
            prompt.append(f"I will include entities of type {tetypes} in my response.")

            if ii == len(exemplars) - 1:
                prompt.append(f"I will include these entities -")
            else:
                tmp = sample['gold_entities']
                for hh in range(len(tmp)):
                    if tmp[hh][0] == 'ref':
                        tmp[hh] = ['reference number', tmp[hh][1]]
                    elif tmp[hh][0] == 'pricerange':
                        tmp[hh] = ['price range', tmp[hh][1]]

                tents = []
                for cc in sorted(set(tetypes)):
                    for ee in tmp:
                        if ee[0] == cc:
                            tents.append(ee)

                prompt.append(f"I will include these entities - {tents}")
                prompt.append(f"assistant: {sample['output']}")
                prompt.append("")

    prompt = '\n'.join(prompt)

    return prompt


def get_bitod_prompts(args):
    with open(args['src_path'], 'r') as fp:
        data = json.load(fp)
    print('Number of samples', len(data))

    with open(args['index_path'], 'r') as fp:
        index_data = json.load(fp)
    print('Number of index samples', len(index_data))

    ablation = args['ablation']
    print('Ablation:', ablation)

    if ablation == "retriever" or args['noret'] == True:
        index_idxs = np.random.randint(0, len(index_data), size=2)
        print('Using fixed indices in samples', index_idxs)
    else:
        retriever = SentenceBertRetriever(
            index_data, dataset=args['dataset'], hint_ranking=args['hint_ranking']
        )
    kb_retriever = KBRetriever(dataset=args['dataset'])

    if args['mode'] == 'manual':
        print('Mode set to manual.')
        assert args['manual_hint_key'] is not None, f"Manual mode requires a key"
    elif args['mode'] == 'oracle':
        print('Using gold hints.')
    else:
        print('Using predicted hints.')

    etype_metric = None
    if args['entity_generation']:
        assert args['entity_file'] is not None
        print('Enabling entity generation...')
        etype_metric = EntityTypeMetric(args['dataset'], args['entity_file'])

    num_exemplars = 2
    prompt_samples = []
    gpt35_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    kb_prune_count = 0
    for ii in trange(len(data)):
        text = ' '.join(data[ii]['context'])

        if ablation == "retriever" or args['noret'] == True:
            exemplars = [index_data[jj] for jj in index_idxs]
        elif args['hint_ranking'] and 'predicted_hints' in data[ii]:
            exemplars = retriever.search_top_k(
                text, num_exemplars, uuid=data[ii]['uuid'], etype=data[ii]['type'],
                hints=data[ii]['predicted_hints']
            )[::-1]
        else:
            exemplars = retriever.search_top_k(
                text, num_exemplars, uuid=data[ii]['uuid'], etype=data[ii]['type'],
            )[::-1]
        sample = deepcopy(data[ii])

        if args['mode'] == 'oracle':
            pass
        elif args['mode'] == 'predicted':
            sample['hints'] = deepcopy(sample['predicted_hints'])
        elif args['mode'] == 'manual':
            sample['hints'] = deepcopy(sample.get(args['manual_hint_key'], None))
            assert sample['hints'] is not None
        else:
            raise NotImplementedError

        fix_bitod_typos(sample)
        # print(sample['hints'])
        for jj in range(len(exemplars)):
            fix_bitod_typos(exemplars[jj], flag='gold')
            # print(exemplars[jj]['hints'], exemplars[jj]['gold_entities'])

        prompt = bitod_prompt(sample, deepcopy(exemplars), etype_metric, ablation=ablation)
        # print(prompt)

        if len(gpt35_enc.encode(prompt)) > THRESH - 500:
            print('WARNING... extreme reduction')
            retrieve_kb(sample, kb_retriever)
            for jj in range(len(exemplars)):
                retrieve_kb(exemplars[jj], kb_retriever)
            prompt = bitod_prompt(sample, deepcopy(exemplars), etype_metric, ablation=ablation)
            kb_prune_count += 1

        sample['prompt'] = prompt
        sample['prompt_version'] = MULTIWOZ_PROMPT_VERSION
        prompt_samples.append(sample)

    print('Samples with pruned KB', kb_prune_count)
    print_estimated_costs(prompt_samples, gpt35_enc)

    print(f'Saving prompts in {args["tar_path"]}')
    with open(args["tar_path"], 'w') as fp:
        json.dump(prompt_samples, fp, indent=2)


if __name__ == '__main__':
    args = read_cli()
    if args['dataset'] == 'MultiWOZ':
        get_multiwoz_prompts(args)
    elif args['dataset'] == 'SMD':
        get_smd_prompts(args)
    elif args['dataset'] == 'BiTOD':
        get_bitod_prompts(args)
