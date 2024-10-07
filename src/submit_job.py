import json
import argparse
import tempfile
import os
import time
import datetime

from openai import AzureOpenAI


def read_cli():
    parser = argparse.ArgumentParser(description='OpenAI')
    parser.add_argument(
        "-prompt_file",
        "--prompt_file",
        help="Prompt file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-model",
        "--model",
        help="Model",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-max_output_len",
        "--max_output_len",
        help="max_output_len",
        required=False,
        type=int,
        default=512,
    )

    args = vars(parser.parse_args())
    return args


def get_converted_data(input_path, model, max_tokens, temperature=0):
    with open(input_path, 'r') as f:
        data = json.load(f)

    tar_data = []
    for entry in data:
        uuid, turn_id = entry['uuid'], entry['turn_id']
        idx = f"sid_{uuid}_{turn_id}"
        sample = {
            'custom_id': idx,
            'method': 'POST',
            'url': '/chat/completions',
            'body': {
                'model': model,
                'messages': [{'role': 'system', 'content': entry['prompt']}],
                'temperature': temperature,
                'max_tokens': max_tokens,
            }
        }
        tar_data.append(sample)

    return tar_data


def submit_job(fname):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-07-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Upload a file with a purpose of "batch"
    with open(fname, "rb") as fp:
        file = client.files.create(file=fp, purpose="batch")

    print(file.model_dump_json(indent=2))
    file_id = file.id

    print('Waiting for the file upload...')

    status = "pending"
    while status != "processed":
        time.sleep(15)
        file_response = client.files.retrieve(file_id)
        status = file_response.status
        print(f"{datetime.datetime.now()} File Id: {file_id}, Status: {status}")

    # Submit a batch job with the file
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h",
    )
    # Save batch ID for later use
    batch_id = batch_response.id
    print(batch_response.model_dump_json(indent=2))

    status = "validating"
    while status not in ("completed", "failed", "canceled", "in_progress"):
        time.sleep(60)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

    print(f"Batch Id: {batch_id},  Status: {status}")


def main(args):
    data = get_converted_data(args['prompt_file'], args['model'], max_tokens=args['max_output_len'])
    print(f'Number of prompts', len(data))

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as temp_file:
        lines = [json.dumps(ee) for ee in data]
        temp_file.write('\n'.join(lines))
        temp_file_name = temp_file.name

    print('Processed prompts are stored in', temp_file_name)

    submit_job(temp_file_name)

    os.remove(temp_file_name)


if __name__ == "__main__":
    args = read_cli()
    main(args)
