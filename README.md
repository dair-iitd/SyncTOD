# SyncTOD: Synergizing In-context Learning with Hints for End-to-end Task-oriented Dialog Systems

This is an official repository for the paper - Synergizing In-context Learning with Hints for End-to-end Task-oriented Dialog Systems.

## Code Structure

`src/` contains source code to train SyncTOD.

`data/` contains dataset used to evaluate SyncTOD.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/dair-iitd/SyncTOD.git
    ```

2. Navigate to the source directory:

    ```bash
    cd SyncTOD/src
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed, you can start training or evaluating the system. All our code is run from the <b>src<b> directory.

Use following commands to train SyncTOD and generate the prompts.

1. MultiWOZ:

    ```bash
    bash run_multiwoz.sh
    ```

2. SMD:

    ```bash
    bash run_smd.sh
    ```

3. BiTOD:

    ```bash
    bash run_bitod.sh
    ```

Above command stores the test prompts in `prompts.json` file in respective dataset folders.

We use Azure OpenAI batch API to run the prompts. 
    ```
     python -u submit_job.py --prompt_file=../data/BiTOD/prompts.json --model=gpt-35-turbo --max_output_len=256
    ```
