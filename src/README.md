We provide the datasets and code used our experiments in <b>data</b> and <b>src</b> directory respectively. All our code is run from the 'src' directory.

1. Create the train, validation datasets for learning hint prediction models.
```
python -m prompting.add_hints -src=<path_to_dataset>/train.json -ent=<path_to_dataset>/entities.json -ds=<dataset> -hints=oracle
python -m prompting.add_hints -src=<path_to_dataset>/valid.json -ent=<path_to_dataset>/entities.json -ds=<dataset> -hints=oracle
python -m prompting.add_hints -src=<path_to_dataset>/test.json -ent=<path_to_dataset>/entities.json -ds=<dataset> -hints=oracle
```
Here, <dataset> can be either MultiWOZ or SMD.

2. Train the hint prediction models using the provided config files.
```
python -m hints.train -cfg=hints/configs/<dataset>/deberta_base_closure.yml
python -m hints.train -cfg=hints/configs/<dataset>/flan_t5_large_etypes.yml
```
Above commands will put the trained models in runs/ directory

3. Obtain the hint predictions on the test set.
```
python -m hints.test -cfg=hints/configs/<dataset>/deberta_base_closure.yml -rs=../data/<dataset>/closure_test_results.json

python -m hints.test -cfg=hints/configs/<dataset>/flan_t5_large_etypes.yml -rs=../data/<dataset>/etypes_test_results.json
```
where <model> points to the result directory created by the training.

4. Generate prompts into a JSON file.
```
python -m prompting.add_hints -src=../data/<dataset>/test.json -ent=../data/<dataset>/entities.json -ds=CamRest -hints=predicted -cpred_path=../data/<dataset>/closure_test_results.json -epred_path=../data/<dataset>/etypes_test_results.json -wl=<avg_response_length>

python -u -m prompting.prompt_gen -ds=<dataset> -idx=../data/<dataset>/train.json -src=../data/<dataset>/test.json -tar=../data/<dataset>/prompt_test.json -mode=predicted -hr=True -eg=True -kf=False -ef=../data/<dataset>/entities.json
```
