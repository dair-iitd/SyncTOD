PATH_TO_DATASET=../data/MultiWOZ/

python -u -m prompting.add_hints -src=$PATH_TO_DATASET/train.json -ent=$PATH_TO_DATASET/entities.json -ds=MultiWOZ -hints=oracle
python -u -m prompting.add_hints -src=$PATH_TO_DATASET/valid.json -ent=$PATH_TO_DATASET/entities.json -ds=MultiWOZ -hints=oracle
python -u -m prompting.add_hints -src=$PATH_TO_DATASET/test.json -ent=$PATH_TO_DATASET/entities.json -ds=MultiWOZ -hints=oracle

python -u -m hints.train -cfg=hints/configs/multiwoz/deberta_base_closure.yml
python -u -m hints.test -cfg=hints/configs/multiwoz/deberta_base_closure.yml -rs=$PATH_TO_DATASET/closure_test_results.json
python -u -m hints.train -cfg=hints/configs/multiwoz/flan_t5_large_etypes.yml
python -m hints.test -cfg=hints/configs/multiwoz/flan_t5_large_etypes.yml -rs=$PATH_TO_DATASET/etypes_test_results.json

python -m prompting.add_hints -src=$PATH_TO_DATASET/test.json \
    -ent=$PATH_TO_DATASET/entities.json -ds=MultiWOZ -hints=predicted \
    -cpred_path=$PATH_TO_DATASET/closure_test_results.json \
    -epred_path=$PATH_TO_DATASET/etypes_test_results.json -wl=15

python -m prompting.prompt_gen -ds=MultiWOZ \
    -idx=$PATH_TO_DATASET/train.json -src=$PATH_TO_DATASET/test.json \
    -tar=$PATH_TO_DATASET/prompts.json -mode=predicted -hr=True \
    -eg=True -ef=$PATH_TO_DATASET/entities.json
