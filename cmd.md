python3 -W ignore main_quant.py \
    --config configs/my-internvl2-2b/MBQ_search/2b_weight_only.yaml

git submodule update --init --recursive 3rdparty/LLaVA-NeXT

python3 -W ignore main_quant.py \
    --config configs/internvl2/MBQ_search/my_1b_weight_only.yaml

python3 -W ignore main.py \
    --config configs/internvl2/Eval/my_eval.yaml