## Environment configuration
To reproduce our experiments, machines with GPUs and NVIDIA CUDA toolkit are required.
The environment dependencies are listed in the file "requirements.txt". You can create conda environment to install required dependenc
```
conda create --name <env> --file requirements.txt
```
## Tasks and Datasets
We evaluate the calibraiton of pre-trained code models on different code understanding tasks. The datasets can be downloaded from the following sources:

Clone Detection: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench
Defect Detection: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection

## How to run 
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --output_dir=./saved_models/FreeLB/ \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../attack/attack_csv/defect/codebert/ori/test_adv_coda.jsonl \
    --epoch 1 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3  2>&1 | tee ./saved_models/train.log
```
