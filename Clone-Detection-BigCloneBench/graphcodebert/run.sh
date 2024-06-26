CUDA_VISIBLE_DEVICES=0 python run.py \
    --output_dir=./saved_models/FreeLB/ \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_graphcodebert_coda.txt \
    --epoch 2 \
    --code_length 448 \
    --data_flow_length 64 \
    --train_batch_size 12 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3 2>&1| tee ./saved_models/train.log
