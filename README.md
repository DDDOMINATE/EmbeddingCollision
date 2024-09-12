# Run Vec2text Training on server

## 1. Activate conda env
```
conda activate vec2text
```

## 2. Go to workdir
```
cd ~/workspace/vec2text/vec2text
```

## 3. Run training
```
CUDA_VISIBLE_DEVICES=3 python run.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 128 --model_name_or_path /data0/lsf/t5-base --dataset_name msmarco --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 100 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment inversion --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001 --output_dir ./saves/gtr-1 --save_steps 2000 
```
