# Train and eval on rare50:

CUDA_VISIBLE_DEVICES=0 python run_coder.py \
                --ddp_find_unused_parameters False \
                --finetune_terms "bias;lm_head" \
                --disable_tqdm False \
                --version mimic3-50l --model_name_or_path C:\\Users\\Joe\\Desktop\\KEPT\\longformer-original-clinical-prompt2alpha\\longformer-original-clinical-prompt2alpha\\checkpoint-20165 \
                --do_train --do_eval --max_seq_length 2048 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1e-3 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --load_best_model_at_end True --metric_for_best_model eval_f1_macro --greater_is_better True \
                --logging_first_step --global_attention_strides 1 \
                --output_dir rare50_maxseqlen2048\\checkpoint-996

# Eval the specific checkpoint on rare50.
CUDA_VISIBLE_DEVICES=0 python run_coder.py \
                --ddp_find_unused_parameters False \
                --disable_tqdm False \
                --version mimic3-50l --model_name_or_path rare50_maxseqlen2048\\checkpoint-996 \
                --do_eval --do_predict --max_seq_length 2048 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1.5e-5 --weight_decay 1e-3 --adam_epsilon 1e-7 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step --global_attention_strides 1 \
                --output_dir rare50_maxseqlen2048\\checkpoint-996 
 
# TODO
- [x] analyse errors
- [ ] analyse successes