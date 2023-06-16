### Evaluate their best model!
CUDA_VISIBLE_DEVICES=0 python run_coder.py \
                --ddp_find_unused_parameters False \
                --disable_tqdm True \
                --version mimic3-50 --model_name_or_path C:\\Users\\Joe\\Desktop\\KEPT\\longformer-original-clinical-prompt2alpha\\longformer-original-clinical-prompt2alpha\\checkpoint-20165 \
                --do_eval --do_predict --max_seq_length 8192 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1.5e-5 --weight_decay 1e-3 --adam_epsilon 1e-7 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step --global_attention_strides 1 \
                --output_dir OUTPUT_DIR