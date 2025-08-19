SAVE_PATH=./models
# python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf --path $SAVE_PATH --hf_token YOUR_HF_TOKEN
python scripts/convert_hf_checkpoint.py --checkpoint_dir $SAVE_PATH/meta-llama/Llama-2-7b-hf 
