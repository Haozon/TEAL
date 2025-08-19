check_path=./models
CUDA_VISIBLE_DEVICES=1 python generate.py --compile --checkpoint_path $check_path/meta-llama/Llama-2-7b-hf/model.pth --interactive