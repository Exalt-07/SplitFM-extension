#!/bin/bash

echo "downloading pretrained model checkpoints..."
mkdir weights
modelscope download --model Qwen/Qwen2-VL-7B-Instruct --local_dir ./weights/Qwen2-VL-7B-Instruct
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local_dir ./weights/DeepSeek-R1-Distill-Llama-8B
modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct --local_dir ./weights/Llama-3-8B-Instruct

echo "script complete!"