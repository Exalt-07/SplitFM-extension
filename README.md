## SFF : Split Federated Finetuning

SFF builds upon the foundation of [LoRA](https://github.com/microsoft/LoRA) and Split Learning to enable privacy-preserving, parameter-efficient fine-tuning of foundation models. We introduce significant enhancements to support **heterogeneous client environments** and **Federated aggregation strategies**.

**Key Contributions:**

1. **Heterogeneous LoRA Ranks**: We have extended the framework to support heterogeneous ranks across clients (e.g., `[4, 8, 16]`).
2. **Federated Aggregation Schemes**: We have integrated four distinct aggregation methods to handle the updates from these heterogeneous clients:
    * **Average**: Performs naive element-wise averaging of adapter matrices, though this can introduce cross-term interference that may destabilize the split forward pass.
    * **Freeze**: Freezes one projection matrix while training only the other to enforce linearity and eliminate cross-term noise, at the cost of reduced trainable parameters.
    * **Stack**: Concatenates adapters to preserve distinct client subspaces without information loss, though this increases the global rank linearly with the number of clients, leading to higher communication costs.
    * **SVD**: Aggregates updates in the full update space and re-projects via SVD to maintain a fixed rank, balancing noise reduction with computational efficiency by identifying principal update directions.

This framework currently supports PyTorch-based GPT-2 models, with plans to integrate more open-source LLMs in the future.

### User Guide

#### 1. Build

##### 1.1 Environment Requirements

We have verified in the environment below:

+ OS: Ubuntu 22.04
+ Python: 3.10.0

##### 1.2 Installation

1. Clone the repo and set up the environment.

```bash
conda create -n SFF python=3.10 -y
conda activate SFF
```

2. Navigate to the examples directory and install the required packages.

```bash
cd SplitLoRA/examples
pip install -r requirements.txt
```

3. Download the necessary pre-trained models, datasets, and evaluation scripts.

```bash
# Download pre-trained GPT-2 checkpoints
bash download_pretrained_checkpoints.sh

# Prepare datasets
bash create_datasets.sh

# Download evaluation scripts
cd ./eval
bash download_evalscript.sh
cd ..
```

#### 2. SplitLoRA Module Libraries

##### 2.1 Repository

Our implementation is based on the fine-tuning code for GPT-2 in Hugging Face.
There are several directories in this repo:

* `src/` contains the source code used for data processing, training, and decoding.
* `eval/` contains the code for task-specific evaluation scripts.
* `data/` contains the raw data we used in our experiments.
* `vocab/` contains the GPT-2 vocabulary files.

##### 2.2 Key Hyper-Parameters

| Argument | Description | Default/Example |
|---|---|---|
| `--train_batch_size` | Training batch size. | 4 |
| `--grad_acc` | Number of gradient accumulation steps. | 1 |
| `--seq_len` | Sequence length. | 512 |
| `--model_card` | Path to the model configuration file. | ${MODEL_CARD} |
| `--init_checkpoint` | Path to the initial checkpoint file. | gpt2-large-pytorch_model.bin |
| `--platform` | Execution platform. | local |
| `--lr` | Learning rate. | 0.0002 |
| `--max_epoch` | Maximum number of training epochs. | 1 |
| `--lora_dim` | The dimension of LoRA. | ${LORA_DIM} |
| `--lora_alpha` | Alpha hyperparameter for LoRA. | 32 |
| `--cut_layer` | The layer index where the model is split. | ${CUT_LAYER} |
| `--agg_method` | Aggregation method (e.g., stack, svd, avg, freeze). | ${AGG_METHOD} |
| `--lora_ranks` | List of heterogeneous ranks for clients. | ${LORA_RANKS} |
| `--work_dir` | Working directory for saving models/logs. | . |

#### 3. Training Process

##### 1. Train GPT-2 with Heterogeneous SplitLoRA

Run the following command to start training. Ensure you set your environment variables (like MODEL_CARD, LORA_DIM, AGG_METHOD, etc.) before running.

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM%10000+20000)) --use_env src/gpt2_ft_sfl.py \
    --train_data0 ./data/e2e/train0.jsonl \
    --train_data1 ./data/e2e/train1.jsonl \
    --train_data2 ./data/e2e/train2.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 4 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card ${MODEL_CARD} \
    --init_checkpoint ./pretrained_checkpoints/gpt2-large-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 1 \
    --save_interval 999999 \
    --lora_dim ${LORA_DIM} \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir "." \
    --random_seed ${SEED} \
    --cut_layer ${CUT_LAYER} \
    --agg_method ${AGG_METHOD} \
    --lora_ranks ${LORA_RANKS}
```

##### 2. Generate Outputs (Inference)

Use beam search to generate outputs from the trained model.

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM%10000+20000)) --use_env src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 8 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card ${MODEL_CARD} \
    --init_checkpoint "$CHECKPOINT_PATH" \
    --platform local \
    --lora_dim ${CURRENT_RANK} \
    --lora_alpha 32 \
    --beam 5 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir "${JOB_WORK_DIR}" \
    --output_file "${PREDICT_FILE}"
```

##### 3. Decode Outputs

Convert the generated JSONL outputs into flat text files for evaluation.

```bash
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file "${JOB_WORK_DIR}/${PREDICT_FILE}" \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file "${REF_FILE}" \
    --output_pred_file "${PRED_FILE}"
```

##### 4. Run Evaluation

Evaluate the decoded predictions against the reference file.

```bash
python eval/e2e/measure_scores.py "${REF_FILE}" "${PRED_FILE}" -p
```
