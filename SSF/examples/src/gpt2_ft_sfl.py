import argparse
import time
import math
import os
import itertools
import warnings
import torch
import random
import copy
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F

from gpu import add_gpu_params, parse_gpu, distributed_opt, distributed_sync, cleanup
from torch.linalg import svd
from optimizer import (
    create_optimizer_scheduler,
    add_optimizer_params,
    create_adam_optimizer_from_args,
)

from data_utils import FT_Dataset
from splitmodel import GPT2Config, GPT2LMModel_Server, GPT2LMModel_Client
from exp_utils import create_exp_dir

import loralib as lora

torch.set_printoptions(threshold=100000)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="PyTorch GPT2 ft script")

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument(
    "--train_data0", required=True, help="location of training data corpus"
)

parser.add_argument(
    "--train_data1", required=True, help="location of training data corpus"
)

parser.add_argument(
    "--train_data2", required=True, help="location of training data corpus"
)

parser.add_argument(
    "--valid_data", required=True, help="location of validation data corpus"
)

parser.add_argument(
    "--train_batch_size", type=int, default=8, help="training batch size"
)

parser.add_argument(
    "--valid_batch_size", type=int, default=4, help="validation batch size"
)

parser.add_argument(
    "--grad_acc", type=int, default=1, help="gradient accumulation steps"
)

parser.add_argument("--clip", type=float, default=0.0, help="gradient clip")

parser.add_argument(
    "--seq_len", type=int, default=512, help="number of tokens to predict."
)

parser.add_argument(
    "--model_card",
    default="gpt2.md",
    choices=["gpt2.sm", "gpt2.md", "gpt2.lg"],
    help="model names",
)

parser.add_argument(
    "--init_checkpoint", default=None, help="pretrained checkpoint path"
)

parser.add_argument("--fp16", action="store_true", help="train model with fp16")

parser.add_argument("--log_interval", type=int, default=100, help="log interval")

parser.add_argument("--eval_interval", type=int, default=2000, help="eval interval")

parser.add_argument("--save_interval", type=int, default=500, help="save interval")

parser.add_argument(
    "--work_dir",
    type=str,
    default=os.getenv("PT_OUTPUT_DIR", "gpt2_model"),
    help="working folder.",
)

parser.add_argument("--lora_dim", type=int, default=0, help="lora attn dimension")

parser.add_argument("--lora_alpha", type=int, default=128, help="lora attn alpha")

parser.add_argument(
    "--obj",
    default="clm",
    choices=["jlm", "clm"],
    help="language model training objective",
)

parser.add_argument(
    "--lora_dropout",
    default=0.0,
    type=float,
    help="dropout probability for lora layers",
)
parser.add_argument("--label_smooth", default=0.0, type=float, help="label smoothing")

parser.add_argument("--roll_interval", type=int, default=-1, help="rolling interval")

parser.add_argument(
    "--roll_lr", type=float, default=0.00001, help="rolling learning rate"
)

parser.add_argument("--roll_step", type=int, default=100, help="rolling step")

parser.add_argument(
    "--eval_epoch", type=int, default=1, help="eval per number of epochs"
)
parser.add_argument(
    "--num_clients",
    type=int,
    default=3,
    help="Number of clients to simulate (overridden by lora_ranks length when provided).",
)
parser.add_argument(
    "--lora_ranks",
    type=str,
    default=None,
    help="Comma-separated LoRA ranks per client for heterogeneous setups. If unset, all clients use lora_dim.",
)
parser.add_argument('--cut_layer', type=int, default=3, help='layer index to split the model')
parser.add_argument("--agg_method", default="avg", choices=["avg", "stack", "svd", "freeze"], help="Aggregation scheme to use")
def debug_lora_integrity(tag, tensor_name, tensor):
    """
    Diagnoses if a LoRA_A matrix has been corrupted by naive padding.
    """
    if "lora_A" not in tensor_name or tensor is None:
        return

    # tensor shape is [2*rank, dim]
    total_rows = tensor.shape[0]

    # Inferred Rank
    r = total_rows // 2

    # Slice the top (Query) and bottom (Value)
    q_slice = tensor[:r, :]
    v_slice = tensor[r:, :]

    q_norm = q_slice.abs().sum().item()
    v_norm = v_slice.abs().sum().item()

    # Check for the specific "Naive Padding" signature:
    # If the bottom half is ALL zeros, but we expect trained weights there, it's broken.
    status = "OK"
    if v_norm == 0.0 and q_norm > 0.0:
        status = "CRITICAL: VALUE PROJECTION IS DEAD (Zeroes)"

    print(f"[{tag}] {tensor_name} | Shape: {tensor.shape} | Rank: {r}")
    print(f"    > Q_Norm (Rows 0-{r-1}): {q_norm:.4f}")
    print(f"    > V_Norm (Rows {r}-{total_rows-1}): {v_norm:.4f}  <-- {status}")
    print("-" * 60)
# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print("=" * 100)
        for k, v in args.__dict__.items():
            print(f"        - {k} : {v}")
        print("=" * 100)
def init_and_freeze_lora_A(model, seed=42):
    if seed is not None:
        torch.manual_seed(seed)
    print(f"DEBUG: Freezing LoRA_A matrices with seed {seed}...")
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            param.requires_grad = False

def save_checkpoint(w_glob_client, model_server, args, train_step, num_clients):
    if args.rank != 0:
        return

    model_state_dict = {}

    # rename the key in client model
    for key, value in w_glob_client.items():
        new_key = ""
        if key.startswith("transformer_Client"):
            new_key = key.replace("transformer_Client", "module.transformer")
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    # rename the key in server model
    for key, value in model_server.state_dict().items():
        new_key = ""
        # print(key)
        if key.startswith("module.transformer_Server"):
            new_key = key.replace("module.transformer_Server", "module.transformer")
        else:
            print(key)
            model_state_dict[key] = value

        if new_key.startswith("module.transformer.h."):
            parts = key.split(".")
            layer_idx = int(parts[3])
            new_key = ".".join(["module.transformer.h", str(layer_idx + args.cut_layer)] + parts[4:])
            model_state_dict[new_key] = value
        else:
            model_state_dict[new_key] = value
    ranks_str = args.lora_ranks.replace(',', '_') if args.lora_ranks else str(args.lora_dim)

    filename = f"model_sfl.{train_step}_r={args.lora_dim}_ranks={ranks_str}_num={num_clients}_block={args.cut_layer}_agg={args.agg_method}_seed={args.random_seed}.pt"

    if args.model_card == "gpt2.lg":
        model_path = os.path.join("./trained_models/GPT2_L/e2e", filename)
    elif args.model_card == "gpt2.md":
        model_path = os.path.join("./trained_models/GPT2_M/e2e", filename)
    else: # gpt2.sm
        model_path = os.path.join("./trained_models/GPT2_S/e2e", filename)

    print("\n=== DEBUG: SAVING CHECKPOINT ===")
    for k, v in model_state_dict.items():
        if "h.0.attn.c_attn.lora_A" in k:
            debug_lora_integrity("SAVING_CKPT", k, v)

    print("saving checkpoint", model_path)
    torch.save({"model_state_dict": model_state_dict}, model_path)

def infer_rank_from_lora_A(tensor):
    return tensor.shape[0] // 2

def infer_rank_from_lora_B(tensor):
    return tensor.shape[1]

def pad_lora_tensor(tensor, target_rank, is_A=True):
    # Determine current rank
    if is_A:
        current_rank = tensor.shape[0] // 2 # Assuming MergedLinear structure [2*r, dim]
    else:
        current_rank = tensor.shape[1]      # Assuming [dim, r]

    if current_rank == target_rank:
        return tensor

    if is_A:
        # MergedLinear A: [2*r, dim]. We need to split Q and V, pad them, and concat.
        chunks = torch.chunk(tensor, 2, dim=0)
        padded_chunks = []
        pad_rows = target_rank - current_rank
        for chunk in chunks:
            # Pad each chunk individually (Q gets padding, V gets padding)
            padded_chunk = F.pad(chunk, (0, 0, 0, pad_rows))
            padded_chunks.append(padded_chunk)
        return torch.cat(padded_chunks, dim=0)
    else:
        # MergedLinear B: [dim, r]. Just pad columns.
        pad_cols = target_rank - current_rank
        return F.pad(tensor, (0, pad_cols, 0, 0))

def truncate_lora_tensor(tensor, target_rank, is_A=True):
    if target_rank < 0:
        raise ValueError("target_rank must be non-negative")
    if is_A:
        return tensor[: 2 * target_rank, :]
    return tensor[:, :target_rank]

def extract_lora_state(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if key.endswith("lora_A") or key.endswith("lora_B")
    }

def resize_lora_state(state_dict, target_rank, mode="pad"):
    resized = {}
    for key, value in state_dict.items():
        if key.endswith("lora_A"):
            resized[key] = pad_lora_tensor(value, target_rank, is_A=True) if mode == "pad" else truncate_lora_tensor(value, target_rank, is_A=True)
        elif key.endswith("lora_B"):
            resized[key] = pad_lora_tensor(value, target_rank, is_A=False) if mode == "pad" else truncate_lora_tensor(value, target_rank, is_A=False)
        else:
            resized[key] = value
    return resized

def prefix_lora_keys(state_dict, prefix):
    return {f"{prefix}{k}": v for k, v in state_dict.items()}

def update_state_with_lora(base_state, lora_updates):
    for key, value in lora_updates.items():
        if key.endswith("lora_A") or key.endswith("lora_B"):
            base_state[key] = value
    return base_state


def extract_qv_lora_weights(W, n_embd):
    """
    Splits MergedLinear LoRA weights (Q and V) into separate dictionaries.
    Handles 'Packed' lora_B (missing Keys) and 'Narrow' lora_B (shared columns).
    """
    new_w = {}
    for key in W:
        if not key.endswith('lora_A'):
            continue

        layer_prefix = key.replace('.attn.c_attn.lora_A', '')
        A = W[f"{layer_prefix}.attn.c_attn.lora_A"]
        B = W[f"{layer_prefix}.attn.c_attn.lora_B"]
        r = A.shape[0] // 2

        # Check actual width of B
        b_width = B.shape[1]

        # Define Column Slices based on width
        if b_width == r:
            # Narrow B: Columns are reused/shared (0:r for both)
            q_col_slice = slice(0, r)
            v_col_slice = slice(0, r)
        else:
            # Standard B: Columns are concatenated (0:r for Q, r:2r for V)
            q_col_slice = slice(0, r)
            v_col_slice = slice(r, 2*r)

        # 1. Extract Q Projection
        # A: Top half. B: Top rows (0 to n_embd).
        new_w[f"{layer_prefix}.attn.q_proj.lora_A"] = A[0:r, :]
        new_w[f"{layer_prefix}.attn.q_proj.lora_B"] = B[0:n_embd, q_col_slice]

        # 2. Extract V Projection
        # A: Bottom half. B: Bottom rows (n_embd to 2*n_embd).
        new_w[f"{layer_prefix}.attn.v_proj.lora_A"] = A[r:2*r, :]
        new_w[f"{layer_prefix}.attn.v_proj.lora_B"] = B[n_embd:2*n_embd, v_col_slice]

    return new_w

def pad_lora_state_to_rank(w_state, target_rank):
    """Zero-pad LoRA A (rows) and B (columns) tensors up to target_rank."""
    padded = {}
    for key, tensor in w_state.items():
        if key.endswith("lora_A"):
            # USE THE HELPER FUNCTION!
            padded[key] = pad_lora_tensor(tensor, target_rank, is_A=True)
        elif key.endswith("lora_B"):
            # USE THE HELPER FUNCTION!
            padded[key] = pad_lora_tensor(tensor, target_rank, is_A=False)
        else:
            padded[key] = tensor
    return padded

def truncate_lora_state_to_rank(w_state, target_rank):
    """Truncate padded LoRA A (rows) and B (columns) tensors down to target_rank."""
    truncated = {}
    for key, tensor in w_state.items():
        if key.endswith("lora_A"):
            truncated[key] = tensor[: target_rank * 2, :]
        elif key.endswith("lora_B"):
            target_width = target_rank
            new_tensor = torch.zeros((tensor.shape[0], target_width), device=tensor.device, dtype=tensor.dtype)
            copy_width = min(tensor.shape[1], target_width)
            new_tensor[:, : copy_width] = tensor[:, : copy_width]
            truncated[key] = new_tensor
        else:
            truncated[key] = tensor
    return truncated

def fed_avg(w, args, config, client_ranks):
    method = args.agg_method
    if len(w) != len(client_ranks):
        raise ValueError("Number of local LoRA states does not match number of client ranks.")

    # 1. Standard FedAvg / Freeze / Fedit
    if method in ['avg', 'freeze', 'fedit']:
        max_rank = max(client_ranks)
        padded_states = [pad_lora_state_to_rank(client_w, max_rank) for client_w in w]

        w_avg = copy.deepcopy(padded_states[0])
        for k in w_avg.keys():
            for i in range(1, len(padded_states)):
                w_avg[k] += padded_states[i][k]
            w_avg[k] = torch.div(w_avg[k], len(padded_states))

        per_client = [
            truncate_lora_state_to_rank(copy.deepcopy(w_avg), client_rank)
            for client_rank in client_ranks
        ]
        return {
            "global_max": w_avg,
            "per_client": per_client,
            "stack_rank": max_rank,
        }

    # 2. Stacking
    if method == 'stack':
        n_embd = config.n_embd
        n_clients = len(w)
        combined_w = {}
        stack_rank = sum(client_ranks)

        # Identify unique layer prefixes
        layer_prefixes = set()
        for key in w[0].keys():
            # specifically look for attn.c_attn to cleanly strip it
            if key.endswith('.attn.c_attn.lora_A'):
                layer_prefixes.add(key.replace('.attn.c_attn.lora_A', ''))

        for layer_prefix in layer_prefixes:
            A_q_list, B_q_list = [], []
            A_v_list, B_v_list = [], []

            for k in range(n_clients):
                cleaned_w = extract_qv_lora_weights(w[k], n_embd)

                # Scale A by 1/N to simulate averaging
                #scale_factor = 1.0 / n_clients

                # Stack A (Scaled)
                A_q_list.append(cleaned_w[f"{layer_prefix}.attn.q_proj.lora_A"])
                A_v_list.append(cleaned_w[f"{layer_prefix}.attn.v_proj.lora_A"])

                # Stack B (Unscaled)
                B_q_list.append(cleaned_w[f"{layer_prefix}.attn.q_proj.lora_B"])
                B_v_list.append(cleaned_w[f"{layer_prefix}.attn.v_proj.lora_B"])

            # Stack A vertically (dim 0) -> Shape (N*r, n_embd)
            A_q_stack = torch.cat(A_q_list, dim=0)
            A_v_stack = torch.cat(A_v_list, dim=0)

            # Stack B horizontally (dim 1) -> Shape (n_embd, N*r)
            B_q_stack = torch.cat(B_q_list, dim=1)
            B_v_stack = torch.cat(B_v_list, dim=1)

            # --- RECONSTRUCTION ---

            # 1. lora_A: Simple concatenation of Q and V parts
            # Shape: (2 * N * r, n_embd)
            combined_w[f"{layer_prefix}.attn.c_attn.lora_A"] = torch.cat([A_q_stack, A_v_stack], dim=0)

            # 2. lora_B: Block Diagonal Construction
            # Shape: (3 * n_embd, 2 * N * r)
            # We must ensure Q_inputs affect Q_outputs and V_inputs affect V_outputs.

            total_rank_q = B_q_stack.shape[1] # N * r
            total_rank_v = B_v_stack.shape[1] # N * r
            total_cols = total_rank_q + total_rank_v
            # DEBUG: Check dimensions
            if layer_prefix == "h.0":
                 print(f"DEBUG: avg B Matrix: Q_width={total_rank_q}, V_width={total_rank_v}")
                 if total_rank_v == 0:
                     raise RuntimeError("Value Projection B-Matrix is empty! Check extraction logic.")

            B_full = torch.zeros(2 * n_embd, total_cols, device=B_q_stack.device, dtype=B_q_stack.dtype)

            # Place B_q in Top-Left (Row: 0 to n_embd, Col: 0 to Nr)
            B_full[0:n_embd, 0:total_rank_q] = B_q_stack

            # Place B_v in Bottom-Right (Row: 2n_embd to 3n_embd, Col: Nr to 2Nr)
            B_full[n_embd:2*n_embd, total_rank_q:total_cols] = B_v_stack

            combined_w[f"{layer_prefix}.attn.c_attn.lora_B"] = B_full

        # Return stacked adapters and the effective stacked rank for scaling
        return {
            "stacked": combined_w,
            "stack_rank": stack_rank,
        }
    # 2. SVD-based Aggregation
    if method == 'svd':
        n_embd = config.n_embd
        aggregated_BA = {}
        n_clients = len(w)
        weight_per_client = 1.0 / n_clients
        max_rank = max(client_ranks)

        # Step A: Compute B @ A
        for k in range(n_clients):
            cleaned_w = extract_qv_lora_weights(w[k], n_embd)
            for key in cleaned_w:
                if key.endswith('lora_A'):
                    B_key = key.replace('lora_A', 'lora_B')
                    BA = torch.matmul(cleaned_w[B_key], cleaned_w[key])
                    weighted_BA = BA * weight_per_client
                    if key not in aggregated_BA:
                        aggregated_BA[key] = weighted_BA
                    else:
                        aggregated_BA[key] += weighted_BA

        # Step B: SVD and Reconstruct
        svd_cache = {}
        for key, BA in aggregated_BA.items():
            U, S, Vh = svd(BA, full_matrices=False)
            svd_cache[key] = (U, S, Vh)

        def build_combined_for_rank(target_rank):
            aggregated_weights = {}
            for key, (U, S, Vh) in svd_cache.items():
                r_use = min(target_rank, U.shape[1], Vh.shape[0], S.shape[0])
                U_k = U[:, :r_use]
                S_k = torch.diag(S[:r_use])
                V_k = Vh[:r_use, :]

                # Reconstruct (Assign Singular values to B)
                B_new = U_k @ S_k
                A_new = V_k
                aggregated_weights[key] = A_new
                aggregated_weights[key.replace('lora_A', 'lora_B')] = B_new

            combined_w = {}
            for key in aggregated_weights:
                if 'q_proj' in key and key.endswith('lora_A'):
                    layer_prefix = ".".join(key.split(".")[:2])
                    A_q = aggregated_weights[key]
                    A_v = aggregated_weights[key.replace('q_proj', 'v_proj')]
                    combined_w[f"{layer_prefix}.attn.c_attn.lora_A"] = torch.cat([A_q, A_v], dim=0)
                    B_q = aggregated_weights[key.replace('lora_A', 'lora_B')]
                    B_v = aggregated_weights[key.replace('q_proj', 'v_proj').replace('lora_A', 'lora_B')]
                    combined_w[f"{layer_prefix}.attn.c_attn.lora_B"] = torch.cat([B_q, B_v], dim=0)
            return combined_w

        global_max = build_combined_for_rank(max_rank)
        per_client = [build_combined_for_rank(rk) for rk in client_ranks]

        return {
            "global_max": global_max,
            "per_client": per_client,
            "stack_rank": max_rank,
        }

    return {"global_max": w[0], "per_client": [w[0] for _ in client_ranks], "stack_rank": client_ranks[0]}


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(
    _loss,
    optimizer_server,
    model_server,
    optimizer_client,
    _schedule,
    client_hidden_states,
    hidden_states,
    args,
    is_update=True,
):
    if args.fp16:
        with amp.scale_loss(_loss, optimizer_server) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    dfx_client = client_hidden_states.grad.clone().detach()

    if is_update and args.clip > 0:
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer_server), args.clip
            )
        else:
            torch.nn.utils.clip_grad_norm_(model_server.parameters(), args.clip)
    optimizer_server.step()
    optimizer_server.zero_grad()

    if _schedule is not None:
        _schedule.step()

    hidden_states.backward(dfx_client)
    optimizer_client.step()
    optimizer_client.zero_grad()


def evaluate(model_client, model_server, valid_loader,args):
    model_client.eval()
    model_server.eval()
    device = torch.device("cuda")
    model_server = model_server.to(device)

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value.to(device) for key, value in data.items()}

            _input = data["input"]
            _target = data["target"]
            _msk = data["mask"]

            hidden_states, presents, _ = model_client(_input)

            _, _loss = model_server(
                _input.shape, hidden_states, presents, lm_labels=_target, lm_mask=_msk
            )
            loss = _loss.mean()

            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print("eval samples:", idx, "loss:", loss.float())

        print("average loss", avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)

def federated_merge_gpt2(model, w_stacked_dict, alpha, r_target=None):
    """
    Implements 'Merge and Reset' strategy.
    Handles 'Packed' Delta W (Query+Value) merging into 'Full' Base Weights (Query+Key+Value).
    AUTO-DETECTS Conv1D (Transposed) vs Linear weight shapes.
    """
    with torch.no_grad():
        merged_cnt = 0
        missing_cnt = 0

        for name, module in model.named_modules():
            if not isinstance(module, lora.MergedLinear):
                continue

            key_A = f"{name}.lora_A"
            key_B = f"{name}.lora_B"

            if key_A not in w_stacked_dict or key_B not in w_stacked_dict:
                missing_cnt += 1
                continue

            merged_cnt += 1
            A_stack = w_stacked_dict[key_A].to(module.weight.device)
            B_stack = w_stacked_dict[key_B].to(module.weight.device)
            inferred_rank = A_stack.shape[0] // 2
            rank_for_scale = r_target if r_target is not None else inferred_rank
            if rank_for_scale == 0:
                continue
            scale = alpha / rank_for_scale
            # print(f"RANK: {rank_for_scale}")
            # Compute Delta. Shape: (2*n_embd, n_embd) -> Packed [Query; Value] by Input
            delta_w = (B_stack @ A_stack) * scale

            # Calculate embedding dim based on the delta
            # delta_w is (Packed_Out, In). So In = n_embd.
            n_embd = delta_w.shape[1]

            # --- CASE 1: Conv1D (Standard GPT-2) ---
            # Shape is (In, Out_Full) -> (n_embd, 3*n_embd)
            if module.weight.shape[1] == 3 * n_embd:
                # delta_w is (2 * n_embd, n_embd), need to transpose to (n_embd, 2 * n_embd)
                dW = delta_w.T
                module.weight.data[:, :n_embd] += dW[:, :n_embd] # Update Q
                module.weight.data[:, 2*n_embd:] += dW[:, n_embd:] # Update V (skip K)

            # --- CASE 2: Linear (Standard PyTorch) ---
            # Shape is (Out_Full, In) -> (3*n_embd, n_embd)
            elif module.weight.shape == (3 * n_embd, n_embd):
                # 1. Add Query (Rows 0 to n_embd)
                module.weight.data[0:n_embd, :] += delta_w[0:n_embd, :]

                # 2. Add Value (Rows 2*n_embd to 3*n_embd) -> SKIPPING KEYS
                module.weight.data[2*n_embd:3*n_embd, :] += delta_w[n_embd:2*n_embd, :]

            else:
                print(f"DEBUG: Skipping {name} due to shape mismatch.")
                print(f"Weight: {module.weight.shape}, Delta: {delta_w.shape}, n_embd: {n_embd}")

            # Reset LoRA Layers to low rank
            # Reset ONLY LoRA params
            if hasattr(module, "lora_A"):
                torch.nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                torch.nn.init.zeros_(module.lora_B)

            # also ensure it's in "unmerged" state
            if hasattr(module, "merged"):
                module.merged = False


if __name__ == "__main__":
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception:
            warnings.warn("Could not import amp, apex may not be installed")

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    if args.lora_ranks:
        client_ranks = [int(r.strip()) for r in args.lora_ranks.split(",") if r.strip()]
    else:
        client_ranks = [args.lora_dim for _ in range(args.num_clients)]
    num_clients = len(client_ranks)
    max_client_rank = max(client_ranks) if len(client_ranks) > 0 else args.lora_dim
    args.lora_dim = max_client_rank  # use the highest rank as the global adapter dimension
    args.num_clients = num_clients
    args.client_ranks = client_ranks

    # Create train dataset and valid dataset
    train_data0 = FT_Dataset(
        args.train_data0,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )

    train_data1 = FT_Dataset(
        args.train_data1,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )

    train_data2 = FT_Dataset(
        args.train_data2,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )

    valid_data = FT_Dataset(
        args.valid_data,
        args.valid_batch_size,
        args.seq_len,
    )

    train_loader0 = DataLoader(
        train_data0,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data0, seed=args.random_seed, shuffle=True
        ),
    )

    train_loader1 = DataLoader(
        train_data1,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data1, seed=args.random_seed, shuffle=True
        ),
    )

    train_loader2 = DataLoader(
        train_data2,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data2, seed=args.random_seed, shuffle=True
        ),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.valid_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            valid_data, seed=args.random_seed
        ),
    )

    if args.model_card == "gpt2.sm":
        config = GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            cut_layer=args.cut_layer,
        )
    elif args.model_card == "gpt2.md":
        config = GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            cut_layer=args.cut_layer,
        )
    elif args.model_card == "gpt2.lg":
        config = GPT2Config(
            n_embd=1280,
            n_layer=36,
            n_head=20,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            cut_layer=args.cut_layer,
        )

    lm_net_Client = GPT2LMModel_Client(copy.deepcopy(config))
    lm_net_Server = GPT2LMModel_Server(copy.deepcopy(config))

    state_dict = torch.load(args.init_checkpoint)
    if args.init_checkpoint is not None:
        print("loading model pretrained weight.")
        lm_net_Client.load_weight(state_dict)
        lm_net_Server.load_weight(state_dict)

    lm_net_Client = lm_net_Client.cuda()
    lm_net_Server = lm_net_Server.cuda()

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net_Client)
        lora.mark_only_lora_as_trainable(lm_net_Server)
        if args.agg_method == 'freeze':
             print("DEBUG: Applying Freeze-A to Global Models")
             init_and_freeze_lora_A(lm_net_Client, seed=42)
             init_and_freeze_lora_A(lm_net_Server, seed=42)

    optimizer_Client = create_adam_optimizer_from_args(lm_net_Client, args)
    optimizer_Server = create_adam_optimizer_from_args(lm_net_Server, args)

    # nums of clients:
    client_models = []
    optimizers = []

    # Create client models for different clients
    for i, client_rank in enumerate(client_ranks):
        client_config = copy.deepcopy(config)
        client_config.lora_attn_dim = client_rank
        client_model = GPT2LMModel_Client(client_config)
        client_model.load_weight(state_dict)
        client_model = client_model.cuda()
        if client_rank > 0:
            lora.mark_only_lora_as_trainable(client_model)
            if args.agg_method == 'freeze':
                print(f"DEBUG: Applying Freeze-A to Client {i}")
                init_and_freeze_lora_A(client_model, seed=42)
        optimizer = create_adam_optimizer_from_args(client_model, args)
        client_models.append(client_model)
        optimizers.append(optimizer)

    if args.max_step is None:
        args.max_step = (
            args.max_epoch * train_data0.num_batches * num_clients + args.world_size - 1
        ) // args.world_size
        print("set max_step:", args.max_step)

    scheduler_Client = create_optimizer_scheduler(optimizer_Client, args)
    scheduler_Server = create_optimizer_scheduler(optimizer_Server, args)

    if args.fp16:
        lm_net_Client, optimizer_Client = amp.initialize(
            lm_net_Client, optimizer_Client, opt_level="O1"
        )
        lm_net_Server, optimizer_Server = amp.initialize(
            lm_net_Server, optimizer_Server, opt_level="O1"
        )
    lm_net_Client, optimizer_Client = distributed_opt(
        args, lm_net_Client, optimizer_Client, grad_acc=args.grad_acc
    )
    lm_net_Server, optimizer_Server = distributed_opt(
        args, lm_net_Server, optimizer_Server, grad_acc=args.grad_acc
    )

    log_list = []

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            model_client = lm_net_Client
            model_server = lm_net_Server
            optimizer_server = optimizer_Server
            scheduler_server = scheduler_Server
            model_client.train()
            model_server.train()
            # Meter to average language model loss
            avg_lm_loss = AverageMeter()
            print("start to train the model................", epoch)
            log_start_time = time.time()

            # Meter to average language model loss
            best_val_ppl = None

            device = torch.device("cuda")
            if num_clients != len(client_ranks):
                raise ValueError(
                    f"num_clients ({num_clients}) must match length of client_ranks ({len(client_ranks)})."
                )
            for loader in [train_loader0, train_loader1, train_loader2]:
                loader.sampler.set_epoch(epoch)

            # Initialize global client model
            net_glob_client = GPT2LMModel_Client(config)
            net_glob_client = net_glob_client.to(device)

            # Load weights to global client model
            net_glob_client.load_weight(state_dict)
            if args.lora_dim > 0:
                lora.mark_only_lora_as_trainable(net_glob_client)
                if args.agg_method == "freeze":
                    init_and_freeze_lora_A(net_glob_client, seed=42)
            net_glob_client.train()

            # For Stacking, use the persistent global client passed in; otherwise use net_glob_client.
            if args.agg_method == "stack":
                # global_client_model = model_client.to(device)
                if epoch == 1:
                    global_client_model = net_glob_client
                    if args.lora_dim > 0:
                        lora.mark_only_lora_as_trainable(global_client_model)
                    global_client_model.train()
            else:
                global_client_model = net_glob_client
            w_glob_client = global_client_model.state_dict()


            # aggregate every 100 train_step
            aggregate_step = 100

            w_locals_client = []

            if num_clients > 3:
                raise ValueError("num_clients larger than available loaders is not supported")
            train_loaders = [train_loader0, train_loader1, train_loader2][:num_clients]

            # get train data from different client train dataset
            for idx, data in enumerate(zip(*train_loaders)):
                # The client interacts with the server in turn
                for i in range(num_clients):
                    client_data = {key: value.to(device) for key, value in data[i].items()}

                    _input = client_data["input"]
                    _target = client_data["target"]
                    _msk = client_data["mask"]

                    client_models[i].train()

                    _input = _input.to(device)

                    hidden_states, presents, w_client = client_models[i](_input)
                    train_step += 1

                    if (train_step + num_clients) % aggregate_step < num_clients:
                        w_locals_client.append(copy.deepcopy(w_client))

                    client_hidden_states = hidden_states.clone().detach().requires_grad_(True)

                    _, _lm_loss = model_server(
                        _input.shape,
                        client_hidden_states,
                        presents,
                        lm_labels=_target,
                        lm_mask=_msk,
                        label_smooth=args.label_smooth,
                    )

                    _lm_loss = _lm_loss.mean()

                    is_update = train_step % args.grad_acc == 0
                    avg_lm_loss.update(_lm_loss.item())

                    optimizer_step(
                        _lm_loss / args.grad_acc,
                        optimizer_server,
                        model_server,
                        optimizers[i],
                        scheduler_server,
                        client_hidden_states,
                        hidden_states,
                        args,
                        is_update=is_update,
                    )

                    # aggregate client LoRA model every 100 train_step
                    if train_step % aggregate_step == 0:
                        print(f"DEBUG: Aggregating using method: {args.agg_method}")
                        print(f"\n=== DEBUG: AGGREGATION STEP {train_step} ===")

                        # 1. Extract LoRA states
                        w_locals_client_lora = [
                            extract_lora_state(copy.deepcopy(w_client))
                            for w_client in w_locals_client
                        ]

                        # --- DEBUG CHECK 1: RAW CLIENT WEIGHTS ---
                        # Check the first layer of the first client (Rank 2 or 4)
                        client0 = w_locals_client_lora[0]
                        print("DEBUG: Checking Client 0 RAW Weights (Before Padding)")
                        for k, v in client0.items():
                            if "h.0.attn.c_attn.lora_A" in k:
                                debug_lora_integrity("CLIENT_0_RAW", k, v)

                        # --- DEBUG CHECK 2: PADDED WEIGHTS ---
                        # Manually pad to check if our Fix in Change #2 worked
                        padded_client0 = pad_lora_state_to_rank(client0, args.lora_dim)
                        print(
                            f"DEBUG: Checking Client 0 PADDED Weights (Target Rank {args.lora_dim})"
                        )
                        for k, v in padded_client0.items():
                            if "h.0.attn.c_attn.lora_A" in k:
                                debug_lora_integrity("CLIENT_0_PADDED", k, v)

                        # Proceed with aggregation
                        agg_result = fed_avg(w_locals_client_lora, args, config, client_ranks)

                        # ... rest of the logic ...

                        # --- STACKING SPECIFIC LOGIC ---
                        if args.agg_method == "stack":
                            stack_rank = agg_result["stack_rank"]
                            w_glob_client_lora_prefixed = prefix_lora_keys(
                                agg_result["stacked"], "transformer_Client."
                            )

                            print(
                                f"DEBUG:Merge and Reset at step {train_step} (stack_rank={stack_rank})"
                            )

                            # 1. Merge into Global Model
                            federated_merge_gpt2(
                                global_client_model,
                                w_glob_client_lora_prefixed,
                                args.lora_alpha,
                                stack_rank,
                            )

                            # 2. Merge into Client Models
                            for client_model in client_models:
                                federated_merge_gpt2(
                                    client_model,
                                    w_glob_client_lora_prefixed,
                                    args.lora_alpha,
                                    stack_rank,
                                )

                            # Update the global weights dict to reflect the merged base model
                            w_glob_client = global_client_model.state_dict()

                            # Do NOT load_state_dict the huge LoRA matrices.
                            # The models are already updated via the merge function.

                        # --- STANDARD LOGIC (FedAvg/Freeze/SVD) ---
                        else:
                            w_glob_client_lora_new = prefix_lora_keys(
                                agg_result["global_max"], "transformer_Client."
                            )
                            w_glob_client = update_state_with_lora(
                                w_glob_client, w_glob_client_lora_new
                            )

                            global_client_model.load_state_dict(w_glob_client, strict=False)
                            w_glob_client = global_client_model.state_dict()

                            # Broadcast rank-specific adapters to each client
                            for client_model, client_lora_state in zip(
                                client_models, agg_result["per_client"]
                            ):
                                client_prefixed = prefix_lora_keys(
                                    client_lora_state, "transformer_Client."
                                )
                                client_sd = update_state_with_lora(
                                    client_model.state_dict(), client_prefixed
                                )
                                client_model.load_state_dict(client_sd, strict=False)

                        w_locals_client = []

                    # Output the training process data
                    if train_step % args.log_interval == 0:
                        elapsed = time.time() - log_start_time
                        lr = optimizer_server.param_groups[0]["lr"]
                        log_str = (
                            f"| epoch {epoch:3d} step {train_step:>8d} | {idx*num_clients + 1:>6d} batches | "
                            f"lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | "
                            f"loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | "
                            f"ppl {math.exp(avg_lm_loss.avg):5.2f}"
                        )

                        log_list.append(log_str)

                        if args.rank == 0:
                            print(log_str)
                        log_start_time = time.time()
                        avg_lm_loss.reset()

                    # save checkpoint at each save_interval
                    if train_step % args.save_interval == 0:
                        save_checkpoint(
                            w_glob_client, model_server, args, train_step, num_clients
                        )
                    distributed_sync(args)

                    if train_step % args.eval_interval == 0:
                        eval_start_time = time.time()

                        valid_loss, valid_ppl = evaluate(
                            global_client_model, model_server, valid_loader, args
                        )
                        if best_val_ppl is None or valid_ppl < best_val_ppl:
                            best_val_ppl = valid_ppl

                        log_str = (
                            f"| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | "
                            f"time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | "
                            f"valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} "
                        )
                        log_list.append(log_str)

                        if args.rank == 0:
                            print("-" * 100)
                            print(log_str)
                            print("-" * 100)

                        # Reset to train mode after evaluation
                        global_client_model.train()
                        model_server.train()
                        distributed_sync(args)

                    # Save training process
                    if train_step == args.max_step:
                        df = pd.DataFrame(log_list, columns=["Log"])
                        convergence_dir = (
                            "/projects/bewi/smurugaiyan1/SplitFM-main/SplitLoRA/examples/convergence_sheets"
                        )
                        os.makedirs(convergence_dir, exist_ok=True)

                        # Create the full path with the aggregation method in the filename
                        ranks_str = (
                            args.lora_ranks.replace(",", "_")
                            if args.lora_ranks
                            else str(args.lora_dim)
                        )
                        filename = (
                            f"{args.model_card} ranks={ranks_str} rank_max={args.lora_dim} "
                            f"num={num_clients} block={args.cut_layer} agg={args.agg_method} "
                            f"seed={args.random_seed}.xlsx"
                        )

                        full_path = os.path.join(convergence_dir, filename)

                        print(f"Saving convergence sheet to: {full_path}")
                        df.to_excel(
                            full_path,
                            sheet_name="Sheet1",
                            index=False,
                        )
                        break

            # Save the final checkpoint
            if train_step == args.max_step:
                save_checkpoint(w_glob_client, model_server, args, train_step, num_clients)
            distributed_sync(args)

            if train_step >= args.max_step or (
                args.max_epoch is not None and epoch >= args.max_epoch
            ):
                if args.rank == 0:
                    print("-" * 100)
                    print("End of training")
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print("-" * 100)
            print("Exiting from training early")

    distributed_sync(args)
    print("cleanup dist ...")
    cleanup(args)
