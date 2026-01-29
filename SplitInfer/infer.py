import os
import sys
import argparse
import importlib.util

MODEL_REGISTRY = {
    # 1. Qwen2-VL
    "Qwen2-VL-7B-Instruct": {
        "folder": "Qwen2-VL-7B-Instruct",      # model path
        "adapter_class": "QwenAdapter"            # model adapter
    },

    # 2. DeepSeek-R1 (Distill Llama)
    "DeepSeek-R1-Distill-Llama-8B": {
        "folder": "DeepSeek-R1-Distill-Llama-8B",
        "adapter_class": "DeepSeekAdapter"          
    },

    # 3. Llama-3
    "Llama-3-8B-Instruct": {
        "folder": "Llama-3-8B-Instruct",
        "adapter_class": "LlamaAdapter"
    }
}

def load_adapter(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported models: {model_name}.")
    
    config = MODEL_REGISTRY[model_name]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(base_dir, "Models", config["folder"])
    adapter_path = os.path.join(code_dir, "adapter.py")
    weights_path = os.path.join(base_dir, "weights", model_name)

    spec = importlib.util.spec_from_file_location(f"{model_name}_adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{model_name}_adapter"] = module 
    spec.loader.exec_module(module)
    AdapterClass = getattr(module, config["adapter_class"])
    adapter = AdapterClass(model_name)
    
    return adapter, weights_path


def main():
    parser = argparse.ArgumentParser(description="SplitInfer Unified Interface")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--input_sentence", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Image path (for Qwen2-VL-7B-Instruct)")
    parser.add_argument("--gpu", type=str, default="0,1,2")
    args = parser.parse_args()

    # Set the visible CUDA devices (GPUs) for PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # load 
    adapter, weights_path = load_adapter(args.model_name)
    adapter.load(weights_path)
    
    # inference
    result = adapter.infer(args.input_sentence, image_path=args.image)
    print(result)

if __name__ == "__main__":
    main()