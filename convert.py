import sys
import torch
import safetensors
import safetensors.torch

if len(sys.argv) != 3:
    print("Usage: convert.py <input.safetensors> <output.safetensors>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]
tensors = {}

with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        if tensor.dtype == torch.bfloat16:
            tensors[key] = tensor.to(torch.float16)
        else:
            tensors[key] = tensor

safetensors.torch.save_file(tensors, output_path)

print("Done")