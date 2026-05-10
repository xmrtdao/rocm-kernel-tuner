import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

class ROCmKernelTuner:
    def __init__(self, model_path="checkpoints/rocm-qwen-7b", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.device = device
        self.model.eval()

    def generate(self, prompt, max_length=2048, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def optimize_kernel(self, cuda_code, target="hip", gpu="mi300x"):
        prompt = f"""<|im_start|>system
You are ROCmKernelTuner, an AMD ROCm GPU optimization expert.
Convert the following CUDA kernel to optimized {target.upper()} for {gpu.upper()}.
Use __builtin_amdgcn intrinsics where applicable. Add hashrate estimates if relevant.<|im_end|>
<|im_start|>user
{cuda_code}<|im_end|>
<|im_start|>assistant
"""
        return {"optimized_code": self.generate(prompt)}

    def generate_config(self, gpu, pool, wallet, threads=128):
        prompt = f"""<|im_start|>system
You are ROCmKernelTuner. Generate an optimized XMRig config.json for {gpu} with {threads} threads.
Include predicted hashrate and wattage.<|im_end|>
<|im_start|>user
GPU: {gpu}
Pool: {pool}
Wallet: {wallet}
Threads: {threads}<|im_end|>
<|im_start|>assistant
"""
        return {"xmrig_config": self.generate(prompt)}

    def predict_hashrate(self, kernel, gpu, memory_gb, threads):
        prompt = f"""<|im_start|>system
Predict hashrate (H/s), wattage (W), and efficiency (H/W) for RandomX mining.
Be precise. Use known benchmarks for {gpu}.<|im_end|>
<|im_start|>user
Kernel: {kernel}
GPU: {gpu}
Memory: {memory_gb} GB
Threads: {threads}<|im_end|>
<|im_start|>assistant
"""
        text = self.generate(prompt)
        # Try to parse JSON-like output
        try:
            # Find first JSON block
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                result = json.loads(m.group())
                return result
        except Exception:
            pass
        return {"raw": text}


def main():
    parser = argparse.ArgumentParser(description="ROCm Kernel Tuner Inference")
    parser.add_argument("--model", default="checkpoints/rocm-qwen-7b", help="Path to fine-tuned model")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    tuner = ROCmKernelTuner(args.model)
    output = tuner.generate(args.prompt, max_length=args.max_length)
    print(output)


if __name__ == "__main__":
    main()
