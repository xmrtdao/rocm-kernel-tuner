import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
ROCm-KernelTuner — Hugging Face Space Demo
Fine-tuned Qwen2.5-Coder-7B for ROCm optimization
"""

# Placeholder — in production this loads the fine-tuned checkpoint
MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

def generate_response(system, user_input, max_tokens=1024, temperature=0.7):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

system_prompts = {
    "kernel_migration": "You are ROCmKernelTuner, an AMD ROCm GPU optimization expert. Convert CUDA kernels to optimized HIP for MI300X. Use __builtin_amdgcn intrinsics where applicable.",
    "xmrig_config": "You are ROCmKernelTuner. Generate optimized XMRig config.json files for AMD GPUs with accurate hashrate and wattage predictions.",
    "hashrate_prediction": "You are ROCmKernelTuner. Predict Monero RandomX hashrate, wattage, and efficiency for AMD GPUs. Be precise and cite known benchmarks."
}

def migrate_kernel(cuda_code, target_gpu):
    system = system_prompts["kernel_migration"]
    user = f"Convert this CUDA kernel to HIP for {target_gpu}:\n{cuda_code}"
    return generate_response(system, user)

def generate_xmrig(gpu, threads, pool, wallet):
    system = system_prompts["xmrig_config"]
    user = f"GPU: {gpu}\nThreads: {threads}\nPool: {pool}\nWallet: {wallet}"
    return generate_response(system, user)

def predict_hashrate(gpu, threads, memory_gb):
    system = system_prompts["hashrate_prediction"]
    user = f"GPU: {gpu}\nThreads: {threads}\nMemory: {memory_gb} GB"
    return generate_response(system, user)

with gr.Blocks(title="ROCm Kernel Tuner — AMD Hackathon") as demo:
    gr.Markdown("""
    # ROCm-KernelTuner
    ## Fine-Tuned Code Model for AMD ROCm Optimization
    Built for the AMD Developer Hackathon — Track 2: Fine-Tuning on AMD GPUs.
    
    This model was fine-tuned on MI300X using PEFT LoRA on a curated dataset of:
    - CUDA→HIP kernel migrations
    - XMRig mining configs
    - ROCm documentation Q&A
    - Hashrate benchmark logs
    """)

    with gr.Tab("Kernel Migration"):
        with gr.Row():
            cuda_input = gr.Textbox(label="CUDA Kernel", lines=10, placeholder="Paste your CUDA kernel here...")
            target_gpu = gr.Dropdown(["AMD Instinct MI300X", "AMD Instinct MI210", "AMD Radeon RX 7900 XTX"], value="AMD Instinct MI300X", label="Target GPU")
        migrate_btn = gr.Button("Migrate to HIP", variant="primary")
        hip_output = gr.Textbox(label="Optimized HIP Kernel", lines=15)
        migrate_btn.click(migrate_kernel, inputs=[cuda_input, target_gpu], outputs=hip_output)

    with gr.Tab("XMRig Config Generator"):
        with gr.Row():
            gpu_sel = gr.Dropdown(["MI300X", "MI210", "RX7900XTX", "RadeonVII"], value="MI300X", label="GPU")
            threads_slider = gr.Slider(64, 512, value=128, step=64, label="Threads")
        pool_input = gr.Textbox(label="Mining Pool URL", value="xmrt.moneroport.com:7777")
        wallet_input = gr.Textbox(label="Wallet Address", value="YOUR_WALLET_ADDRESS")
        config_btn = gr.Button("Generate Config", variant="primary")
        config_output = gr.Textbox(label="XMRig Config + Predictions", lines=20)
        config_btn.click(generate_xmrig, inputs=[gpu_sel, threads_slider, pool_input, wallet_input], outputs=config_output)

    with gr.Tab("Hashrate Predictor"):
        with gr.Row():
            pred_gpu = gr.Dropdown(["MI300X", "MI210", "RX7900XTX", "RadeonVII"], value="MI300X", label="GPU")
            pred_threads = gr.Slider(64, 512, value=128, step=64, label="Threads")
            pred_mem = gr.Slider(8, 192, value=192, step=8, label="Memory (GB)")
        pred_btn = gr.Button("Predict Performance", variant="primary")
        pred_output = gr.Textbox(label="Predicted Performance", lines=8)
        pred_btn.click(predict_hashrate, inputs=[pred_gpu, pred_threads, pred_mem], outputs=pred_output)

if __name__ == "__main__":
    demo.launch()
