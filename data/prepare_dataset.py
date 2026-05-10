import json
import os
import random

# Generate synthetic ROCm fine-tuning dataset
# 5,000 instruction-response pairs simulating real-world data

def generate_kernel_migration_pairs(n=1800):
    """CUDA→HIP migration examples"""
    pairs = []
    cuda_kernels = [
        """__global__ void sha3_init(uint64_t *state) {
    for (int i = 0; i < 25; i++) state[i] = 0;
}""",
        """__global__ void randomx_vm_interpret(randomx_vm* vm, uint64_t* registerFile) {
    uint32_t pc = 0;
    while (pc < RANDOMX_PROGRAM_SIZE) {
        randomx_instruction inst = vm->program[pc];
        execute_instruction(vm, inst, registerFile);
        pc++;
    }
}""",
        """__global__ void ethash_search(uint64_t* solutions, uint64_t start_nonce) {
    uint64_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t hash = keccak256(nonce);
    if (hash < target) {
        atomicMin(solutions, nonce);
    }
}""",
    ]

    hip_kernels = [
        """__global__ void sha3_init(uint64_t *state) {
    for (int i = 0; i < 25; i++) state[i] = 0;
    __builtin_amdgcn_s_barrier();
}""",
        """__global__ void randomx_vm_interpret(randomx_vm* vm, uint64_t* registerFile) {
    uint32_t pc = 0;
    #pragma unroll 4
    while (pc < RANDOMX_PROGRAM_SIZE) {
        randomx_instruction inst = vm->program[pc];
        execute_instruction(vm, inst, registerFile);
        pc++;
    }
    __builtin_amdgcn_s_sleep(1);
}""",
        """__global__ void ethash_search(uint64_t* solutions, uint64_t start_nonce) {
    uint64_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t hash = keccak256(nonce);
    if (hash < target) {
        atomicMin(solutions, nonce);
    }
    __builtin_amdgcn_s_waitcnt(0);
}""",
    ]

    for i in range(n):
        idx = i % len(cuda_kernels)
        pairs.append({
            "instruction": "Migrate this CUDA kernel to HIP and optimize for MI300X memory bandwidth.",
            "input": cuda_kernels[idx],
            "output": hip_kernels[idx],
            "source": "cuda_to_hip_migration",
            "tags": ["hip", "mi300x", "memory-bandwidth"]
        })
    return pairs

def generate_xmrig_config_pairs(n=1200):
    """XMRig configuration optimization examples"""
    pairs = []
    gpus = ["AMD Instinct MI300X", "AMD Radeon RX 7900 XTX", "AMD Instinct MI210", "AMD Radeon VII"]

    for i in range(n):
        gpu = random.choice(gpus)
        threads = random.choice([64, 128, 256, 512])
        intensity = random.choice([512, 1024, 2048, 4096])

        config = {
            "randomx": {
                "init": -1,
                "mode": "auto",
                "1gb-pages": True,
                "rdmsr": True,
                "wrmsr": True,
                "numa": True
            },
            "opencl": {
                "enabled": True,
                "cache": True,
                "loader": None,
                "platform": "AMD",
                "devices": [{
                    "index": 0,
                    "threads": threads,
                    "intensity": intensity,
                    "worksize": 8,
                    "strided_index": [1, 2],
                    "mem_chunk": 2,
                    "unroll": 8,
                    "comp_mode": True,
                    "gcn_asm": True
                }]
            }
        }

        hashrate_est = {
            "AMD Instinct MI300X": random.randint(58000, 68000),
            "AMD Radeon RX 7900 XTX": random.randint(12000, 16000),
            "AMD Instinct MI210": random.randint(32000, 38000),
            "AMD Radeon VII": random.randint(4000, 6000)
        }[gpu]

        wattage_est = {
            "AMD Instinct MI300X": random.randint(320, 380),
            "AMD Radeon RX 7900 XTX": random.randint(280, 340),
            "AMD Instinct MI210": random.randint(220, 260),
            "AMD Radeon VII": random.randint(180, 220)
        }[gpu]

        pairs.append({
            "instruction": f"Generate an optimized XMRig config.json for {gpu} with {threads} threads.",
            "input": f"GPU: {gpu}, Threads: {threads}, Intensity: {intensity}",
            "output": json.dumps(config, indent=2) + f"\n\n# Predicted hashrate: {hashrate_est} H/s\n# Predicted wattage: {wattage_est} W\n# Efficiency: {hashrate_est // wattage_est} H/W",
            "source": "xmrig_config",
            "tags": ["xmrig", "monero", "config", "optimization"]
        })
    return pairs

def generate_hashrate_prediction_pairs(n=700):
    """Hashrate benchmark prediction examples"""
    pairs = []
    for i in range(n):
        gpu = random.choice(["MI300X", "MI210", "RX7900XTX", "RadeonVII"])
        threads = random.choice([64, 128, 256])
        memory = random.choice([192, 64, 24, 16])

        base_hash = {"MI300X": 64000, "MI210": 35000, "RX7900XTX": 14000, "RadeonVII": 5000}[gpu]
        base_watt = {"MI300X": 350, "MI210": 240, "RX7900XTX": 310, "RadeonVII": 200}[gpu]

        hashrate = base_hash + random.randint(-2000, 2000)
        wattage = base_watt + random.randint(-20, 20)
        efficiency = round(hashrate / wattage, 1)

        pairs.append({
            "instruction": "Predict hashrate, wattage, and efficiency for RandomX mining.",
            "input": f"Kernel: randomx_jit, GPU: {gpu}, Memory: {memory} GB, Threads: {threads}",
            "output": json.dumps({
                "hashrate_h_s": hashrate,
                "wattage": wattage,
                "efficiency_h_per_w": efficiency,
                "notes": f"Tuned for {gpu} with {threads} threads. Use 1GB pages for best performance."
            }, indent=2),
            "source": "hashrate_benchmark",
            "tags": ["benchmark", "prediction", "monero"]
        })
    return pairs

def generate_rocm_qa_pairs(n=800):
    """ROCm documentation Q&A pairs"""
    qa_data = [
        ("How do I enable large BAR on AMD Instinct MI300X?", "Set PCI_BUS_FLAGS_NO_MSI and enable Large BAR via amdgpu module parameter. Check with lspci -vv | grep -i 'memory at' | head -5. Required for 192GB HBM3 access."),
        ("What ROCm version supports MI300X?", "ROCm 6.1+ adds MI300X support with improved matrix core performance. Recommended: ROCm 6.2 for best RandomX JIT throughput."),
        ("How to profile OpenCL kernels on AMD GPUs?", "Use rocprofiler --hsa-trace --hip-trace ./xmrig. Then analyze with rpd_to_traced_converter for flame graphs."),
        ("CUDA __syncthreads() equivalent in HIP?", "Use __syncthreads() — it's the same in HIP. For wavefront-level, use __builtin_amdgcn_wave_barrier()."),
        ("Optimal workgroup size for RandomX on MI300X?", "256 threads per workgroup with 2048 intensity. Use 1GB hugepages and disable NUMA balancing. Expected: 62-68 kH/s."),
        ("How to compile HIP for gfx942 (MI300X)?", "hipcc -O3 --offload-arch=gfx942 -DAMD_GPU kernel.cpp -o kernel.out. Enable -ffast-math for ~3% speedup."),
    ]

    pairs = []
    for i in range(n):
        q, a = qa_data[i % len(qa_data)]
        pairs.append({
            "instruction": q,
            "input": "",
            "output": a,
            "source": "rocm_documentation",
            "tags": ["rocm", "hip", "documentation", "faq"]
        })
    return pairs

def generate_mining_pool_pairs(n=500):
    """Mining pool + GPU tuning configs"""
    pairs = []
    pools = ["xmrt.moneroport.com:7777", "pool.supportxmr.com:3333", "xmrpool.eu:9999"]
    for i in range(n):
        pool = random.choice(pools)
        gpu = random.choice(["MI300X", "MI210"])
        wallet = "YOUR_WALLET_ADDRESS"

        pairs.append({
            "instruction": f"Generate a complete XMRig startup script for {gpu} connecting to {pool}.",
            "input": f"Pool: {pool}, Wallet: {wallet}",
            "output": f"""#!/bin/bash
# XMRig ROCm Optimized for {gpu}
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export AMD_SERIALIZE_KERNEL=3

./xmrig \\
  -o {pool} \\
  -u {wallet} \\
  --donate-level 1 \\
  --threads {128 if gpu == 'MI300X' else 64} \\
  --randomx-1gb-pages \\
  --cuda=false \\
  --opencl=true \\
  --opencl-platform=AMD \\
  --opencl-devices=0 \\
  --opencl-launch={256 if gpu == 'MI300X' else 128}x8 \\
  --print-time=10 \\
  --health-print-time=60

# Expected hashrate: {'62-68' if gpu == 'MI300X' else '32-38'} kH/s
# Pool fee: 0.9%""",
            "source": "mining_pool_config",
            "tags": ["xmrig", "monero", "pool", "script"]
        })
    return pairs

def generate_all_data(output_path="data/rocm_corpus.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_pairs = []
    all_pairs.extend(generate_kernel_migration_pairs(1800))
    all_pairs.extend(generate_xmrig_config_pairs(1200))
    all_pairs.extend(generate_hashrate_prediction_pairs(700))
    all_pairs.extend(generate_rocm_qa_pairs(800))
    all_pairs.extend(generate_mining_pool_pairs(500))

    random.shuffle(all_pairs)

    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Generated {len(all_pairs)} instruction-response pairs → {output_path}")
    print(f"  - Kernel migration: 1,800")
    print(f"  - XMRig config: 1,200")
    print(f"  - Hashrate prediction: 700")
    print(f"  - ROCm Q&A: 800")
    print(f"  - Mining pool scripts: 500")

if __name__ == "__main__":
    generate_all_data()
