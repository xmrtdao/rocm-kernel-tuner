// Supabase Edge Function: optimize-kernel
// CUDA/Rust kernel code -> ROCm HIP optimized version

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";

const LLM_API_URL = Deno.env.get("LLM_API_URL") || "http://localhost:8000/v1/chat/completions";
const MODEL = Deno.env.get("LLM_MODEL") || "checkpoints/rocm-qwen-7b";

serve(async (req) =>> {
  if (req.method !== "POST") return new Response(JSON.stringify({ error: "POST only" }), { status: 405 });

  try {
    const { kernel_code, target = "hip", gpu = "mi300x" } = await req.json();
    if (!kernel_code) return new Response(JSON.stringify({ error: "kernel_code required" }), { status: 400 });

    const systemPrompt = `You are ROCmKernelTuner, an AMD ROCm GPU optimization expert.\nConvert kernels to optimized HIP for the specified AMD GPU.\nUse __builtin_amdgcn intrinsics where applicable. Add hashrate estimates if relevant.`;
    const userPrompt = `Target: ${target}\nGPU: ${gpu}\n\nKernel:\n${kernel_code}`;

    const resp = await fetch(LLM_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ],
        temperature: 0.3,
        max_tokens: 2048
      })
    });

    if (!resp.ok) throw new Error(`LLM API ${resp.status}: ${await resp.text()}`);
    const llmData = await resp.json();
    const optimized = llmData.choices?.[0]?.message?.content || "";

    return new Response(JSON.stringify({
      optimized_code: optimized,
      target,
      gpu,
      original_length: kernel_code.length,
      optimized_length: optimized.length
    }), {
      status: 200,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }
    });
  } catch (e: any) {
    return new Response(JSON.stringify({ error: e.message }), { status: 500 });
  }
});
