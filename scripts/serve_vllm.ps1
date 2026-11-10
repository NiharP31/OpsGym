# Start the vLLM server for rollouts (D17). Stop it with scripts\stop_serving.ps1 before training.
# First run pulls the image (~10 GB) and the model (~3.5 GB, cached in your HF cache afterwards).
# Usage: .\scripts\serve_vllm.ps1 [-Model Qwen/Qwen3-4B-AWQ]   (must be a Qwen3 model: prompts use its template)
param([string]$Model = "Qwen/Qwen3-1.7B")
docker rm -f opsgym-vllm 2>$null | Out-Null   # a leftover container would hold the GPU and port

docker run -d --name opsgym-vllm --gpus all --ipc=host -p 8000:8000 `
  -v "$env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface" `
  vllm/vllm-openai:v0.11.0 `
  --model $Model `
  --gpu-memory-utilization 0.8 `
  --max-model-len 8192 `
  --swap-space 1   # GiB of CPU RAM for preempted KV blocks (default 4); the host ran out of memory (D31)
# Exit code, not $ErrorActionPreference: Windows PowerShell 5 treats any native stderr as fatal.
if ($LASTEXITCODE -ne 0) { Write-Host "docker run failed"; exit 1 }
# ponytail: no --enable-lora yet; add it in Phase 6 when there are adapters to serve.

Write-Host "Waiting for http://localhost:8000/health (model load takes a minute or two)..."
for ($i = 0; $i -lt 120; $i++) {
  try { Invoke-WebRequest -UseBasicParsing http://localhost:8000/health | Out-Null; Write-Host "vLLM is up."; exit 0 }
  catch { Start-Sleep -Seconds 5 }
}
Write-Host "vLLM did not come up in 10 minutes. Logs:"; docker logs --tail 40 opsgym-vllm; exit 1
