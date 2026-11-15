# Free the GPU before LoRA training (D17): vLLM pre-allocates 80% of VRAM and holds it while running.
docker rm -f opsgym-vllm 2>$null | Out-Null
Start-Sleep -Seconds 2
$used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
Write-Host "vLLM stopped. GPU memory in use: $used MiB"
if ($used -gt 1500) { Write-Host "WARNING: something else is still holding GPU memory (check nvidia-smi)."; exit 1 }
