# GPU setup

The project runs correctly on CPU. GPU training is optional and uses the PyTorch build selected for the installed NVIDIA driver. The example laptop has a GeForce GTX 1650 with about 4 GB of VRAM, so the sentiment profiles use a micro-batch of one, sequence length 64, gradient accumulation, and mixed precision where it is stable.

## Verify the driver

Open PowerShell and run:

```powershell
nvidia-smi
```

The recorded run used the CUDA 12.8 wheel listed in PyTorch's official installation guidance. Keep it in a separate environment so the CPU verification setup remains unchanged:

```powershell
py -3.12 -m venv .venv-gpu
.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.venv-gpu\Scripts\python.exe -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.venv-gpu\Scripts\python.exe -m pip install -c requirements\constraints-gpu-cu128.txt -e ".[all]"
```

Verify the installation:

```powershell
.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.venv-gpu\Scripts\python.exe scripts\check_hardware.py
```

If `torch.cuda.is_available()` is false, keep the CPU environment. The experiment records will state that the GPT-2 sentiment runs were unavailable rather than relabelling CPU measurements as GPU measurements.

## GTX 1650 profile

Start with the supplied LoRA configuration. It uses GPT-2 small, sequence length 64, micro-batch 1, accumulation 16, rank 4, and alpha 8. Run one batch first and watch the reported peak allocation. If the run fits, continue with the same configuration and seed. If it fails with CUDA out of memory, keep the failed record and do not change the method without creating a new configuration.

```powershell
.venv-gpu\Scripts\transformer-lab.exe train-sentiment --config configs\sentiment\lora-gpt2-small.toml --device cuda
```

The supplied full-fine-tuning profile fitted this card. The recorded three-seed matrix peaked at 3.28 GB allocated on the full-tuning profile; available memory varies with other applications, so close GPU-heavy programs before reproducing it. If a run fails, retain the failed record rather than changing the comparison in place.

To reproduce the reported comparison, use the installed matrix command. It keeps the split seed at 17, varies optimisation seeds 17, 23 and 41, and writes the paired comparison summary:

```powershell
.venv-gpu\Scripts\transformer-lab.exe run-sentiment-matrix --config configs\sentiment\baseline.toml --config configs\sentiment\head-only-gpt2-small.toml --config configs\sentiment\lora-gpt2-small.toml --config configs\sentiment\full-gpt2-small.toml --seeds 17 23 41 --device cuda --output reports\results\financial-phrasebank-comparison.json --resume
```

The raw dataset and model checkpoints remain outside Git. Tracked result files contain the revision, split hashes, configuration, elapsed time, memory, metrics, and failure reason where applicable.
