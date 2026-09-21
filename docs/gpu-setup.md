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

The supplied full-fine-tuning profile fitted this card. The recorded historical matrix peaked at 3.748 GiB allocated on the full-tuning profile; available memory varies with other applications, so close GPU-heavy programmes before reproducing it. If a run fails, retain the failed record rather than changing the comparison in place.

To reproduce the historical fixed-profile comparison, use the installed matrix command. It keeps the split seed at 17, varies optimisation seeds 17, 23 and 41, and writes the paired comparison summary:

```powershell
.venv-gpu\Scripts\transformer-lab.exe run-sentiment-matrix --config configs\sentiment\baseline.toml --config configs\sentiment\head-only-gpt2-small.toml --config configs\sentiment\lora-gpt2-small.toml --config configs\sentiment\full-gpt2-small.toml --seeds 17 23 41 --device cuda --output reports\results\financial-phrasebank-comparison.json --resume
```

## Validation-selected v3 follow-up

The v3 protocol uses the same pinned dataset revision, duplicate-safe split and GPT-2 revision as the historical profile. It defines 12 validation-only candidates, selects one setting per method, and then evaluates the frozen choices on seeds 17, 23 and 41. Training candidates and final evaluation require the CUDA environment, prepared data and local GPT-2 artefacts. Selection itself can run in the CPU environment after candidate summaries exist.

Run the stages in this order:

```powershell
.venv-gpu\Scripts\python.exe scripts\run_v3_candidates.py --protocol configs\sentiment\v3-protocol.toml --device cuda --resume
.venv\Scripts\transformer-lab.exe select-sentiment --protocol configs\sentiment\v3-protocol.toml --output reports\results\v3\selection.json
.venv-gpu\Scripts\transformer-lab.exe evaluate-matrix --selection reports\results\v3\selection.json --seeds 17 23 41 --device cuda --output reports\results\v3\comparison.json --resume
```

The first and third commands write to their configured result paths and may overwrite only when the supplied evidence is valid and `--resume` or an explicit refresh policy permits it. Do not point them at a different experiment's artefacts. The selection command also writes its output, so use a temporary destination when checking the interface. For a read-only verification of the tracked v3 records, use the replay script described in the final audit notes rather than these writing commands. The checked v3 table is [`../reports/generated-results-v3.md`](../reports/generated-results-v3.md); it is exploratory because the final comparison reuses the inspected test partition.

The raw dataset and model checkpoints remain outside Git. Tracked result files contain the revision, split hashes, configuration, elapsed time, memory, metrics, and failure reason where applicable.
