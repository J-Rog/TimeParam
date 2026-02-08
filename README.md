# Multi-Resolution End-to-End Deep Neural Network for Optimizing Latency-Accuracy Tradeoff in Autonomous Driving

This repository contains the code and experiment scripts for the paper:

**Multi-Resolution End-to-End Deep Neural Network for Optimizing Latency-Accuracy Tradeoff in Autonomous Driving**

The project is built on top of **PCLA (Pretrained CARLA Leaderboard Agents)** and keeps the CARLA evaluation stack needed to run autonomous driving agents in simulation.

## What This Repo Contains

- Paper-oriented training scripts:
  - `finetune_bn_from_json.py`
  - `finetune_resaware_from_json.py`
- WoR (World on Rails) evaluation runner:
  - `test_wor.py`
  - `run_experiments.py`
- PCLA infrastructure and agent integrations under `pcla_agents/`
- CARLA Leaderboard/Scenario Runner support code under `leaderboard_codes/` and `scenario_runner/`

## Compatibility

- OS: Ubuntu 22 (tested)
- Python: 3.8+
- CARLA: 0.9.16 (UE4) recommended for this repo
- GPU: CUDA-capable GPU strongly recommended

## 1. Setup

### 1.1 Prerequisites

1. Install the CARLA simulator (official binary or source build).
2. Ensure NVIDIA driver, CUDA runtime, and PyTorch are available.
3. Install Conda (or Mamba).

### 1.2 Environment Installation

```bash
git clone https://github.com/qtweng/ResAwareWoR.git
cd ResAwareWoR
conda env create -f environment.yml
conda activate PCLA
```

Optional: check your PyTorch/CUDA pair:

```bash
python pcla_functions/cuda.py
```

If your workflow needs `torch-scatter`, install the wheel matching your exact PyTorch and CUDA versions.

### 1.3 CARLA Python API Setup

`test_wor.py` expects CARLA under `~/CARLA_0.9.16` by default.  
If yours is elsewhere, set:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
```

For CARLA 0.9.16, install the bundled wheel:

```bash
cd dist
python -m pip install carla-0.9.16-cp38-cp38-linux_x86_64.whl
cd ..
```

## 2. Run Evaluation (WoR)

Start CARLA:

```bash
./CarlaUE4.sh -vulkan
```

Run a quick WoR test:

```bash
python test_wor.py --agent wor_nc --route sample_route.xml
```

Useful flags:

- `--agent wor_nc` or `--agent wor_lb`
- `--route path/to/route.xml`
- `--route-id 0`
- `--agent-config path/to/config.yaml`
- `--sweep-control-latencies 0.05 0.1 --sweep-log outputs/sweeps.csv`
- `--sweep-vehicle-density 0 5 20`
- `--sweep-pedestrian-density 0 10 40`

See full options:

```bash
python test_wor.py --help
```

Batch experiment runner:

```bash
python run_experiments.py --config experiments/full.yaml
```

## 3. WoR Dataset (Training)

For fine-tuning, use the WoR Rails dataset (converted from lmdb):

- Download: https://utexas.box.com/s/vuf439jafqvi8u4rc37sdx9xvbrn59z2
- Data format: per-trajectory folders with `data.json` and sensor files (RGB/semantic labels).
- In this repo, point `DATA_DIR` in `finetune_bn_from_json.py` and `finetune_resaware_from_json.py` to the dataset root.

## 4. Run Paper Fine-Tuning

The two paper training scripts are:

- `finetune_bn_from_json.py`
- `finetune_resaware_from_json.py`

Before running, edit script-level config values (at the top of each file), especially:

- `DATA_DIR`
- `CONFIG_PATH`
- `CHECKPOINT_PATH`
- `OUTPUT_PATH`

Then run:

```bash
python finetune_bn_from_json.py
python finetune_resaware_from_json.py
```

## 5. Notes

- WoR pretrained configs and weights are under `pcla_agents/wor_pretrained/`.
- Routes follow Leaderboard XML format (`sample_route.xml` is included).
- `sample.py` shows direct usage of the `PCLA` class in a CARLA loop.

## Acknowledgements

- PCLA: https://github.com/MasoudJTehrani/PCLA
- World on Rails: https://github.com/dotchen/WorldOnRails
- CARLA Leaderboard: https://leaderboard.carla.org

## Citation

If you use this repository, please cite:

- This paper: *Multi-Resolution End-to-End Deep Neural Network for Optimizing Latency-Accuracy Tradeoff in Autonomous Driving*
- PCLA (FSE 2025): https://dl.acm.org/doi/abs/10.1145/3696630.3728577
