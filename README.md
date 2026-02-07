# WoR runner for CARLA (PCLA-based)

This repo is a slimmed-down fork of PCLA focused on running the World on Rails (WoR) agent in CARLA. The primary entrypoint is `test_wor.py`, which runs WoR and reports leaderboard-style metrics.

## Setup

- Install CARLA (tested with 0.9.16) and ensure the Python API is available.
  - If your CARLA install is not at `~/CARLA_0.9.16`, set `CARLA_ROOT`.
  - Optional wheel install for 0.9.16:
    ```bash
    cd dist
    python -m pip install carla-0.9.16-cp38-cp38-linux_x86_64.whl
    ```
- Create the env:
  ```bash
  conda env create -f environment.yml
  conda activate PCLA
  ```

## Run

Start CARLA (WoR expects Vulkan):
```bash
./CarlaUE4.sh -vulkan
```

Then:
```bash
python test_wor.py
```

Common options:
- `--agent wor_nc` or `--agent wor_lb`
- `--route path/to/route.xml`
- `--town Town05`
- `--vehicle-density 20 --pedestrian-density 50`
- `--control-latency 0.05`
- `--sweep-control-latencies 0.05 0.1 --sweep-log outputs/sweeps.csv`

See `python test_wor.py --help` for the full list.

## Experiments

Run the full experiment suite with:
```bash
python run_experiments.py --config experiments/full.yaml
```

## Notes

- WoR weights and configs live in `pcla_agents/wor_pretrained`.
- Routes are Leaderboard-style XML. `sample_route.xml` is included for a quick sanity check.

## Acknowledgements

This work is based on the PCLA framework and the World on Rails agent.
- PCLA: https://github.com/MasoudJTehrani/PCLA
- World on Rails: https://github.com/dotchen/WorldOnRails
