# Unmanned active observation planning of near-surface ocean under tropical cyclone condition

Supplementary materials for the paper published in *Journal of Ocean Engineering and Science*:

> **Unmanned active observation planning of near-surface ocean under tropical cyclone condition**  
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S246801332600077X) · [DOI: 10.1016/j.joes.2026.05.012](https://doi.org/10.1016/j.joes.2026.05.012)

This repository provides **paper figures and training reward histories**. The proprietary planning / reinforcement-learning implementation (TD3 / PPO / SAC inner-loop agents, environment, and training scripts) is **not** included. For research collaboration or code access, please contact the corresponding author.

## Overview

We study multi-AUV swarm task planning for tropical-cyclone (TC) observation. The framework couples game-theoretic multi-agent assignment with continuous-action RL (SAC, PPO, TD3) in the inner loop, and is evaluated on TC Hinnamnor (2022) and six additional representative TC cases.

## Repository layout

```
figs/                  # Selected paper / experiment figures
rewardhistory/         # Per-TC cumulative reward curves (.npy)
pure_rewardhistory/    # Reward curves without exploration noise (.npy)
requirements.txt       # Minimal deps for loading / plotting arrays
```

### Tropical cyclone cases (index → name)

| Index | TC name    |
|------:|------------|
| 0     | Bebinca    |
| 1     | Yagi       |
| 2     | Doksuri    |
| 3     | Haikui     |
| 4     | Guchol     |
| 5     | Khanun     |
| 6     | Hinnamnor  |

### Figures

| File | Description |
|------|-------------|
| `figs/auv_movement.gif` | Example multi-AUV motion visualization |
| `figs/{0,1,3,5,6}-TD3threward.png` | TD3 reward histories for selected TCs |
| `figs/TD3threward_K=100.png` | TD3 reward with horizon / window \(K=100\) |
| `figs/TD3threward_K=1000.png` | TD3 reward with \(K=1000\) |

### Reward arrays

Each `第{i}次total_reward_history.npy` / `第{i}次total_pure_reward_history.npy` stores the episode-wise reward trajectory for TC case `i` (see table above).

```python
import numpy as np

r = np.load("rewardhistory/第6次total_reward_history.npy")  # Hinnamnor
print(r.shape)
```

## Citation

If you use these materials, please cite the JOES paper:

```bibtex
@article{JOES2026AUVTC,
  title   = {Unmanned active observation planning of near-surface ocean under tropical cyclone condition},
  journal = {Journal of Ocean Engineering and Science},
  year    = {2026},
  doi     = {10.1016/j.joes.2026.05.012},
  url     = {https://www.sciencedirect.com/science/article/pii/S246801332600077X}
}
```

## License / code availability

Supplementary figures and numerical reward histories are released for academic use with the paper. **Core algorithm source code is omitted** from this public repository and may be shared upon reasonable request for non-commercial research.
