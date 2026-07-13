# Unmanned active observation planning of near-surface ocean under tropical cyclone condition

Supplementary materials for the paper published in *Journal of Ocean Engineering and Science*:

> **Unmanned active observation planning of near-surface ocean under tropical cyclone condition**  
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S246801332600077X) · [DOI: 10.1016/j.joes.2026.05.012](https://doi.org/10.1016/j.joes.2026.05.012)

## Overview

We study multi-AUV swarm task planning for tropical-cyclone (TC) observation. The framework couples game-theoretic multi-agent assignment with continuous-action RL (SAC, PPO, TD3) in the inner loop, and is evaluated on TC Hinnamnor (2022) and six additional representative TC cases.

## Repository layout

```
code/                  # Inner-loop RL algorithms (TD3 / PPO / SAC)
figs/                  # Selected experiment figures
rewardhistory/         # Per-TC cumulative reward curves (.npy)
pure_rewardhistory/    # Reward curves without exploration noise (.npy)
requirements.txt
```

### `code/` — algorithms only

| File | Description |
|------|-------------|
| `TD3.py` | Twin Delayed DDPG agent |
| `PPO.py` | Proximal Policy Optimization agent |
| `SAC.py` | Soft Actor-Critic agent |
| `networks.py` | Shared Actor / Critic / StatePredictor nets (used by TD3 & SAC) |
| `buffer.py` | Replay buffer (used by TD3 & SAC) |

The **main training / planning program** (environment, multi-agent game loop, experiment scripts) is **not** included. For collaboration or full-code access, please contact the corresponding author.

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
| `figs/{0,1,3,5,6}-TD3threward.png` | TD3 reward histories for selected TCs |

### Reward arrays

Each `第{i}次total_reward_history.npy` / `第{i}次total_pure_reward_history.npy` stores the episode-wise reward trajectory for TC case `i`.

```python
import numpy as np

r = np.load("rewardhistory/第6次total_reward_history.npy")  # Hinnamnor
print(r.shape)
```

## Citation

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

Figures, reward histories, and the inner-loop RL algorithm modules are released for academic use with the paper. The full planning / training entry program is omitted from this public repository and may be shared upon reasonable request for non-commercial research.
