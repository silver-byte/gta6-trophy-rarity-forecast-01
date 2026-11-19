# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/silver-byte/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/silver-byte/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic, standard-library-only Python forecaster for the earn rate of every Grand Theft Auto VI PlayStation 5 trophy at the case target. For each trophy it fits a recency-weighted trend of the logit earn rate against log-time, shrinks the slope toward the pooled cross-trophy slope, blends the extrapolation with persistence as the horizon grows, and calibrates the five quantile spreads by backtesting the fit inside the case's own history, so the bands widen honestly when the data has been moving.

## Run it

```sh
MAKEFUN_INPUT_DIR=<directory holding the case .in files> MAKEFUN_OUTPUT_DIR=<output directory> python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=sztec758&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
