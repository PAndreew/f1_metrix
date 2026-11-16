# F1 Metrix — legacy model

Historical Bayesian Formula One model and saved results, recovered from the original 2025 repository. This preserves the original three commits and model code; it is not the current website or its newer modelling experiments.

Current website/source: [f1-metrix-front](https://github.com/PAndreew/f1-metrix-front).

## Contents

- `main.py`: original PyMC model and result-export pipeline.
- `main_alt.py`: alternative historical implementation.
- `f1_results.db`: saved model summaries, driver performance-over-expectation, yearly driver skill rankings, all-time rankings and 2025 head-to-head predictions.
- `environment.yml` / `requirements.txt`: original numerical environment definitions.
- `Dockerfile` / `docker-compose.yml`: original container configuration, retained unchanged.

## Running the historical code

The scripts expect an F1DB SQLite database named `f1db.db` in the working directory. That source database and the large local posterior trace are intentionally ignored and **not included** here. See [F1DB](https://github.com/f1db/f1db) for the upstream data; schema compatibility must be checked against the historical queries.

A starting environment is:

```sh
conda env create -f environment.yml
conda activate pymc_env
python main.py
```

Running the script can trigger a long model fit and overwrite result artifacts. Back up `f1_results.db` before experimenting. This recovery did not rerun, modernise or validate the old model.

The retained Dockerfile references `model.py`, whereas this repository contains `main.py` and `main_alt.py`; that historical mismatch must be corrected before using the container setup. Dependencies are not fully pinned, so the environment is not guaranteed reproducible with current package releases.

## Historical artifacts

The original history includes a large Miniforge installer. It is retained to avoid rewriting the recovered history; it is not necessary if Conda/Miniforge is already installed. Its SHA256 was checked against the official Miniforge 24.11.3-2 release during recovery. The saved rankings are historical model outputs, not validated present-day forecasts or definitive measures of driver quality.

F1 data originates from F1DB; consult its attribution and licence terms before redistributing derived data. No new software licence is assigned by this recovery.
