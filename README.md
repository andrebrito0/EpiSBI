# EpiSBI

Simulation-based inference resources for epidemic models.

This repository has two related purposes:

1. It provides code and data associated with the paper *A comparative study of simulation-based inference methods for epidemic models with identifiability considerations*.
2. It provides `episbi`, a reusable Python toolbox for simulation-based inference workflows with epidemic and compartmental models.

## Paper Repository

### Manuscript

This repository accompanies:

Jang G, Candan KS, Chowell G. A comparative study of simulation-based inference methods for epidemic models with identifiability considerations. *PLOS Computational Biology*. 2026;22(6):e1014364.

### Paper Goal

The paper compares likelihood-free and neural simulation-based inference approaches for epidemic models, with emphasis on how model identifiability affect posterior inference.

### Inference Methods Compared

| Method | Class | Toolkit | Description |
| --- | --- | --- | --- |
| ABC | Approximate Bayesian computation | `pyabc` | Sequential Monte Carlo ABC for likelihood-free posterior inference. |
| NPE | Neural posterior estimation | `sbi` | Learns the posterior distribution `p(theta | x)` from simulated parameter-observation pairs. |
| NPE-LSTM | Neural posterior estimation | `sbi` + PyTorch | Adds an LSTM embedding network for time-series epidemic observations. |
| PNPE | Preconditioned neural posterior estimation | `pyabc` + `sbi` | Uses ABC-based preconditioning before neural posterior estimation. |

### Epidemic Models

The manuscript materials include three compartmental epidemic models:

- Model1: SEIR model for standard susceptible-exposed-infectious-recovered dynamics.
- Model2: Ebola model tailored to the transmission dynamics of Ebola virus disease.
- Model3: SIRTEM model for COVID-19 epidemic modeling with testing and spatially informed response structure.

### Paper Materials

| Path | Contents |
| --- | --- |
| `data/` | Example data used by the original manuscript workflows. |
| `notebooks/` | Manuscript-oriented notebooks and exploratory analyses. |
| `temp_backup/` | Supporting source files from the earlier paper-code organization. |

## EpiSBI Toolbox

### Overview

`episbi` is a lightweight toolbox for building simulation-based inference workflows around user-defined epidemic simulators. It is designed to keep model simulation, prior definition, inference, posterior predictive evaluation, and tutorial examples in separate, reusable pieces.

### Main Features

- `SBIEngine` for ABC, NPE, NPE-LSTM, NRE, PNPE, and BSL (Bayesian synthetic likelihood) workflows.
- `episbi.adjustment.regression_adjust` for post-hoc local-linear or ridge regression adjustment of an accepted ABC sample (Beaumont, Zhang & Balding 2002), reusing simulations already run.
- `LSTMembedding` for neural inference with time-series observations.
- Prior helpers for `pyabc` and `sbi` workflows.
- `simulate_for_sbi` and `sample_prior` utilities for generating simulation datasets.
- Compartment-model builders for deterministic and stochastic models.
- Built-in model examples including deterministic SEIR, stochastic SEIR, stochastic SE1E2E3IR, and stochastic SEIRD.
- Posterior predictive evaluation metrics including MAE, RMSE, 95% interval coverage, interval score, and weighted interval score.
- Plotting utilities for inference and forecast windows.

### Workflow API Map

| Step | Goal | Main APIs |
| --- | --- | --- |
| 1. Define a model | Use a built-in epidemic simulator or define a compartment model. | `episbi.models.deterministic_seir`, `episbi.models.stochastic_seir`, `episbi.models.stochastic_seird`, `episbi.models.stochastic_se1e2e3ir`, `episbi.models.compartment_model`, `episbi.models.Transition` |
| 2. Define a prior | Keep parameter names and bounds in one object. | `episbi.prior.UniformPrior`, `episbi.prior.MixedPrior` |
| 3. Simulate training data | Generate `(theta, x)` arrays for SBI training. | `episbi.simulation.simulate_for_sbi`, `episbi.simulation.sample_prior` |
| 4. Fit inference | Run ABC, NPE, NPE-LSTM, NRE, PNPE, or BSL. | `SBIEngine.run_abc`, `SBIEngine.run_npe`, `SBIEngine.run_npe_lstm`, `SBIEngine.run_nre`, `SBIEngine.run_pnpe`, `SBIEngine.run_bsl` |
| 4b. Sharpen an ABC sample (optional) | Post-hoc regression adjustment reusing simulations `run_abc` already ran, at no extra simulation cost. | `episbi.adjustment.regression_adjust` |
| 5. Generate posterior predictive samples | Re-simulate trajectories from posterior samples. | `episbi.forecasting.posterior_predictive_forecast`, `episbi.forecasting.run_posterior_forecast_evaluation` |
| 6. Evaluate predictions | Compute trajectory summaries, WIS, interval score, RMSE, MAE, and coverage. | `episbi.metric.summarize_trajectories`, `episbi.metric.evaluate_trajectory`, `episbi.metric.evaluate_prediction_windows`, `episbi.metric.weighted_interval_score`, `episbi.metric.save_metrics`, `episbi.metric.summarize_metrics_by_output` |
| 7. Plot checks | Draw posterior predictive checks and inference/forecast windows. | `episbi.utils.plot_prediction_windows`, `episbi.forecasting.plot_forecast_result` |

### Installation

Clone the repository:

```bash
git clone https://github.com/geunsoojang/EpiSBI.git
cd EpiSBI
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install makes the local `episbi` package available to scripts and notebooks while you work from this repository.

### Quick Start

```python
import numpy as np

from episbi import SBIEngine
from episbi.forecasting import run_posterior_forecast_evaluation
from episbi.models import Transition, compartment_model
from episbi.prior import UniformPrior
from episbi.simulation import simulate_for_sbi

engine = SBIEngine(device="cpu", batch_size=256)

model = compartment_model(
    compartments=["S", "E", "I", "R"],
    transitions=[
        Transition("S", "E", "beta * S * I / N", name="S_to_E"),
        Transition("E", "I", "sigma * E", name="E_to_I"),
        Transition("I", "R", "gamma * I", name="I_to_R"),
    ],
    param_names=["beta", "sigma", "gamma"],
    initial_conditions={"S": 990, "E": 5, "I": 5, "R": 0},
    observed={"transitions": ["S_to_E"]},
    model_type="deterministic",
)

prior = UniformPrior(
    {
        "beta": (0.1, 0.8),
        "sigma": (0.05, 0.5),
        "gamma": (0.05, 0.4),
    }
)

theta_true = np.array([0.35, 0.2, 0.1], dtype=np.float32)
simulator_kwargs = {"population": 1000}
y_obs = model(theta_true, total_days=100, **simulator_kwargs)["S_to_E"]

thetas, xs = simulate_for_sbi(
    prior,
    model,
    num_simulations=1000,
    total_days=100,
    simulator_kwargs=simulator_kwargs,
)

fit = engine.run_npe(
    obs_data=y_obs,
    prior=prior,
    thetas=thetas,
    xs=xs,
    num_samples=1000,
)

result = run_posterior_forecast_evaluation(
    posterior_samples=fit["samples"],
    simulator=model,
    y_obs=y_obs,
    inference_days=90,
    forecast_days=10,
    param_names=prior.names,
    output_names=["incidence"],
    n_samples=300,
    simulator_kwargs=simulator_kwargs,
)

print(result["metrics"])
```

### Tutorials

The `tutorials/` directory contains notebook examples organized by model type and inference method.

| Notebook | Purpose |
| --- | --- |
| `01-1_Deterministic_seir_model.ipynb` | Build and simulate a deterministic SEIR model. |
| `01-2_Stochastic_seir_model.ipynb` | Build and simulate a stochastic SEIR model. |
| `02-1_ABC_deterministic_seir.ipynb` | ABC inference for deterministic SEIR. |
| `02-2_NPE_deterministic_seir.ipynb` | NPE inference for deterministic SEIR. |
| `02-3_NPE_LSTM_deterministic_seir.ipynb` | NPE-LSTM inference for deterministic SEIR. |
| `02-4_PNPE_deterministic_seir.ipynb` | PNPE inference for deterministic SEIR. |
| `03-1_ABC_stochastic_seir.ipynb` | ABC inference for stochastic SEIR. |
| `03-2_NPE_stochastic_seir.ipynb` | NPE inference for stochastic SEIR. |
| `03-3_NPE_LSTM_stochastic_seir.ipynb` | NPE-LSTM inference for stochastic SEIR. |
| `03-4_PNPE_stochastic_seir.ipynb` | PNPE inference for stochastic SEIR. |
| `04-1_NPE_SEIAR.ipynb` | NPE inference for a deterministic SEIAR model with symptomatic and asymptomatic infections. |
| `04-2_NPE_SEIAR.ipynb` | NPE inference for a deterministic SEIAR model where both symptomatic and asymptomatic infections contribute to transmission. |
| `04-3_NPE_SEIRD.ipynb` | NPE inference for a deterministic SEIRD model with disease-induced deaths. |
| `05_Custom_model.ipynb` | NRE inference with a notebook-defined stochastic SE1E2E3IR simulator using weekly incidence and `numba` acceleration. |
| `06_GPU.ipynb` | GPU-enabled NPE workflow showing how to run neural inference on CUDA when available. |

The custom Hantavirus model in `05_Custom_model.ipynb` follows the model setting from *Cruise Ship-Associated Andes Virus Cluster aboard MV Hondius, 2026: A Stochastic Scenario Analysis* ([arXiv:2606.08391](https://arxiv.org/abs/2606.08391)).

Start Jupyter from the repository root:

```bash
jupyter notebook
```

### Repository Structure

```text
EpiSBI/
|-- data/            # paper data
|-- notebooks/       # paper and exploratory notebooks
|-- temp_backup/     # earlier manuscript-code organization
|-- episbi/          # reusable toolbox package
|-- tutorials/       # toolbox tutorials
|-- README.md
`-- requirements.txt
```

## Citation

If you use the paper materials or the toolbox, please cite:

Jang G, Candan KS, Chowell G. A comparative study of simulation-based inference methods for epidemic models with identifiability considerations. *PLOS Computational Biology*. 2026;22(6):e1014364.
