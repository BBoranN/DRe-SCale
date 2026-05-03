# NSGD: Non-Stationary Stochastic Gradient Descent for Serverless Autoscaling

A serverless computing simulator implementing Non-Stationary Stochastic Gradient Descent (NSGD) algorithms for autoscaling optimization. Used for the experiments in **Autoscaling in Serverless Platforms via Online Learning with Convergence Guarantees**. Based on the [SimFaas](https://github.com/pacslab/simfaas) simulator.

## Overview

This repository provides four clean implementations of the NSGD autoscaler, varying along two axes:

| | Sequential | Parallel |
|---|---|---|
| **Scalar** (1 parameter) | `AutoscalerFaasScalar/` | `AutoscalerFaasScalarPar/` |
| **Vectorial** (3 parameters) | `AutoscalerFaasScalarVectorial/` | `AutoscalerFaasScalarVectorialPar/` |

**Scalar** optimizes a single parameter `theta` (number of extra servers to spawn on cold start) using the Non-Stationary Kiefer-Wolfowitz (NSKW) algorithm from Section VI-A of the paper.

**Vectorial** jointly optimizes three parameters `theta = (theta_stock, theta_idle, theta_exp)` using Simultaneous Perturbation Stochastic Approximation (SPSA):
- `theta_stock`: number of init-free servers to spawn on cold/warm starts
- `theta_idle`: provisioning threshold — triggers preemptive scale-up when idle count drops below this value
- `theta_exp`: expiration rate parameter controlling idle-to-cold transitions (rate = theta_exp / K_exp)

**Sequential** versions run a single simulation thread with interleaved plus/minus phases. **Parallel** versions spawn `2*K` multiprocessing workers (K plus, K minus), synchronized at `tau_n` boundaries via a Barrier. A leader process (`plus_0`) computes gradient updates after all workers sync.

## Algorithm

The SPSA iteration at step `n`:

1. **Compute sequences:**
   - Perturbation: `delta_n = k_delta / n^(2/3)`
   - Step size: `gamma_n = k_gamma / n`
   - Observation window: `tau_n = tau * ln(n+1)`

2. **Estimate costs:** Run the simulator for `K * tau_n` steps with `theta + perturbation` (plus phase), then `K * tau_n` steps with `theta - perturbation` (minus phase). Compute time-averaged cost for each phase.

3. **Estimate gradient:**
   - Scalar: `grad = (cost_plus - cost_minus) / (2 * delta_n)`
   - Vectorial: `grad_j = (cost_plus - cost_minus) / (2 * u_j * delta_n)` where `u` is a Rademacher random vector

4. **Update theta:** `theta_{n+1} = theta_n - gamma_n * optimizer(grad)`

Supported optimizers: SGD (paper default), Adam, RMSProp.

### Cost Function

```
C(x) = w1*x_idle_on + w2*x_busy + w3*x_init + w4*x_reserved + w_rej * I{rejected}
```

Default weights: `w1=2, w2=1, w3=5, w4=100, w_rej=200`.

### Stochastic Rounding

Integer parameters (`theta_stock`, `theta_idle`) use stochastic rounding:
`pi_theta = floor(theta)` with probability `ceil(theta) - theta`, else `ceil(theta)`, so that `E[pi_theta] = theta`.

## Project Structure

```
NSGD/
├── AutoscalerFaasScalar/                # Sequential, 1 parameter (NSKW)
│   ├── Algorithm.py                     # NSKW: sequences, gradient, optimizers, cost
│   ├── ServerlessSimulator.py           # Event-driven simulator with generate_trace()
│   ├── FunctionInstance.py              # Function instance lifecycle
│   ├── SimProcess.py                    # Arrival/service process generators
│   └── utils.py                         # FunctionState, SystemState enums
│
├── AutoscalerFaasScalarVectorial/       # Sequential, 3 parameters (SPSA)
│   ├── Algorithm.py                     # SPSA: Rademacher perturbations, per-component gamma
│   ├── ServerlessSimulator.py           # Policy 1: theta_stock, theta_idle, theta_exp
│   ├── api_algorithm.py                 # SPSA decoupled from simulator (for Flask API)
│   ├── flask_app.py                     # Flask REST API for real-system integration
│   ├── FunctionInstance.py
│   ├── SimProcess.py
│   └── utils.py
│
├── AutoscalerFaasScalarPar/             # Parallel, 1 parameter (NSKW)
│   ├── Algorithm.py                     # Full SPSA logic (gradient, optimizers, sequences)
│   ├── ServerlessSimulator.py           # Multiprocessing with Barrier sync
│   ├── FunctionInstance.py
│   ├── SimProcess.py
│   └── utils.py
│
├── AutoscalerFaasScalarVectorialPar/    # Parallel, 3 parameters (SPSA)
│   ├── Algorithm.py                     # 3-parameter SPSA with learn_mask, prtb
│   ├── ServerlessSimulator.py           # Parallel Policy 1 with shared state
│   ├── FunctionInstance.py
│   ├── SimProcess.py
│   └── utils.py
│
├── algo_config.json                     # Example config for Flask API (real deployment)
├── input_scalar_paper.json              # Scalar config matching paper Section VI-A
├── input_scalar_quick.json              # Scalar quick test (reduced tau, T)
├── input_scalar_par_smoke.json          # Scalar parallel smoke test
├── input_vectorial_paper.json           # Vectorial config matching paper Section VI-A
├── input_vectorial_quick.json           # Vectorial quick test
├── input_vectorial_par_smoke.json       # Vectorial parallel smoke test
└── requirements.txt                     # Python dependencies
```

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

```bash
git clone https://github.com/Wamuhindo/NSGD.git
cd NSGD
git checkout impl-clean
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.18.4
scipy>=1.4.1
pandas>=1.0.3
tqdm>=4.46.0
matplotlib>=3.2.1
flask>=2.0.0        # only needed for the REST API
```

## Usage

All four implementations accept JSON configuration files and can be run as modules or scripts.

### Sequential Scalar (NSKW, 1 parameter)

```bash
# Quick test (~30s)
python3 -m AutoscalerFaasScalar.ServerlessSimulator --input input_scalar_quick.json

# Full paper experiment
python3 -m AutoscalerFaasScalar.ServerlessSimulator --input input_scalar_paper.json
```

### Sequential Vectorial (SPSA, 3 parameters)

```bash
# Quick test
python3 -m AutoscalerFaasScalarVectorial.ServerlessSimulator --input input_vectorial_quick.json

# Full paper experiment
python3 -m AutoscalerFaasScalarVectorial.ServerlessSimulator --input input_vectorial_paper.json
```

### Parallel Scalar (NSKW, 1 parameter, multiprocessing)

```bash
# Smoke test (~2s)
python3 -m AutoscalerFaasScalarPar.ServerlessSimulator --input input_scalar_par_smoke.json

# Full run (uses same config format as sequential)
python3 -m AutoscalerFaasScalarPar.ServerlessSimulator --input input_scalar_paper.json
```

### Parallel Vectorial (SPSA, 3 parameters, multiprocessing)

```bash
# Smoke test (~2s)
python3 -m AutoscalerFaasScalarVectorialPar.ServerlessSimulator --input input_vectorial_par_smoke.json

# Full run
python3 -m AutoscalerFaasScalarVectorialPar.ServerlessSimulator --input input_vectorial_paper.json
```

## Flask API (Real-System Integration)

The vectorial SPSA algorithm can run as a REST API for integration with a real Kubernetes/OpenFaaS platform. Instead of running a simulator, the algorithm receives system state observations via HTTP and returns scaling parameters (`theta`).

### Setup

```bash
pip install flask
python -m AutoscalerFaasScalarVectorial.flask_app --config algo_config.json
```

To restore from a previously saved snapshot:

```bash
python -m AutoscalerFaasScalarVectorial.flask_app --config algo_config.json --restore logs/api/autosave.json
```

Options: `--host` (default `0.0.0.0`), `--port` (default `5000`), `--debug`.

### API Endpoints

**POST /event** -- Report a system state observation and advance the algorithm.

```json
// Request
{
  "state": [40, 3, 5, 1, 1],
  "has_rejected_job": false,
  "timestamp": 1709000000.0
}
// Response
{
  "theta": [1.0, 1.0, 5.0],
  "theta_step": [1, 1, 5],
  "iteration": 1,
  "phase": "plus",
  "step_in_phase": 1,
  "phase_budget": 466,
  "cost": 12.0
}
```

The `state` array has 5 elements: `[cold, idle_on, busy, initializing, init_reserved]`. The response includes the current `theta_step` (stochastically rounded, ready for scaling decisions) and phase progress.

**GET /theta** -- Get the current scaling parameters without advancing the algorithm.

**GET /status** -- Full algorithm introspection (iteration, phase, theta history, cost history, sequences).

**POST /snapshot** -- Save algorithm state to disk. Optionally include replay data for simulator replay.

```json
// Request (optional)
{ "path": "/custom/path.json", "include_replay_data": true }
```

**POST /force-update** -- Force a gradient update with accumulated costs so far. Useful when real-system traffic is low and phases take too long to complete naturally.

**POST /config** -- Update weights, optimizer, or learn_mask at runtime without restart.

```json
{ "weights": {"w_idle_on": 3}, "optimization": "adam", "learn_mask": [true, true, false] }
```

**POST /reset** -- Reset the algorithm to its initial state (keeping config).

**GET /health** -- Health check.

### API Configuration (`algo_config.json`)

The API uses the same SPSA hyperparameters as the simulator. The `tau` parameter controls the base observation window (`tau_n = tau * ln(n+1)` events per phase). In simulation, events are generated instantly, so large values like `tau=100000` are practical. In a real system, each event is an actual function invocation or state change, so the total phase duration depends on your system's event rate. Choose `tau` based on how many observations your system needs to reach approximate stationarity under a given theta — this depends on the arrival rate, service times, and mixing time of your platform. Too small means noisy gradient estimates; too large means slow adaptation.

```json
{
  "max_concurrency": 50,
  "theta_init": [1, 1, 5],
  "k_delta": 1,
  "k_gamma": [1, 1, 1],
  "tau": 100,
  "K": 2,
  "optimization": "adam",
  "seed": 1,
  "K_exp": 1000,
  "gamma_min": 1,
  "prtb": [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]],
  "learn_mask": [true, true, true],
  "accumulate_cost": true,
  "weights": { "w_cold": 0, "w_idle_on": 2, "w_busy": 1, "w_init": 5, "w_reserved": 100, "w_rej": 200 },
  "log_dir": "logs/api",
  "persistence": { "auto_save_interval": 100, "auto_save_path": "logs/api/autosave.json" }
}
```

The `persistence` block enables automatic state saving every N events for crash recovery.

### Logging

The API writes JSONL log files under `log_dir`:
- `events.jsonl` -- every state observation with cost and phase info
- `gradient_updates.jsonl` -- each gradient update with costs, gradient, and new theta

### State Persistence and Replay

Snapshots saved via `/snapshot` (or auto-save) include the full algorithm state and RNG states, allowing exact recovery after restart. When `include_replay_data` is set, a companion file with all observed states and costs is saved for offline simulator replay using `FunctionInstance.py`.

## Configuration Reference

### Common Parameters

| Parameter | Type | Description |
|---|---|---|
| `arrival_rate` | float | Poisson arrival rate (lambda). Paper: 7.5 (lambda/N = 0.15 with N=50) |
| `warm_service` | object | `{"rate": float, "type": "Exponential"}`. Warm service rate (mu). Paper: 1 |
| `cold_service` | object | Cold service rate. Paper: 100 (1/beta_cs) |
| `cold_start` | object | Cold start initialization rate (beta). Paper: 0.1 |
| `expiration` | object | Idle-to-cold expiration rate (gamma). Paper: 0.01 |
| `optimization` | object | `{"type": "SGD"\|"adam"\|"RMSprop"}` |
| `tau` | int | Base observation window. tau_n = tau * ln(n+1). Paper: 10^5 |
| `max_concurrency` | int | Maximum number of function instances (N). Paper: 50 |
| `max_time` | float | Total simulation time (T). Paper: 10^7 (scalar), 4*10^6 (vectorial) |
| `K` | int | Number of independent cost samples per perturbation direction |
| `seeds` | list[int] | Random seeds for reproducibility |
| `k_delta` | float | Perturbation scale. delta_n = k_delta / n^(2/3). Paper: 1 |
| `log_dir` | string | Output directory for logs. Default: `"logs/"` |
| `experiment_name` | string | Prefix for the log subdirectory |

### Scalar-Specific Parameters

| Parameter | Type | Description |
|---|---|---|
| `theta_inits` | list[float] | Initial theta values to sweep. Each produces a separate run |
| `k_gamma` | float | Step size scale. gamma_n = k_gamma / n |
| `k_gamma_per_theta` | list[float] or null | Per-initialization k_gamma override (see below). Length must match `theta_inits` |

### Vectorial-Specific Parameters

| Parameter | Type | Description |
|---|---|---|
| `theta` | list[list[float]] | Initial theta vectors `[theta_stock, theta_idle, theta_exp]` to sweep |
| `k_gamma` | list[float] (length 3) | Per-component step size scale. gamma_n_j = k_gamma_j / n |
| `k_gamma_per_theta` | list[list[float]] or null | Per-initialization k_gamma override (see below). Each entry is a 3-element vector |
| `prtb` | list[list[float]] | Perturbation choices per dimension. Default: `[[-0.5,0.5], [-0.5,0.5], [-1,1]]` |
| `learn_mask` | list[bool] (length 3) | Which dimensions to optimize. `false` freezes that dimension |
| `accumulate_cost` | bool | `true`: time-average over all steps (stable). `false`: endpoint sampling |
| `K_exp` | float | Expiration rate divisor. Actual rate = theta_exp / K_exp. Paper: 1000 |
| `gamma_min` | float | Minimum allowed value for theta_exp. Paper: 1 |

### Per-Initialization Step Size (`k_gamma_per_theta`)

By default, all theta initializations share the same `k_gamma` value (or vector, in the vectorial case). The `k_gamma_per_theta` parameter allows setting a different step size for each initialization, which is useful when different starting points benefit from different learning rates.

When `k_gamma_per_theta` is provided, the i-th entry overrides `k_gamma` for the i-th theta initialization. When omitted or `null`, the global `k_gamma` is used for all initializations.

**Scalar example** — different step sizes for each of 3 starting points:
```json
{
  "theta_inits": [1, 5, 10],
  "k_gamma": 1,
  "k_gamma_per_theta": [1, 0.5, 2]
}
```
Here `theta_init=1` uses `k_gamma=1`, `theta_init=5` uses `k_gamma=0.5`, and `theta_init=10` uses `k_gamma=2`.

**Vectorial example** — per-component step sizes for each of 2 starting points:
```json
{
  "theta": [[1,1,5], [10,2,1]],
  "k_gamma": [1, 1, 1],
  "k_gamma_per_theta": [[1, 1, 1], [0.5, 0.5, 0.5]]
}
```
Here the first initialization uses `k_gamma=[1,1,1]` and the second uses `k_gamma=[0.5,0.5,0.5]`.

### Example Configurations

**Scalar (paper Section VI-A):**
```json
{
  "arrival_rate": 7.5,
  "warm_service": {"rate": 1, "type": "Exponential"},
  "cold_service": {"rate": 100, "type": "Exponential"},
  "cold_start": {"rate": 0.1, "type": "Exponential"},
  "expiration": {"rate": 0.01, "type": "Exponential"},
  "optimization": {"type": "RMSprop"},
  "theta_inits": [1, 3, 5, 10],
  "tau": 100000,
  "max_concurrency": 50,
  "max_time": 10000000,
  "K": 2,
  "seeds": [1000],
  "k_delta": 1,
  "k_gamma": 1
}
```

**Vectorial (paper Section VI-A):**
```json
{
  "arrival_rate": 7.5,
  "warm_service": {"rate": 1, "type": "Exponential"},
  "cold_service": {"rate": 100, "type": "Exponential"},
  "cold_start": {"rate": 0.1, "type": "Exponential"},
  "expiration": {"rate": 0.01, "type": "Exponential"},
  "optimization": {"type": "adam"},
  "theta": [[1,1,5], [1,5,5], [3,4,10], [10,2,1]],
  "tau": 100000,
  "max_concurrency": 50,
  "max_time": 4000000,
  "K": 2,
  "seeds": [1],
  "k_delta": 1,
  "k_gamma": [1, 1, 1],
  "prtb": [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]],
  "learn_mask": [true, true, true],
  "accumulate_cost": true,
  "K_exp": 1000,
  "gamma_min": 1
}
```

## Output

Results are saved under `log_dir` in a hierarchical structure:

```
logs/
└── <experiment_name>_arr<rate>_<timestamp>/
    ├── experiment_config.json          # Copy of the input config
    └── theta_<init>/                   # One directory per theta initialization
        └── seed_<seed>/                # One directory per seed
            ├── config.json             # Run-specific parameters
            ├── theta_costs_v.txt       # Per-iteration: theta, cost+, cost-, avg_cost
            ├── results.json            # Final summary metrics
            └── sim_<id>_results.txt    # Per-process results (parallel only)
```

Key metrics reported per run:
- Cold start probability
- Rejection probability
- Average server count, idle count, busy count
- Average instance lifespan
- Theta convergence trace

## Architecture Notes

### Sequential vs Parallel

The **sequential** implementations (`AutoscalerFaasScalar`, `AutoscalerFaasScalarVectorial`) use a single-threaded event loop via `generate_trace()`. The Algorithm class internally manages plus/minus phases, accumulating costs and computing gradients within `simulate_step()`.

The **parallel** implementations (`AutoscalerFaasScalarPar`, `AutoscalerFaasScalarVectorialPar`) use `multiprocessing.Process` workers. Each worker runs an independent simulator via `process_next_event()` (single event per call, driven by the parallel loop). Workers synchronize at `tau_n` boundaries using a `Barrier`. The leader process (`plus_0`) reads averaged costs from shared memory and computes the gradient update via `Algorithm` methods:
- `algo.get_delta_n(n)`, `algo.get_gamma_n(n)`, `algo.get_tau_n(n)` — SPSA sequences
- `algo.compute_gradient(...)` — gradient estimation
- `algo.apply_optimizer(grad, optimization)` — optimizer step (SGD/Adam/RMSProp)

### Scalar vs Vectorial Simulator Differences

The scalar simulator spawns `pi_theta` init-free servers on cold starts only.

The vectorial simulator implements **Policy 1**:
- On cold starts: spawn 1 init-reserved + `pi_theta_stock` init-free servers
- On warm starts: if `#idle_on < pi_theta_idle`, preemptively spawn up to `pi_theta_stock` additional servers
- Expiration rate is set dynamically to `theta_exp / K_exp`

## Citation

```bibtex
@software{nsgd_simulator,
  title={SimFaas-NSGD: Simulator with Non-Stationary Stochastic Gradient Descent for Serverless Autoscaling},
  author={Kambale, Abednego Wamuhindo and Anselmi, Jonatha and Ardagna, Danilo and Gaujal, Bruno},
  year={2025},
  url={https://github.com/Wamuhindo/NSGD}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
