# Electrolyzer

[![CI Status](https://github.com/NREL/electrolyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/NREL/electrolyzer/actions/workflows/ci.yml)
[![Lint](https://github.com/NREL/electrolyzer/actions/workflows/black.yml/badge.svg)](https://github.com/NREL/electrolyzer/actions/workflows/black.yml)

Electrolyzer is a controls-oriented engineering model for hydrogen production systems. It simulates
multi-stack electrolyzer operation, supports PEM and alkaline cell models, tracks degradation, and
includes levelized cost of hydrogen (LCOH) analysis utilities.

## What this repo provides

- Time-series simulation of one or more stacks with supervisory control logic.
- PEM and alkaline electrochemical cell models with polarization curve fitting.
- Degradation tracking (steady, fatigue, and on/off cycling) with optional penalty modes.
- Cost and LCOH analysis tools tied to simulation outputs.
- YAML-based modeling configuration with a JSON schema for validation and defaults.

## Project structure

- Core simulation: [electrolyzer/simulation](electrolyzer/simulation)
- Cell models: [electrolyzer/simulation/cell_models](electrolyzer/simulation/cell_models)
- Validation/schema: [electrolyzer/tools/validation.py](electrolyzer/tools/validation.py), [electrolyzer/tools/modeling_schema.yaml](electrolyzer/tools/modeling_schema.yaml)
- LCOH analysis: [electrolyzer/tools/analysis](electrolyzer/tools/analysis)
- Examples: [examples](examples)
- Documentation: [docs](docs)

## Installation

Python 3.11+ is required.

Create conda environment:

```bash
conda create --name bert python=3.11 -y
conda activate bert
```

Optional extras:

```bash
pip install ".[examples]"   # notebooks + example dependencies
pip install -e ".[develop]"  # dev + docs tooling
pip install -e ".[all]"      # everything
```

More detail is in [docs/installing.md](docs/installing.md).

## Quick start

Run a simulation from a YAML configuration and a power signal:

```python
import numpy as np

from electrolyzer.simulation.bert import run_electrolyzer

power_signal = np.ones(3600) * 1e6  # 1 MW for 1 hour, in Watts
elec_sys, results = run_electrolyzer("examples/example_02_electrolyzer/modeling_options.yaml", power_signal)

print(results.head())
```

Compute LCOH using the same signal:

```python
import numpy as np

from electrolyzer.tools.analysis.run_lcoh import run_lcoh

power_signal = np.ones(3600) * 1e6
lcoe = 0.04418  # $/kWh

lcoh_breakdown, lcoh_value = run_lcoh(
    "examples/example_04_lcoh/cost_modeling_options.yaml",
    power_signal,
    lcoe,
)

print(lcoh_value)
```

## Modeling configuration

Models are configured with YAML files validated against a JSON schema. The schema defines defaults
and accepted ranges for parameters like stack rating, cell geometry, degradation rates, and control
policy settings.

- Schema: [electrolyzer/tools/modeling_schema.yaml](electrolyzer/tools/modeling_schema.yaml)
- Example PEM configuration: [examples/example_02_electrolyzer/modeling_options.yaml](examples/example_02_electrolyzer/modeling_options.yaml)
- Example alkaline configuration: [examples/example_06_alkaline/default_alkaline.yaml](examples/example_06_alkaline/default_alkaline.yaml)

Key configuration blocks:

- `electrolyzer.supervisor`: system rating and number of stacks.
- `electrolyzer.controller`: control strategy and decision policy flags.
- `electrolyzer.stack`: stack sizing, cell type, and operational settings.
- `electrolyzer.degradation`: degradation rates and end-of-life parameters.
- `electrolyzer.cell_params`: PEM or alkaline cell model parameters.
- `electrolyzer.costs`: LCOH input data for capex, opex, feedstock, and finance.

## Control strategies

The supervisor supports multiple control modes for stack scheduling and power distribution:

- `PowerSharingRotation`, `SequentialRotation`
- `EvenSplitEagerDeg`, `EvenSplitHesitantDeg`
- `SequentialEvenWearDeg`, `SequentialSingleWearDeg`
- `BaselineDeg`
- `DecisionControl` (composed from policy flags in the YAML)

See [electrolyzer/simulation/supervisor.py](electrolyzer/simulation/supervisor.py) for logic.

## Degradation modeling

Each stack tracks voltage degradation from steady operation, fatigue, and on/off cycling. You can
choose whether degradation penalizes hydrogen production or increases power draw. The end-of-life
voltage delta drives replacement calculations in the LCOH workflow.

## Outputs

`run_electrolyzer` returns a supervisor object and a `pandas.DataFrame` of time-series results.
The frame includes overall power and curtailment plus per-stack columns for degradation, cycles,
uptime, hydrogen production rate, and current density.

## Examples

- Basic simulation: [examples/example_02_electrolyzer/example_run.py](examples/example_02_electrolyzer/example_run.py)
- Polarization curve fitting: [examples/example_01_polarization/example_run.py](examples/example_01_polarization/example_run.py)
- Controller behavior: [examples/example_05_controller/example_05_controller_options.py](examples/example_05_controller/example_05_controller_options.py)
- Alkaline configuration: [examples/example_06_alkaline/alkaline_example_run.py](examples/example_06_alkaline/alkaline_example_run.py)
- LCOH calculation: [examples/example_04_lcoh/cost_example_run.py](examples/example_04_lcoh/cost_example_run.py)

## Documentation

Docs are in [docs](docs). The landing page is [docs/intro.md](docs/intro.md). If you build the
Jupyter Book locally, the generated site lands in [docs/_build/html](docs/_build/html).

## Testing

```bash
pytest
```
