"""
This module is responsible for running electrolyzer models based on YAML configuration
files.
"""

import numpy as np
import pandas as pd

import electrolyzer.tools.validation as val
from electrolyzer.simulation.supervisor import Supervisor
from electrolyzer.tools.design.optimization import calc_rated_system


def _run_electrolyzer_full(modeling_options, power_signal):
    # Initialize system
    elec_sys = Supervisor.from_dict(modeling_options["electrolyzer"])

    # Define output variables
    kg_rate = np.zeros((elec_sys.n_stacks, len(power_signal)))
    degradation = np.zeros((elec_sys.n_stacks, len(power_signal)))
    curtailment = np.zeros((len(power_signal)))
    tot_kg = np.zeros((len(power_signal)))
    cycles = np.zeros((elec_sys.n_stacks, len(power_signal)))
    uptime = np.zeros((elec_sys.n_stacks, len(power_signal)))
    current_density = np.zeros((elec_sys.n_stacks, len(power_signal)))
    p_in = []

    # Run electrolyzer simulation
    for i in range(len(power_signal)):
        # TODO: replace with proper logging
        # if (i % 1000) == 0:
        #     print('Progress', i)
        # print(i)
        loop_H2, loop_h2_mfr, loop_power_left, curtailed = elec_sys.run_control(
            power_signal[i]
        )
        p_in.append(power_signal[i] / elec_sys.n_stacks / 1000)

        tot_kg[i] = loop_H2
        curtailment[i] = curtailed / 1000000
        for j in range(elec_sys.n_stacks):
            stack = elec_sys.stacks[j]
            kg_rate[j, i] = loop_h2_mfr[j]
            degradation[j, i] = stack.V_degradation
            cycles[j, i] = stack.cycle_count
            uptime[j, i] = stack.uptime
            current_density[j, i] = stack.I / stack.cell.cell_area

    # Collect results into a DataFrame
    results_df = pd.DataFrame(
        {
            "power_signal": power_signal,
            "curtailment": curtailment,
            "kg_rate": tot_kg,
        }
    )

    # for efficiency reasons, create a df for each stack, then concat all at the end
    stack_dfs = []

    for i, stack in enumerate(elec_sys.stacks):
        id = i + 1
        stack_df = pd.DataFrame(
            {
                f"stack_{id}_deg": degradation[i, :],
                f"stack_{id}_fatigue": stack.fatigue_history,
                f"stack_{id}_cycles": cycles[i, :],
                f"stack_{id}_uptime": uptime[i, :],
                f"stack_{id}_kg_rate": kg_rate[i, :],
                f"stack_{id}_curr_density": current_density[i, :],
            }
        )
        stack_dfs.append(stack_df)

    results_df = pd.concat([results_df, *stack_dfs], axis=1)

    return elec_sys, results_df


def _run_electrolyzer_opt(modeling_options, power_signal):
    # Tune to a desired system rating
    options = calc_rated_system(modeling_options)

    # Initialize system
    elec_sys = Supervisor.from_dict(options["electrolyzer"])

    # Define output variables
    tot_kg = 0.0
    max_curr_density = 0.0

    # Run electrolyzer simulation
    for i in range(len(power_signal)):
        # TODO: replace with proper logging
        # if (i % 1000) == 0:
        #     print('Progress', i)
        # print(i)
        loop_H2, loop_h2_mfr, loop_power_left, curtailed = elec_sys.run_control(
            power_signal[i]
        )

        tot_kg += loop_H2
        new_curr = max([s.I / s.cell.cell_area for s in elec_sys.stacks])
        max_curr_density = max(max_curr_density, new_curr)

    return tot_kg, max_curr_density


def run_electrolyzer(input_modeling, power_signal, optimize=False):
    """
    Runs an electrolyzer simulation based on a YAML configuration file and power
    signal input.

    Args:
        input_modeling (`str` or `dict`): filepath specifying the YAML config
            file, OR a dict representing a validated YAML config.
        power_signal (`list`): An array representing power input
        optimize (`bool`, optional): Whether the run will be based on an optimization.
            For now, this entails tuning a system to a desired system rating, running
            the simulation, and returning a simplified result for optimization runs.

    Returns:
        `Supervisor`: the instance used to run the simulation
        `pandas.DataFrame`: a `DataFrame` representing the time series output
            for the system, including values for each electrolyzer stack
    """
    err_msg = "Model input must be a str or dict object"
    assert isinstance(
        input_modeling,
        (
            str,
            dict,
        ),
    ), err_msg

    if isinstance(input_modeling, str):
        # Parse/validate yaml configuration
        modeling_options = val.load_modeling_yaml(input_modeling)
    else:
        modeling_options = input_modeling

    if optimize:
        return _run_electrolyzer_opt(modeling_options, power_signal)

    return _run_electrolyzer_full(modeling_options, power_signal)
