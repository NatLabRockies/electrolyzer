"""
This module is responsible for running electrolyzer models based on YAML configuration
files.
"""

import numpy as np
import pandas as pd

import electrolyzer.tools.validation as val
from electrolyzer.simulation.supervisor import Supervisor  # ESG
from electrolyzer.tools.design.optimization import calc_rated_system
from electrolyzer.simulation.bert import _run_electrolyzer_full
from electrolyzer.tools.analysis.lcoh import LCOH


def _run_electrolyzer_lcoh_opt(modeling_options, power_signal):
    # This function is similar to `run_electrolyzer_opt`, but keeps track of
    # some extra necessities for LCOH calcs.

    # Tune to a desired system rating
    options = calc_rated_system(modeling_options)

    # Initialize system
    elec_sys = Supervisor.from_dict(options["electrolyzer"])

    # Define output variables
    kg_rate = np.zeros(len(power_signal))
    curtailment = np.zeros((len(power_signal)))
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

        kg_rate[i] = loop_H2
        curtailment[i] = curtailed / 1000000

        new_curr = max([s.I / s.cell.cell_area for s in elec_sys.stacks])
        max_curr_density = max(max_curr_density, new_curr)

    return elec_sys, kg_rate, curtailment, max_curr_density


def _run_lcoh_full(cost_sys):
    lcoh = cost_sys.run_lcoh()

    # all code below this is used to get specific cost information
    # and lcoh breakdowns - feel free to comment out if you just
    # the lcoh
    data = {
        "Life Totals [$]": cost_sys.LCOH_summary["Totals [$]"],
        "Life Totals [$/kg-H2]": cost_sys.LCOH_summary["Totals [$/kg-H2]"],
    }
    lcoh_df_tots = pd.DataFrame(data)

    stack_rep_keys = [
        "Stack Replacement Cost [$/kW]",
        "Stack Replacement Cost [$/stack]",
    ]
    sr_srs = pd.Series(cost_sys.stack_replacement_summary, name="Stack Replacement")[
        stack_rep_keys
    ]

    feedstock_keys = [
        "Annual Electricity Cost [$]",
        "Annual H20 Cost [$]",
        "Total Feedstock Cost [$]",
    ]
    fds_srs = pd.Series(cost_sys.feedstock_summary, name="Feedstock")[feedstock_keys]
    cpx_srs = pd.concat(
        [
            pd.Series(cost_sys.capex_summary["BOP"], name="BOP"),
            pd.Series(cost_sys.capex_summary["PEM"], name="PEM"),
        ],
        axis=1,
    )

    raw_dict = {
        "CapEx": cpx_srs,
        "OpEx": pd.Series(cost_sys.opex_summary, name="OpEx"),
        "Feedstock": fds_srs,
        "Stack Replacement": sr_srs,
    }

    cost_sys.LCOH_summary["Yearly"]
    data = {
        "SR-Cost [$]": cost_sys.stack_replacement_summary[
            "Annual Stack Replacement Cost [$]"
        ],
        "num SR/year": cost_sys.stack_replacement_schedule[
            "Annual Number of Stacks to Replace"
        ],
    }
    stackrep_yrly = pd.DataFrame(data)

    # cost_sys.StackReplacement_schedule['Hrs until Replacement']
    # cost_sys.StackReplacement_schedule['Stacks Replaced']
    lcoh_dict = {
        "Raw Costs": raw_dict,
        "LCOH Breakdown": lcoh_df_tots,
        "LCOE-Yearly": cost_sys.LCOH_summary["Yearly"],
        "Stack Rep Raw": stackrep_yrly,
    }

    # below is for debugging
    # feedstock_dbg_keys=['Sim Electricity Cost [$]','Sim H20 Cost [$]']
    # pd.Series(cost_sys.Feedstock_summary,name="Feedstock")[feedstock_dbg_keys]
    # cost_sys.StackReplacement_schedule['Hrs until Replacement']
    # cost_sys.StackReplacement_schedule['Stacks Replaced']

    return lcoh_dict, lcoh


def run_lcoh(input_modeling, power_signal, lcoe, optimize=False):
    """
    Runs an electrolyzer simulation based on a YAML configuration file and power
    signal input.

    Args:
        input_modeling (`str` or `dict`): filepath specifying the YAML config
            file, OR a dict representing a validated YAML config.
        power_signal (`list`): An array representing power input
        lcoe (`float`): cost of energy for lcoh calculation
        optimize (`bool`, optional): Whether the run will be based on an optimization.
            For now, this entails tuning a system to a desired system rating, running
            the simulation, and returning a simplified result for optimization runs.

    Returns:
        'lcoh_dict': dictionary of LCOH values
        'lcoh' LCOh in $/kg-H2
    """
    err_msg = "Model input must be a str or dict object"
    assert isinstance(input_modeling, (str, dict)), err_msg

    if isinstance(input_modeling, str):
        # Parse/validate yaml configuration
        modeling_options = val.load_modeling_yaml(input_modeling)
    else:
        modeling_options = input_modeling

    if optimize:
        res = _run_electrolyzer_lcoh_opt(modeling_options, power_signal)
        elec_sys, kg_produced, curtailment, max_curr_density = res
    else:
        elec_sys, elec_df = _run_electrolyzer_full(modeling_options, power_signal)
        kg_produced = elec_df["kg_rate"].values
        curtailment = elec_df["curtailment"].values

    lcoh_options = modeling_options["electrolyzer"]["costs"]
    lcoh_options.update(
        {
            "dt": elec_sys.dt,
            "sim_length_hrs": len(power_signal) * elec_sys.dt / 3600,
            "plant_rating_kW": elec_sys.n_stacks * elec_sys.stack_rating_kW,
            "n_stacks": elec_sys.n_stacks,
            "stack_rating_kW": elec_sys.stack_rating_kW,
            "deg_state": elec_sys.deg_state,
            "power_kW_avail": power_signal / 1000,
            "power_kW_curtailed": curtailment / 1000,
            "kg_produced": kg_produced,
            "electrical_feedstock_cost": lcoe,  # [$/kWh]
        }
    )

    cost_sys = LCOH.from_dict(lcoh_options)

    # step 2: run lcoh calculations
    lcoh_dict, lcoh = _run_lcoh_full(cost_sys)

    if optimize:
        return [np.sum(kg_produced), max_curr_density, lcoh]

    return lcoh_dict, lcoh
