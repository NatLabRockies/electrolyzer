"""
This example performs an LCOH calculation for an electrolyzer system.
"""

import os

import numpy as np

from electrolyzer.tools.analysis.run_lcoh import run_lcoh


fname_input_modeling = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "cost_modeling_options.yaml"
)

turbine_rating = 3.4  # MW

# Create cosine test signal
test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
base_value = (turbine_rating / 2) + 0.2
variation_value = turbine_rating - base_value
power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6

lcoe = 44.18 * (1 / 1000)
res = run_lcoh(fname_input_modeling, power_test_signal, lcoe, optimize=False)

print(res)
