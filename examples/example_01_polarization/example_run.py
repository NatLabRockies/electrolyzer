"""
This example runs the Electrolyzer on its own and generates polarization curves.
"""

import numpy as np
import matplotlib.pyplot as plt

from electrolyzer.simulation.stack import Stack
from electrolyzer.simulation.cell_models.pem import PEM_electrolyzer_model as electrolyzer_model


stack_dict = {
    "cell_type": "PEM",
    "max_current": 2000,
    "temperature": 60,
    "n_cells": 100,
    "cell_params": {
        "cell_type": "PEM",
        "PEM_params": {
            "cell_area": 1000,
            "turndown_ratio": 0.1,
            "max_current_density": 2,
        },
    },
    "degradation": {
        "PEM_params": {
            "rate_steady": 1.41737929e-10,
            "rate_fatigue": 3.33330244e-07,
            "rate_onoff": 1.47821515e-04,
        }
    },
    "dt": 1,
}

elec = Stack.from_dict(stack_dict)

cur = np.linspace(0, 2500, 100)
p_fit = elec.calc_stack_power(
    electrolyzer_model(
        (elec.calc_stack_power(cur), stack_dict["temperature"]), *elec.fit_params
    )
)

p_actual = elec.calc_stack_power(cur)
voltage = elec.cell.calc_cell_voltage(cur, stack_dict["temperature"])
fit_error = p_actual - p_fit

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(cur / (1000), fit_error / elec.stack_rating_kW * 100)
ax1.grid(True)
ax1.set_ylabel("error (% rated power)")
ax1.set_title("Power error (actual - fit)")
ax2.plot(cur / (1000), p_actual / elec.stack_rating_kW)
ax2.plot(cur / (1000), p_fit / elec.stack_rating_kW)
ax2.grid(True)
ax2.set_title("power normalized by stack rating")
ax2.set_ylabel("normalized power")
ax2.set_xlabel("Current density (A/cm^2)")
ax2.legend(["actual", "fit"])

plt.figure()
plt.plot(cur / 1000, voltage)
plt.grid(True)
plt.xlabel("Current density (A/cm^2)")
plt.ylabel("Voltage (V)")
plt.title("Polarization Curve")
plt.show()
