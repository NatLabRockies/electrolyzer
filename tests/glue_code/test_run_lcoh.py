import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from electrolyzer.simulation.bert import run_electrolyzer
from electrolyzer.tools.analysis.run_lcoh import run_lcoh


lcoh_breakdown = pd.DataFrame(
    {
        "Life Totals [$]": [5.388657e06, 1.079412e06, 1.1979687e07, 1.225039e06],
        "Life Totals [$/kg-H2]": [
            1.3285182681325287,
            0.2661180588173578,
            2.9534690901554037,
            0.3020209876624967,
        ],
    },
    index=["CapEx", "OM", "Feedstock", "Stack Rep"],
)

RESULT = (lcoh_breakdown, 4.850126404767788)
ROOT = Path(__file__).parent.parent.parent


def test_run_lcoh():
    fname_input_modeling = os.path.join(
        ROOT, "examples", "example_04_lcoh", "cost_modeling_options.yaml"
    )

    turbine_rating = 3.4  # MW

    # Create cosine test signal
    test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
    base_value = (turbine_rating / 2) + 0.2
    variation_value = turbine_rating - base_value
    power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6

    lcoe = 44.18 * (1 / 1000)

    calc_result = run_lcoh(fname_input_modeling, power_test_signal, lcoe)

    assert_frame_equal(
        calc_result[0]["LCOH Breakdown"],
        RESULT[0],
        check_dtype=False,
        check_exact=False,
        atol=1e-1,
    )

    assert np.isclose(calc_result[1], RESULT[1])


def test_run_lcoh_opt():
    fname_input_modeling = os.path.join(
        ROOT, "examples", "example_04_lcoh", "cost_modeling_options.yaml"
    )

    turbine_rating = 3.4  # MW

    # Create cosine test signal
    test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
    base_value = (turbine_rating / 2) + 0.2
    variation_value = turbine_rating - base_value
    power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6

    lcoe = 44.18 * (1 / 1000)

    h2_result = run_electrolyzer(fname_input_modeling, power_test_signal, optimize=True)
    lcoh_result = run_lcoh(fname_input_modeling, power_test_signal, lcoe, optimize=True)

    # h2 prod, max curr density, LCOH
    assert len(lcoh_result) == 3

    # results from regular optimize run should match
    assert_array_almost_equal(h2_result, lcoh_result[:2])

    assert np.isclose(lcoh_result[2], 4.7541633856347625)
