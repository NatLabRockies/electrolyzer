import os

import numpy as np
import pandas as pd
import pytest
from pytest import fixture
from numpy.testing import (
    assert_array_equal,
    assert_almost_equal,
    assert_array_almost_equal,
)

import electrolyzer.tools.validation as val
from electrolyzer.simulation.supervisor import Supervisor
from electrolyzer.simulation.bert import run_electrolyzer
from electrolyzer.tools.validation import load_modeling_yaml
from electrolyzer.tools.design.optimization import calc_rated_system

turbine_rating = 3.4  # MW

# Create cosine test signal
test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
base_value = (turbine_rating / 2) + 0.2
variation_value = turbine_rating - base_value
power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6
fname_input_modeling = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_modeling_options.yaml"
)


@fixture(scope="module")
def result():
    """Run the electrolyzer once, and use its result for subsequent tests."""
    res = run_electrolyzer(fname_input_modeling, power_test_signal)

    return res


def test_run_electrolyzer_yaml(result):
    """An electrolyzer run should return two outputs."""
    assert len(result) == 2

    assert isinstance(result[0], Supervisor)
    assert isinstance(result[1], pd.DataFrame)


def test_run_electrolyzer_dict():
    """`run_electrolyzer` should accept a filename or dict."""
    model_input = val.load_modeling_yaml(fname_input_modeling)

    run_electrolyzer(model_input, [])

    bad_input = 3

    with pytest.raises(AssertionError):
        run_electrolyzer(bad_input, [])


def test_degradation_dt():
    """Larger time steps should undercalculate degradation"""

    model_input = val.load_modeling_yaml(fname_input_modeling)

    # initialize with dt = 1
    model_input["electrolyzer"]["dt"] = 1
    res1 = run_electrolyzer(model_input, power_test_signal)
    _, df1 = res1
    deg1 = df1[[col for col in df1 if "deg" in col]]

    # initialize with dt = 60
    model_input["electrolyzer"]["dt"] = 60
    res60 = run_electrolyzer(model_input, power_test_signal)
    _, df60 = res60
    deg60 = df60[[col for col in df60 if "deg" in col]]

    # initialize with dt = 3600
    model_input["electrolyzer"]["dt"] = 3600
    res3600 = run_electrolyzer(model_input, power_test_signal)
    _, df3600 = res3600
    deg3600 = df3600[[col for col in df3600 if "deg" in col]]

    assert all(deg3600 < deg60)
    assert all(deg60 < deg1)


def test_result_df(result):
    """An electrolyzer run should return a `DataFrame` with time series output."""
    sup, df = result

    assert len(df) == len(power_test_signal)

    kg_rates = df[[col for col in df if "_kg_rate" in col]]
    cycles = df[[col for col in df if "cycles" in col]]
    degradation = df[[col for col in df if "deg" in col]]
    curr_density = df[[col for col in df if "curr_density" in col]]

    # Expected columns
    assert "curtailment" in df.columns
    assert "kg_rate" in df.columns
    assert "power_signal" in df.columns
    assert len(kg_rates.columns) == sup.n_stacks
    assert len(cycles.columns) == sup.n_stacks
    assert len(degradation.columns) == sup.n_stacks
    assert len(degradation.columns) == sup.n_stacks
    assert len(curr_density.columns) == sup.n_stacks

    # Expected data
    assert_array_equal(df["power_signal"], power_test_signal)

    # Individual kg production should sum to full
    assert_almost_equal(df["kg_rate"].sum(), sum(kg_rates.sum()))


def test_optimize():
    """Test the `optimize` optional param."""
    res = run_electrolyzer(fname_input_modeling, power_test_signal, optimize=True)

    # set up the same scenario, but do full run
    modeling_options = load_modeling_yaml(fname_input_modeling)
    options = calc_rated_system(modeling_options)
    _, df = run_electrolyzer(options, power_test_signal)

    assert len(res) == 2
    assert_almost_equal(res[0], df["kg_rate"].sum())

    curr_dens = df[[col for col in df if "curr_density" in col]]
    assert_almost_equal(res[1], max(curr_dens.max().values))


def test_regression(result):
    """
    Test specific values for the result. We expect this test to fail any time we make
    model changes.
    """
    _, df = result

    # Test total kg H2 produced
    assert_almost_equal(df["kg_rate"].sum(), 228.0749041980651, decimal=1)

    # Test degradation state of stacks
    degradation = df[[col for col in df if "deg" in col]]
    assert_array_almost_equal(
        degradation.tail(1).values[0],
        [
            1.0040428622415501e-05,
            9.786503510654988e-06,
            9.527106756951304e-06,
            9.295732960215869e-06,
            9.064896715672033e-06,
            8.80771986702221e-06,
            8.282891454647921e-06,
        ],
        decimal=5,
    )
