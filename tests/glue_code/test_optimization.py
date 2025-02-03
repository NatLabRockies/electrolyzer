import os

from numpy.testing import assert_almost_equal

import electrolyzer.tools.validation as val
from electrolyzer.tools.design.optimization import calc_rated_stack, calc_rated_system


input_modeling = "./test_modeling_options.yaml"
fname_input_modeling = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_modeling_options.yaml"
)


def test_calc_rated_system():
    """Should be able to size a system that meets the desired rating."""
    modeling_options = val.load_modeling_yaml(fname_input_modeling)

    tuned_options = calc_rated_system(modeling_options)

    assert tuned_options["electrolyzer"]["supervisor"]["n_stacks"] == 4
    assert_almost_equal(
        tuned_options["electrolyzer"]["stack"]["stack_rating_kW"], 500.0
    )


def test_calc_rated_stack():
    """
    Should be able to handle a case where the desired rating is lower than
    the calculated rating.
    """
    modeling_options = val.load_modeling_yaml(fname_input_modeling)

    # Update modeling dict in place
    calc_rated_stack(modeling_options)

    # n_cells should get bigger
    assert modeling_options["electrolyzer"]["stack"]["n_cells"] > 100

    # cell area should get smaller
    modeling_options["electrolyzer"]["cell_params"]["PEM_params"]["cell_area"] < 1000

    assert_almost_equal(
        modeling_options["electrolyzer"]["stack"]["stack_rating_kW"], 500.000, decimal=3
    )


def test_calc_rated_stack_lower():
    """
    Should be able to handle a case where the desired rating is higher than
    the calculated rating.
    """
    modeling_options = val.load_modeling_yaml(fname_input_modeling)

    # set the desired stack rating, this time lower than baseline
    modeling_options["electrolyzer"]["stack"]["stack_rating_kW"] = 750

    calc_rated_stack(modeling_options)

    assert modeling_options["electrolyzer"]["stack"]["n_cells"] > 100

    # cell area should decrease
    assert (
        modeling_options["electrolyzer"]["cell_params"]["PEM_params"]["cell_area"]
        < 1000
    )

    # max current should decrease
    assert modeling_options["electrolyzer"]["stack"]["max_current"] < 2000

    assert_almost_equal(
        modeling_options["electrolyzer"]["stack"]["stack_rating_kW"], 750.000, decimal=3
    )
