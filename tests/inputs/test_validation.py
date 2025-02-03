import os

import pytest
from jsonschema.exceptions import ValidationError

from electrolyzer.tools.validation import load_modeling_yaml


def test_basic_model():
    """A model config with all fields properly specified should pass validation."""
    fname_input_modeling = os.path.join(
        os.path.dirname(__file__), "test_modeling_options.yaml"
    )
    modeling_options = load_modeling_yaml(fname_input_modeling)

    # just make sure nothing is empty
    assert modeling_options["electrolyzer"]["supervisor"]
    assert modeling_options["electrolyzer"]["controller"]
    assert modeling_options["electrolyzer"]["stack"]
    assert modeling_options["electrolyzer"]["degradation"]
    assert modeling_options["electrolyzer"]["cell_params"]


def test_model_defaults():
    """A model config should be able to rely on defaults for some fields."""
    fname_input_modeling = os.path.join(
        os.path.dirname(__file__), "test_modeling_options_default.yaml"
    )
    modeling_options = load_modeling_yaml(fname_input_modeling)

    # electrolyzer properties
    assert modeling_options["general"]["verbose"] is False
    assert modeling_options["electrolyzer"]["name"] == "electrolyzer_001"
    assert modeling_options["electrolyzer"]["description"] == "An electrolyzer model"
    assert modeling_options["electrolyzer"]["dt"] == 1.0

    # controller properties
    controller_opts = modeling_options["electrolyzer"]["controller"]
    assert controller_opts["control_type"] == "DecisionControl"
    assert controller_opts["policy"]["baseline"] is True

    # stack properties
    stack_opts = modeling_options["electrolyzer"]["stack"]
    assert stack_opts["cell_type"] == "PEM"
    assert stack_opts["include_degradation_penalty"] is True

    # degradation properties
    degradation_opts = modeling_options["electrolyzer"]["degradation"]
    assert degradation_opts["PEM_params"]["rate_onoff"] == 1.47821515e-04
    assert degradation_opts["ALK_params"]["rate_onoff"] == 3.0726072607260716e-04

    # cell parameters
    cell_opts = modeling_options["electrolyzer"]["cell_params"]
    assert cell_opts["cell_type"] == "PEM"
    assert cell_opts["PEM_params"]["turndown_ratio"] == 0.1

    # cost parameters TODO: include cost parameter defaults
    # cost_opts = modeling_options["electrolyzer"]["costs"]


def test_model_invalid_spec():
    """A model config that specifies invalid fields will fail validation."""
    fname_input_modeling = os.path.join(
        os.path.dirname(__file__), "test_modeling_options_invalid.yaml"
    )

    with pytest.raises(ValidationError):
        load_modeling_yaml(fname_input_modeling)
