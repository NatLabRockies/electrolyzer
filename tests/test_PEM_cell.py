"""This module provides unit tests for `PEMCell`"""

import pytest

from electrolyzer.simulation.cell_models.pem import PEMCell as Cell


# from numpy.testing import assert_almost_equal


@pytest.fixture
def cell():
    return PEMCell.from_dict()
