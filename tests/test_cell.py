"""This module provides unit tests for `Cell`."""

import pytest
from numpy.testing import assert_almost_equal

from electrolyzer.simulation.cell_models.pem import PEMCell as Cell


@pytest.fixture
def cell():
    return Cell.from_dict(
        {"cell_area": 1000, "turndown_ratio": 0.1, "max_current_density": 2}
    )


def test_init():
    """`Cell` should initialize properly from a Dictionary."""
    cell = Cell.from_dict(
        {"cell_area": 1000, "turndown_ratio": 0.1, "max_current_density": 2}
    )

    assert cell.cell_area == 1000
    assert cell.n == 2
    assert cell.gibbs == 237.24e3
    assert cell.M == 2.016  # molecular weight [g/mol]
    assert cell.lhv == 33.33  # lower heating value of H2 [kWh/kg]
    assert cell.hhv == 39.41  # higher heating value of H2 [kWh/kg]


def test_calc_reversible_voltage(cell: Cell):
    """Reversible cell potential should match literature."""
    E_rev = cell.calc_reversible_voltage()

    assert_almost_equal(E_rev, 1.229, decimal=3)


def test_calc_open_circuit_voltage(cell: Cell):
    """Open circuit voltage should follow Nernst equation."""
    T = 60  # C
    E_rev = cell.calc_reversible_voltage()

    # should be greater than reversible cell voltage
    E_cell = cell.calc_open_circuit_voltage(T)
    assert E_cell > E_rev

    # should approach E_rev at near 100C (valid temperature range)
    E_cell_25 = cell.calc_open_circuit_voltage(99.9725)
    assert_almost_equal(E_cell_25, E_rev, decimal=3)


def test_calc_activation_overpotential(cell: Cell):
    """Activation overpotential should follow Butler-Volmer equations."""
    T = 60  # C
    I = 2000  # current
    i = I / cell.cell_area  # current density
    V_act = cell.calc_activation_overpotential(i, T)
    V_act_a, V_act_c = V_act

    # should be gte 0
    assert sum(V_act) >= 0

    # cathode should have a higher overpotential, under assumption that reaction
    # kinetics are faster at the anode (oxidation) than cathode (reduction)
    assert V_act_c > V_act_a

    # should increase with temperature
    T2 = 80
    V_act_T2 = cell.calc_activation_overpotential(i, T2)
    assert sum(V_act_T2) > sum(V_act)

    # should increase with current density
    cell.cell_area /= 2
    i2 = I / cell.cell_area  # current density increases
    V_act_i2 = cell.calc_activation_overpotential(i2, T)
    assert sum(V_act_i2) > sum(V_act)


def test_calc_ohmic_overpotential(cell: Cell):
    """Ohmic overpotential should reflect standard Ohm's law."""
    T = 60  # C
    I = 2000  # current
    i = I / cell.cell_area  # current density

    V_ohm = cell.calc_ohmic_overpotential(i, T)

    # should be gte 0
    assert V_ohm >= 0

    # should decrease with temperature due to increased membrane conductivity
    T2 = 80
    V_ohm_T2 = cell.calc_ohmic_overpotential(i, T2)
    assert V_ohm_T2 < V_ohm

    # should increase with current density (V = IR)
    cell.cell_area /= 2
    i2 = I / cell.cell_area
    V_ohm_i2 = cell.calc_ohmic_overpotential(i2, T)
    assert V_ohm_i2 > V_ohm


def test_calc_concentration_overpotential(cell: Cell):
    """TODO: This method is not yet implemented on the class."""
    assert cell.calc_concentration_overpotential() == 0


def test_calc_overpotentials(cell: Cell):
    """Should calculate and return all overpotentials."""
    T = 60  # C
    I = 2000  # current
    i = I / cell.cell_area
    overpotentials = cell.calc_overpotentials(i, T)

    # Activation (cathode/anode), Ohmic, Concentration
    assert len(overpotentials) == 4

    # Should all be gte zero
    assert all([v >= 0 for v in overpotentials])


def test_calc_cell_voltage(cell: Cell):
    """Should calculate cell voltage"""
    T = 60  # C
    I = 2000  # current
    E_rev = cell.calc_reversible_voltage()

    V_cell = cell.calc_cell_voltage(I, T)

    # should be higher than reversible cell voltage due to overpotentials
    assert V_cell > E_rev

    # should increase with current
    I2 = 2500
    V_cell_I2 = cell.calc_cell_voltage(I2, T)
    assert V_cell_I2 > V_cell

    # should decrease with temperature
    T2 = 100
    V_cell_T2 = cell.calc_cell_voltage(I, T2)
    assert V_cell_T2 < V_cell


def test_calc_faradaic_efficiency(cell: Cell):
    """Should calculate Faraday's efficiency."""
    I = 500

    # should increase with current
    eta = cell.calc_faradaic_efficiency(60, I)

    I2 = 2000
    eta2 = cell.calc_faradaic_efficiency(60, I2)

    assert eta2 > eta

    # should approach 1 as current increases
    I3 = 5000
    eta3 = cell.calc_faradaic_efficiency(60, I3)

    assert_almost_equal(eta3, 0.996, decimal=3)


def test_calc_mass_flow_rate(cell: Cell):
    """Should calculate the mass flow rate of H2."""
    I = 1000

    mfr = cell.calc_mass_flow_rate(60, I)

    # should increase with current
    I2 = 2000
    mfr2 = cell.calc_mass_flow_rate(60, I2)

    assert mfr2 > mfr
