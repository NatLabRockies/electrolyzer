"""This module provides unit tests for `Stack`."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from electrolyzer.simulation.stack import Stack
from electrolyzer.simulation.cell_models.pem import PEMCell


def create_stack():
    stack_dict = {
        "dt": 1,
        "cell_type": "PEM",
        "max_current": 2000,
        "temperature": 60,
        "n_cells": 100,
        # "stack_rating_kW": 750,
        "degradation": {
            "eol_eff_percent_loss": 10,
            "PEM_params": {
                "rate_steady": 1.41737929e-10,
                "rate_fatigue": 3.33330244e-07,
                "rate_onoff": 1.47821515e-04,
            },
        },
        "cell_params": {
            "cell_type": "PEM",
            "PEM_params": {
                "cell_area": 1000,
                "turndown_ratio": 0.1,
                "max_current_density": 2,
                "p_anode": 1.01325,
                "p_cathode": 30,
                "alpha_a": 2,
                "alpha_c": 0.5,
                "i_0_a": 2.0e-7,
                "i_0_c": 2.0e-3,
                "e_m": 0.02,
                "R_ohmic_elec": 50.0e-3,
                "f_1": 250,
                "f_2": 0.996,
            },
        },
    }
    return Stack.from_dict(stack_dict)


@pytest.fixture
def stack():
    return create_stack()


def test_init(mocker):
    """Should initialize properly."""

    # mock side effects (these will have their own unit tests)
    spy_calc_state_space = mocker.spy(Stack, "calc_state_space")
    spy_create_polarization = mocker.spy(Stack, "create_polarization")
    spy_cell = mocker.spy(PEMCell, "from_dict")

    # for this example, set stack rating explicitly
    stack_dict = {
        "dt": 1,
        "cell_type": "PEM",
        "max_current": 2000,
        "temperature": 60,
        "n_cells": 100,
        "stack_rating_kW": 750,
        "degradation": {
            "eol_eff_percent_loss": 10,
            "PEM_params": {
                "rate_steady": 1.41737929e-10,
                "rate_fatigue": 3.33330244e-07,
                "rate_onoff": 1.47821515e-04,
            },
        },
        "cell_params": {
            "cell_type": "PEM",
            "PEM_params": {
                "cell_area": 1000,
                "turndown_ratio": 0.1,
                "max_current_density": 2,
                "p_anode": 1.01325,
                "p_cathode": 30,
                "alpha_a": 2,
                "alpha_c": 0.5,
                "i_0_a": 2.0e-7,
                "i_0_c": 2.0e-3,
                "e_m": 0.02,
                "R_ohmic_elec": 50.0e-3,
                "f_1": 250,
                "f_2": 0.996,
            },
        },
        "hydrogen_degradation_penalty": False,
    }

    stack = Stack.from_dict(stack_dict)

    assert stack.n_cells == stack_dict["n_cells"]
    assert stack.cell.cell_area == stack_dict["cell_params"]["PEM_params"]["cell_area"]
    assert stack.temperature == stack_dict["temperature"]
    assert stack.max_current == stack_dict["max_current"]

    # case: stack rating provided by user
    assert stack.stack_rating_kW == stack_dict["stack_rating_kW"]
    assert stack.stack_rating == stack_dict["stack_rating_kW"] * 1e3
    assert stack.min_power == 0.1 * stack.stack_rating

    assert stack.include_degradation_penalty is True
    assert (
        stack.hydrogen_degradation_penalty == stack_dict["hydrogen_degradation_penalty"]
    )
    assert stack.rf_track == 0.0
    assert stack.V_degradation == 0.0
    assert stack.uptime == 0.0
    assert stack.cycle_count == 0
    assert stack.fatigue_history == 0.0
    assert stack.hourly_counter == 0.0
    assert stack.hour_change is False
    assert_array_equal(stack.voltage_signal, [])
    assert_array_equal(stack.voltage_history, [])
    assert stack.stack_on is False
    assert stack.stack_waiting is False
    assert stack.turn_on_delay == 600
    assert stack.turn_on_time == 0
    assert stack.turn_off_time < 0
    assert stack.wait_time > 0
    assert stack.dt == 1.0
    assert stack.time == 0
    assert stack.tau == 5.0
    assert stack.stack_state == 0.0

    # these side effects are tested in isolation elsewhere, here we just want to
    # make sure they get called properly
    spy_calc_state_space.assert_called_once()
    spy_create_polarization.assert_called_once()
    spy_cell.assert_called_once()


def test_run(mocker):
    stack = create_stack()

    spy_update_deg = mocker.spy(Stack, "update_degradation")
    spy_calc_p = mocker.spy(Stack, "calc_stack_power")
    spy_calc_mfr = mocker.spy(PEMCell, "calc_mass_flow_rate")
    spy_update_dynamics = mocker.spy(Stack, "update_dynamics")
    spy_update_status = mocker.spy(Stack, "update_status")

    # stack off
    orig_state = stack.stack_state
    stack.run(stack.stack_rating)

    spy_update_deg.assert_not_called()
    spy_calc_p.assert_not_called()
    spy_calc_mfr.assert_not_called()
    spy_update_dynamics.assert_called_once()
    spy_update_status.assert_called_once()

    assert stack.time == stack.dt
    assert stack.V_degradation == 0
    assert stack.stack_state == orig_state
    assert len(stack.voltage_history) == 1

    # stack on
    stack = create_stack()
    mocker.resetall()
    stack.stack_on = True
    stack.run(stack.stack_rating)

    spy_update_deg.assert_called_once()
    spy_calc_p.assert_called_once()
    spy_calc_mfr.assert_called_once()
    spy_update_dynamics.assert_called_once()

    assert stack.stack_state != orig_state

    # degradation applied from prev timestep
    stack.run(stack.stack_rating)
    assert stack.V_degradation != 0

    # fast forward an hour
    stack.time = 3600
    stack.run(stack.stack_rating)
    assert stack.hourly_counter == 1
    assert stack.hour_change is True
    assert len(stack.voltage_signal) == 3
    assert_array_equal(stack.voltage_history, [])


def test_create_polarization(stack: Stack):
    # TODO remake this
    """
    Should create a polarization curve based on fit for the specified model over a
    range of temperatures.
    """
    fit_params = stack.create_polarization()

    # this is brittle, so for now use a lenient precision check
    # expected = [
    #    -2.28261081e-03,
    #    -1.50848325e-02,
    #    7.89259537e-03,
    #    4.80671306e00,
    #    9.74923247e-01,
    #    1.36179580e01,
    # ]
    expected = [
        -1.99627103e-03,
        -1.27808907e-02,
        4.76371760e-03,
        4.98846103e00,
        7.96586249e-01,
        1.11445867e01,
    ]

    assert_array_almost_equal(fit_params, expected, decimal=2)


def test_convert_power_to_current(stack: Stack):
    """Converts an input power to stack current."""
    current = stack.convert_power_to_current(stack.stack_rating_kW, stack.temperature)

    # We don't get exactly max current out, for now check within 5% tolerance
    tol = 0.05
    assert abs(current - stack.max_current) < (tol * stack.max_current)

    # TODO: users should enter a power <= rated power, but should
    # this be enforced in the code? Potentially apply curtailment rules once
    # those have been developed.


def test_curtail_power():
    """TODO: This method is still being implemented."""
    pass


def test_calc_fatigue_degradation(stack: Stack):
    """Should return a voltage penalty calculated based on a fatigue analysis."""
    temperature = 60
    v_stack = (
        stack.cell.calc_cell_voltage(stack.max_current, temperature) * stack.n_cells
    )
    signal = np.linspace(0, v_stack, 100)  # 100 sec voltage ramp
    penalty = stack.calc_fatigue_degradation(signal)

    assert penalty > 0 and penalty < v_stack
    rf_track = stack.rf_track
    assert rf_track > 0

    # Longer signal over same range should lead to higher penalty
    signal2 = np.linspace(0, signal, 500)
    stack.rf_track = 0
    penalty2 = stack.calc_fatigue_degradation(signal2)

    assert penalty2 > penalty
    assert stack.rf_track > rf_track


def test_calc_steady_degradation(stack: Stack):
    """Should return a voltage penalty as a function of voltage history"""
    stack.stack_on = True

    penalty_0s = stack.d_s

    assert penalty_0s == 0

    # Ramp power input from 0 to rated
    t = 100
    P_in = np.linspace(0, stack.stack_rating, t)
    for p in P_in:
        stack.run(p)

    # steady degradation should be greater than 0
    penalty_100s = stack.d_s

    # steady degradation should be less than if operating at rated
    max_voltage = stack.cell.calc_cell_voltage(stack.max_current, stack.temperature)
    penalty_max = max_voltage * t * stack.rate_steady

    assert penalty_100s > penalty_0s
    assert penalty_100s < penalty_max


def test_calc_onoff_degradation(stack: Stack):
    """Should return a voltage penalty as a function of on/off cycles."""
    penalty = stack.calc_onoff_degradation()

    assert penalty == 0

    # penalty should increase with cycles
    stack.cycle_count = 10
    penalty_10c = stack.calc_onoff_degradation()

    assert penalty_10c > penalty


def test_update_degradation(mocker):
    # mock side effects, as these are unit tested elsewhere
    spy_calc_fatigue = mocker.spy(Stack, "calc_fatigue_degradation")
    spy_calc_steady = mocker.spy(Stack, "calc_steady_degradation")
    spy_calc_onoff = mocker.spy(Stack, "calc_onoff_degradation")

    temperature = 60
    stack = create_stack()
    v_cell = stack.cell.calc_cell_voltage(stack.max_current, temperature)
    v_stack = v_cell * stack.n_cells
    stack.update_degradation()

    # at startup, no voltage signal, hasn't been running for an hour
    spy_calc_fatigue.assert_not_called()
    spy_calc_steady.assert_called_once()
    spy_calc_onoff.assert_called_once()
    assert stack.fatigue_history == 0
    assert stack.V_degradation == 0

    # after an hour, but voltage difference is too low for fatigue
    mocker.resetall()
    stack = create_stack()
    signal = np.repeat([v_stack], 100)  # 100 sec steady voltage
    stack.voltage_signal = signal
    stack.cell_voltage = v_cell
    stack.hour_change = True
    stack.update_degradation()

    spy_calc_fatigue.assert_not_called()
    spy_calc_steady.assert_called_once()
    spy_calc_onoff.assert_called_once()
    assert stack.fatigue_history == 0
    assert stack.V_degradation > 0

    # after an hour, with sufficient voltage diff
    mocker.resetall()
    stack = create_stack()
    signal = np.linspace(0, v_stack, 100)  # 100 sec voltage ramp
    stack.voltage_signal = signal
    stack.cell_voltage = v_cell
    stack.hour_change = True
    stack.update_degradation()

    spy_calc_fatigue.assert_called_once()
    assert_array_equal(spy_calc_fatigue.call_args.args[1], signal)
    spy_calc_steady.assert_called_once()
    spy_calc_onoff.assert_called_once()
    assert stack.fatigue_history > 0
    assert stack.V_degradation > 0


def test_update_temperature(stack: Stack):
    """TODO: This method is still being implemented."""
    assert stack.update_temperature(0, 0) == stack.temperature


def test_update_dynamics(stack: Stack):
    """
    Should update stack state and apply H2 MFR filter to simulate dynamic response.
    """
    H2_mfr = (
        stack.cell.calc_mass_flow_rate(stack.temperature, stack.max_current)
        * stack.n_cells
    )
    new_state, filtered_H2_mfr = stack.update_dynamics(H2_mfr, stack.stack_state)

    # filtered state should be lower than
    assert new_state != stack.stack_state
    assert filtered_H2_mfr < H2_mfr

    new_state2, filtered_H2_mfr2 = stack.update_dynamics(H2_mfr, new_state)

    # should ramp up if we feed the same MFR again
    assert new_state2 != new_state
    assert H2_mfr > filtered_H2_mfr2 > filtered_H2_mfr


def test_calc_state_space(stack: Stack):
    """Should produce a discretized state space system representing stack operation."""
    ss_d = stack.calc_state_space()

    assert len(ss_d) == 4

    expected = [[[0.81873075]], [[0.90634623]], [[0.2]], [[0.0]]]
    assert_array_almost_equal(ss_d, expected)
    assert_array_almost_equal(stack.DTSS, expected)


def test_update_status(stack: Stack):
    """Should turn a stack on if it has waited long enough."""

    # Should be off to start
    stack.stack_waiting = True
    stack.update_status()
    assert stack.stack_on is False

    # initiate startup
    stack.turn_stack_on()
    # fast forward one step past waiting time
    stack.time = stack.turn_on_delay + stack.dt
    stack.update_status()
    assert stack.stack_on is True


def test_turn_stack_off(stack: Stack):
    """Should turn stack off."""
    stack.time = 100
    # should not do anything if stack is on
    stack.turn_stack_off()
    assert stack.turn_off_time != stack.time
    assert stack.cycle_count == 0

    stack.stack_on = True
    stack.time = 800
    stack.turn_stack_off()
    assert stack.turn_off_time == stack.time
    assert stack.stack_on is False
    assert stack.stack_waiting is False
    assert stack.cycle_count == 1
    assert stack.wait_time == 0


def test_turn_stack_on(stack: Stack):
    """Should initiate the stack startup process."""
    stack.turn_stack_on()

    assert stack.turn_on_time == stack.time
    assert stack.stack_waiting is True
    assert stack.wait_time == stack.turn_on_delay

    # should simply return if already on
    stack.time = 500
    stack.stack_on = True
    assert stack.turn_on_time != stack.time


def test_calc_stack_power(stack: Stack):
    """Should calculate stack power for a given DC current."""
    assert stack.calc_stack_power(stack.max_current) == stack.stack_rating_kW


def test_calc_electrolysis_efficiency(stack: Stack):
    # TODO: remake this test - something is weird about it
    """
    Should calculate values of electrolysis efficiency for given DC Power input and MFR.
    """
    H2_mfr = (
        stack.cell.calc_mass_flow_rate(stack.temperature, stack.max_current * 0.8)
        * stack.n_cells
    )
    eta_values = stack.calc_electrolysis_efficiency(
        stack.stack_rating_kW, H2_mfr * 3600
    )

    assert len(eta_values) == 3

    # efficiency should decrease as we approach max current due to overpotentials
    assert eta_values[0] > 70  # highest efficiency around 80% capacity
    H2_mfr2 = (
        stack.cell.calc_mass_flow_rate(stack.temperature, stack.max_current)
        * stack.n_cells
    )
    eta_values2 = stack.calc_electrolysis_efficiency(
        stack.stack_rating_kW, H2_mfr2 * 3600
    )
    assert eta_values2[0] < eta_values[0]


def test_dt_behavior():
    stack_dict = {
        "dt": 1,
        "cell_type": "PEM",
        "max_current": 2000,
        "temperature": 60,
        "n_cells": 100,
        # "stack_rating_kW": 750,
        "degradation": {
            "eol_eff_percent_loss": 10,
            "PEM_params": {
                "rate_steady": 1.41737929e-10,
                "rate_fatigue": 3.33330244e-07,
                "rate_onoff": 1.47821515e-04,
            },
        },
        "cell_params": {
            "cell_type": "PEM",
            "PEM_params": {
                "cell_area": 1000,
                "turndown_ratio": 0.1,
                "max_current_density": 2,
                "p_anode": 1.01325,
                "p_cathode": 30,
                "alpha_a": 2,
                "alpha_c": 0.5,
                "i_0_a": 2.0e-7,
                "i_0_c": 2.0e-3,
                "e_m": 0.02,
                "R_ohmic_elec": 50.0e-3,
                "f_1": 250,
                "f_2": 0.996,
            },
        },
    }

    stack1 = Stack.from_dict(stack_dict)
    stack1.cell_voltage = 1.5

    stack_dict["dt"] = 60
    stack60 = Stack.from_dict(stack_dict)
    stack60.cell_voltage = 1.5

    stack_dict["dt"] = 3600
    stack3600 = Stack.from_dict(stack_dict)
    stack3600.cell_voltage = 1.5

    # Check that timescale specific attributes were initialized correctly
    assert stack1.dt == 1
    assert stack60.dt == 60
    assert stack3600.dt == 3600

    assert stack1.turn_on_delay == 600
    assert stack60.turn_on_delay == 600
    assert stack3600.turn_on_delay == 0

    assert stack1.wait_time > 0
    assert stack60.wait_time > 0
    assert stack3600.wait_time == 0

    # The steady degfradation calculation should change with the change in dt
    stack1.calc_steady_degradation()
    stack60.calc_steady_degradation()
    stack3600.calc_steady_degradation()

    assert stack1.d_s == 2.126068935e-10
    assert stack60.d_s == 1.2756413610000001e-08
    assert stack3600.d_s == 7.653848166e-07

    # stack.update_dynamics() should only perform state space calculations if dt is
    # small enough
    assert not stack1.ignore_dynamics
    assert stack60.ignore_dynamics
    assert stack3600.ignore_dynamics

    H2_mfr = 0.0015
    stack_state = 0.001

    stack1.stack_state = stack_state
    stack60.stack_state = stack_state
    stack3600.stack_state = stack_state

    next_state1, H2_mfr1 = stack1.update_dynamics(H2_mfr, stack_state)
    next_state60, H2_mfr60 = stack60.update_dynamics(H2_mfr, stack_state)
    next_state3600, H2_mfr3600 = stack3600.update_dynamics(H2_mfr, stack_state)

    assert H2_mfr1 != H2_mfr
    assert H2_mfr60 == H2_mfr
    assert H2_mfr3600 == H2_mfr

    assert next_state1 != stack_state
    assert next_state60 == stack_state
    assert next_state3600 == stack_state
