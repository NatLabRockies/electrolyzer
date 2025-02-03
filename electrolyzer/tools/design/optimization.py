import copy

from scipy.optimize import fsolve

from electrolyzer.simulation.stack import Stack


def calc_rated_system(modeling_options: dict):
    """
    Calculates number of stacks and stack power rating (kW) to match a desired
    system rating (MW).

    Args:
        modeling_options (dict): An options Dict compatible with the modeling schema
    """
    options = copy.deepcopy(modeling_options)

    system_rating_kW = options["electrolyzer"]["supervisor"]["system_rating_MW"] * 1e3
    stack_rating_kW = options["electrolyzer"]["stack"]["stack_rating_kW"]

    # determine number of stacks (int) closest to stack rating (float)
    n_stacks = round(system_rating_kW / stack_rating_kW)
    options["electrolyzer"]["supervisor"]["n_stacks"] = n_stacks

    # determine new desired rating to adjust parameters for
    new_rating = system_rating_kW / n_stacks
    options["electrolyzer"]["stack"]["stack_rating_kW"] = new_rating

    # solve for new stack rating (modifies dict)
    calc_rated_stack(options)

    return options


def _solve_rated_stack(
    desired_rating: float, desired_curr_density: float, stack: Stack
):
    cell_area = stack.cell.cell_area
    max_current = stack.max_current

    # root finding function
    def calc_rated_power_diff(x):
        cell_area, max_current = x
        stack.cell.cell_area = cell_area
        stack.max_current = max_current
        p_rated = stack.calc_stack_power(max_current)

        return [
            p_rated - desired_rating,
            max_current / cell_area - desired_curr_density,
        ]

    return fsolve(calc_rated_power_diff, [cell_area, max_current])


def calc_rated_stack(modeling_options: dict):
    """
    For a given model specification, determines a configuration that meets the
    desired stack rating (kW). Only modifies `n_cells` and `cell_area`.

    NOTE: This is a naive approach: it is only concerned with achieving the desired
        power rating. Any checks on the validity of the resulting design must be
        performed by the user.

    Args:
        modeling_options (dict): An options Dict compatible with the modeling schema
    """
    options = modeling_options["electrolyzer"]["stack"]
    options["dt"] = modeling_options["electrolyzer"]["dt"]
    options["cell_params"] = modeling_options["electrolyzer"]["cell_params"]
    options["degradation"] = modeling_options["electrolyzer"]["degradation"]
    stack = Stack.from_dict(options)

    n_cells = stack.n_cells

    # start with an initial calculation of stack power to compare with desired
    stack_p = stack.calc_stack_power(stack.max_current)
    desired_rating = stack.stack_rating_kW
    desired_curr_density = stack.max_current / stack.cell.cell_area

    # nudge cell count up or down until it overshoots
    if stack_p > desired_rating:
        while stack_p > desired_rating:
            n_cells -= 1
            stack.n_cells = n_cells
            stack_p = stack.calc_stack_power(stack.max_current)

    elif stack_p < desired_rating:
        while stack_p < desired_rating:
            n_cells += 1
            stack.n_cells = n_cells
            stack_p = stack.calc_stack_power(stack.max_current)

    # solve for optimal stack
    res = _solve_rated_stack(desired_rating, desired_curr_density, stack)

    # recalc stack power
    stack.cell.cell_area = res[0]
    stack.max_current = res[1]
    stack_p = stack.calc_stack_power(stack.max_current)

    # TODO alkaline cell characteristic area optimization
    modeling_options["electrolyzer"]["cell_params"]["PEM_params"]["cell_area"] = res[0]
    modeling_options["electrolyzer"]["stack"]["max_current"] = res[1]
    modeling_options["electrolyzer"]["stack"]["n_cells"] = n_cells
    modeling_options["electrolyzer"]["stack"]["stack_rating_kW"] = stack_p
