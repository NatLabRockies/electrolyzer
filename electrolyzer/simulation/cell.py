# This will contain the baseclass for different types of cells
"""This module defines a Hydrogen Electrolyzer Cell."""

from attrs import define

from electrolyzer.tools.type_dec import FromDictMixin


@define
class Cell(FromDictMixin):

    # n: int = 2

    # Constants
    # TODO: change to z_c
    z: int = 2  # number of electrons transferred in reaction
    F: float = 96485.34  # Faraday's Constant (C/mol) or [As/mol]
    R: float = 8.314  # Ideal Gas Constant (J/mol/K)

    M_H: float = 1.00784  # molecular weight of Hydrogen [g/mol]
    M_H2: float = 2.016  # [g/mol]
    M_O: float = 15.999  # molecular weight of Oxygen [g/mol]
    M_O2: float = 31.999  # [g/mol]
    M_K: float = 39.0983  # molecular weight of Potassium [g/mol]

    lhv: float = 33.33  # lower heating value of H2 [kWh/kg]
    hhv: float = 39.41  # higher heating value of H2 [kWh/kg]
    gibbs: float = 237.24e3  # Gibbs Energy of global reaction (J/mol)
