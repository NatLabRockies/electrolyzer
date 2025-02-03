"""This module defines a Hydrogen Electrolyzer Cluster."""

from typing import Union

import numpy as np
import scipy
import pandas as pd
import rainflow
from attrs import field, define
from scipy.signal import tf2ss, cont2discrete

from electrolyzer.simulation.stack import Stack
from electrolyzer.tools.type_dec import NDArrayFloat, FromDictMixin, array_converter
from electrolyzer.tools.validators import contains

@define
class Cluster(FromDictMixin):
    stack_type: str = field()
    
    stack: Stack = field(init=False)