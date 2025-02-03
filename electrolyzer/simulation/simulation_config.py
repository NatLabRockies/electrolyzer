from attrs import define, field
from electrolyzer.simulation.base import BaseClass
from electrolyzer.tools.validators import contains
@define
class SimulationConfig(BaseClass):
    feedback_type: str = field(validator=contains(["open-loop","closed-loop"]))
    constant_temperature: bool = field(default = True)
    