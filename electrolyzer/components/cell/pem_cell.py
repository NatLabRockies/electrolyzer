from attrs import field, define, validators

from electrolyzer.tools.validators import range_val
from electrolyzer.components.cell.cell import CellBaseClass, CellBaseConfig


@define(kw_only=True)
class PEMCellConfig(CellBaseConfig):
    f1: float = field(default=250, validator=validators.ge(0))
    f2: float = field(default=0.996, validator=range_val(0.5, 1.0))
    i_0a: float = field(default=2.0e-7, validator=range_val(0.0, 1.0))
    i_0c: float = field(default=2.0e-3, validator=range_val(0.0, 1.0))
    alpha_a: float = field(default=2.0, validator=range_val(0.0, 4.0))
    alpha_c: float = field(default=0.5, validator=range_val(0.0, 4.0))


class PEMCell(CellBaseClass):
    def setup(self):
        self.config = PEMCellConfig.from_dict(self.options["tech_config"]["cell_parameters"])
        super().setup()
        # Design parameters
        self.add_input("i_0a", val=self.config.i_0a, shape=1, units="A/(cm**2)")
        self.add_input("i_0c", val=self.config.i_0c, shape=1, units="A/(cm**2)")
        self.add_input("f1", val=self.config.f1, shape=1, units="(mA**2)/(cm**4)")
        self.add_input("f2", val=self.config.f2, shape=1, units="unitless")
        self.add_input("alpha_a", val=self.config.alpha_a, shape=1, units="unitless")
        self.add_input("alpha_c", val=self.config.alpha_c, shape=1, units="unitless")
