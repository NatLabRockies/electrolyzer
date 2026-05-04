from typing import Any

import attrs
import numpy as np
from attrs import Attribute, define


@define(kw_only=True)
class BaseConfig:
    """
    A Mixin class to allow for kwargs overloading when a data class doesn't
    have a specific parameter defined. This allows passing of larger dictionaries
    to a data class without throwing an error.
    """

    @classmethod
    def from_dict(cls, data: dict, strict=True, additional_cls_name: str | None = None):
        """Maps a data dictionary to an ``attrs``-defined class.

        Args:
            data (dict): The data dictionary to be mapped.
            strict (bool): A flag enabling strict parameter processing, meaning that no extra
                parameters may be passed in or an AttributeError will be raised.
            additional_cls_name (str | None): The name of the model class creating the configuration
                data class. Provides an easier to diagnose error message for end users when
                the class name is provided.

        Returns:
            cls: The ``attrs``-defined class.
        """
        # Check for any inputs that aren't part of the class definition
        if strict is True:
            class_attr_names = [a.name for a in cls.__attrs_attrs__]
            extra_args = [d for d in data if d not in class_attr_names]
            if len(extra_args):
                if additional_cls_name is not None:
                    msg = (
                        f"{additional_cls_name} setup failed as a result of {cls.__name__}"
                        f" receiving extraneous inputs: {extra_args}"
                    )
                else:
                    msg = (
                        f"The initialization for {cls.__name__} was given extraneous "
                        f"inputs: {extra_args}"
                    )
                raise AttributeError(msg)

        kwargs = {a.name: data[a.name] for a in cls.__attrs_attrs__ if a.name in data and a.init}

        # Map the inputs must be provided: 1) must be initialized, 2) no default value defined
        required_inputs = [
            a.name for a in cls.__attrs_attrs__ if a.init and a.default is attrs.NOTHING
        ]
        undefined = sorted(set(required_inputs) - set(kwargs))

        if undefined:
            if additional_cls_name is not None:
                msg = (
                    f"{additional_cls_name} setup failed as a result of {cls.__name__}"
                    f" missing the following inputs: {undefined}"
                )
            else:
                msg = (
                    f"The class definition for {cls.__name__} is missing the following inputs: "
                    f"{undefined}"
                )
            raise AttributeError(msg)
        return cls(**kwargs)

    def as_dict(self) -> dict:
        """Creates a JSON and YAML friendly dictionary that can be save for future reloading.
        This dictionary will contain only `Python` types that can later be converted to their
        proper `Turbine` formats.

        Returns:
            dict: All key, value pairs required for class re-creation.
        """
        return attrs.asdict(self, filter=attr_filter, value_serializer=attr_serializer)


def attr_serializer(inst: type, field: Attribute, value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def attr_filter(inst: Attribute, value: Any) -> bool:
    if inst.init is False:
        return False
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
    return True
