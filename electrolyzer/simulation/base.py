from typing import Any, Dict

import attrs

from electrolyzer.tools.type_dec import FromDictMixin


class BaseClass(FromDictMixin):
    """
    BaseClass object class. This class does the logging and MixIn class inheritance.
    """

    @classmethod
    def get_model_defaults(cls) -> Dict[str, Any]:
        """Produces a dictionary of the keyword arguments and their defaults.

        Returns
        -------
        Dict[str, Any]
            Dictionary of keyword argument: default.
        """
        return {el.name: el.default for el in attrs.fields(cls)}

    def _get_model_dict(self) -> dict:
        """Convenience method that wraps the `attrs.asdict` method. Returns the object's
        parameters as a dictionary.

        Returns
        -------
        dict
            The provided or default, if no input provided,
            model settings as a dictionary.
        """
        return attrs.asdict(self)
