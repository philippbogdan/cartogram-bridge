"""Cartograms and numerical checks of their transport and diffusion limits."""


def __getattr__(name):
    if name in ("Cartogram", "Result"):
        from . import api
        return getattr(api, name)
    raise AttributeError(name)
