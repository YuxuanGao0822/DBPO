"""Compatibility namespace for DBPO runner aliases.

This package intentionally avoids eager imports. Individual runner alias
modules, e.g. ``dbpo.envs.runners.robomimic_lowdim_runner``, forward to the
actual implementation under ``dbpo.env_runner``.
"""

__all__: list[str] = []
