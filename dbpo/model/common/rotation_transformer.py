"""
Compatibility wrapper for the stage-1 rotation transformer.

The stage-1 repository only relies on axis-angle <-> rotation_6d conversion for
robomimic absolute-action handling. Keep the public import path stable while
dropping the optional pytorch3d dependency from the default install surface.
"""

from dbpo.utils.rotation_transformer import RotationTransformer

__all__ = ["RotationTransformer"]
