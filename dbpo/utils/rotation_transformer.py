"""
Rotation representation transformer for absolute action spaces.
Supports axis-angle ↔ rotation_6d conversion used in Robomimic tasks.
"""
import numpy as np


class RotationTransformer:
    """
    Converts between rotation representations.

    Supported conversions:
      axis_angle  <->  rotation_6d

    Parameters
    ----------
    from_rep : str
        Source representation. Currently supports ``"axis_angle"``.
    to_rep : str
        Target representation. Currently supports ``"rotation_6d"``.
    """

    def __init__(self, from_rep: str = "axis_angle", to_rep: str = "rotation_6d"):
        self.from_rep = from_rep
        self.to_rep = to_rep
        if from_rep == "axis_angle" and to_rep == "rotation_6d":
            self._forward_fn = self._axis_angle_to_rot6d
            self._inverse_fn = self._rot6d_to_axis_angle
        elif from_rep == "rotation_6d" and to_rep == "axis_angle":
            self._forward_fn = self._rot6d_to_axis_angle
            self._inverse_fn = self._axis_angle_to_rot6d
        else:
            raise NotImplementedError(
                f"Conversion from '{from_rep}' to '{to_rep}' is not implemented."
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the forward rotation conversion."""
        return self._forward_fn(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Apply the inverse rotation conversion."""
        return self._inverse_fn(x)

    @staticmethod
    def _axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle rotation vectors to 6D rotation representation.

        Input:  [..., 3]
        Output: [..., 6]

        The 6D representation uses the first two columns of the rotation matrix,
        stored in row-major order: [r1, r2] where r1, r2 are 3-vectors.
        """
        from scipy.spatial.transform import Rotation
        batch_shape = axis_angle.shape[:-1]
        rot = Rotation.from_rotvec(axis_angle.reshape(-1, 3))
        mat = rot.as_matrix()           # (N, 3, 3)
        rot6d = mat[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        return rot6d.reshape(batch_shape + (6,))

    @staticmethod
    def _rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
        """
        Convert 6D rotation representation back to axis-angle.

        Input:  [..., 6]
        Output: [..., 3]
        """
        from scipy.spatial.transform import Rotation
        batch_shape = rot6d.shape[:-1]
        r6 = rot6d.reshape(-1, 6)
        a1 = r6[:, :3]
        a2 = r6[:, 3:6]
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
        b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2)
        mat = np.stack([b1, b2, b3], axis=-1)   # (N, 3, 3)
        rot = Rotation.from_matrix(mat)
        axis_angle = rot.as_rotvec()
        return axis_angle.reshape(batch_shape + (3,))
