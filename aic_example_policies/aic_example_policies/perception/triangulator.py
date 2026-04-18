#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Multi-view triangulation for port distance estimation.

Given N >= 2 simultaneous detections of the same port from the left,
center, and right wrist cameras (each with its own intrinsics K and its
pose in the robot base frame), this module recovers a 3D point in
base_link via linear DLT triangulation. It also supplies a surface-
normal estimate derived from the oriented rect from the center view.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sensor_msgs.msg import CameraInfo


@dataclass
class CameraView:
    """One camera view used for triangulation."""

    K: np.ndarray            # 3x3 intrinsics
    T_base_cam: np.ndarray   # 4x4 camera frame -> base_link
    uv: np.ndarray           # (2,) pixel observation of the port centroid


class Triangulator:
    @staticmethod
    def k_from_camera_info(info: CameraInfo) -> np.ndarray:
        return np.asarray(info.k, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def projection_matrix(K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
        """P maps a base_link-frame point to pixels of this camera.

        Projection uses the optical convention (z-forward). The camera
        poses supplied to this function must therefore be the poses of
        the optical frames of each camera.
        """
        T_cam_base = np.linalg.inv(T_base_cam)
        return K @ T_cam_base[:3, :]

    @classmethod
    def triangulate(cls, views: Sequence[CameraView]) -> Optional[np.ndarray]:
        """Return 3D point in base_link via linear DLT, or None if ill-posed."""
        if len(views) < 2:
            return None
        rows = []
        for v in views:
            P = cls.projection_matrix(v.K, v.T_base_cam)
            u, vv = float(v.uv[0]), float(v.uv[1])
            rows.append(u * P[2] - P[0])
            rows.append(vv * P[2] - P[1])
        A = np.stack(rows, axis=0)
        # SVD solution
        _, _, vt = np.linalg.svd(A)
        X = vt[-1]
        if abs(X[3]) < 1e-9:
            return None
        return X[:3] / X[3]

    @staticmethod
    def reprojection_error(X_base: np.ndarray, views: Sequence[CameraView]) -> float:
        """Mean pixel reprojection error — use as a sanity check on the
        triangulation quality before acting on it."""
        errs: List[float] = []
        X_h = np.append(X_base, 1.0)
        for v in views:
            P = Triangulator.projection_matrix(v.K, v.T_base_cam)
            p = P @ X_h
            if abs(p[2]) < 1e-9:
                return float("inf")
            u = p[0] / p[2]
            vv = p[1] / p[2]
            errs.append(float(np.hypot(u - v.uv[0], vv - v.uv[1])))
        return float(np.mean(errs))

    @staticmethod
    def distance_to_camera(
        X_base: np.ndarray, T_base_cam: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Return (depth along camera z, point in camera frame)."""
        T_cam_base = np.linalg.inv(T_base_cam)
        X_cam = T_cam_base[:3, :3] @ X_base + T_cam_base[:3, 3]
        return float(X_cam[2]), X_cam
