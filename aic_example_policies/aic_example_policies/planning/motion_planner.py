#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Three-phase motion planner for vision-guided cable insertion.

    APPROACH   Move the gripper to a stand-off pose directly above the
               detected port. Uses smooth linear interpolation from the
               current TCP pose to a target that is `approach_height`
               metres above the port along the port's insertion axis.

    ALIGN      Rotate the TCP so the plug axis is colinear with the
               port axis, zeroing lateral xy error via a proportional
               visual-servo update. The stand-off is maintained.

    INSERT     Command a straight-line descent along the port axis, one
               small step at a time, until either the commanded depth
               is reached or the force sensor signals contact.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from transforms3d._gohlketransforms import (
    quaternion_from_matrix,
    quaternion_matrix,
    quaternion_slerp,
)


class Phase(Enum):
    APPROACH = auto()
    ALIGN = auto()
    INSERT = auto()
    DONE = auto()


@dataclass
class PortEstimate:
    """Port pose + axis in base_link."""

    position: np.ndarray              # (3,) port origin in base_link
    orientation_wxyz: np.ndarray      # (4,) port frame orientation (w,x,y,z)
    insertion_axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1.0]))

    @property
    def axis_base(self) -> np.ndarray:
        """Insertion axis expressed in base_link (unit vector)."""
        R = quaternion_matrix(self.orientation_wxyz)[:3, :3]
        v = R @ self.insertion_axis
        return v / max(np.linalg.norm(v), 1e-9)


@dataclass
class PlannerConfig:
    approach_height_m: float = 0.10          # stand-off above port
    insertion_depth_m: float = 0.02          # total linear travel
    approach_steps: int = 80                 # iterations for APPROACH
    align_steps: int = 40                    # iterations for ALIGN
    insert_step_m: float = 5.0e-4            # per-tick descent in INSERT
    xy_servo_gain: float = 0.6               # proportional visual servo
    xy_servo_clip_m: float = 0.01            # max correction per update
    force_contact_n: float = 8.0             # |F| threshold to stop INSERT


class MotionPlanner:
    """Pure-logic planner: given the current TCP pose + latest port
    estimate, produce the next desired TCP pose. No ROS dependency —
    easy to unit test."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.cfg = config or PlannerConfig()
        self._phase = Phase.APPROACH
        self._tick = 0
        self._descent_m = 0.0

    # ---- public state ------------------------------------------------
    @property
    def phase(self) -> Phase:
        return self._phase

    def reset(self) -> None:
        self._phase = Phase.APPROACH
        self._tick = 0
        self._descent_m = 0.0

    # ---- main step ---------------------------------------------------
    def step(
        self,
        tcp_pose_wxyz: np.ndarray,       # (7,) x,y,z, qw,qx,qy,qz
        port: PortEstimate,
        plug_tip_xyz: Optional[np.ndarray] = None,
        force_magnitude_n: float = 0.0,
    ) -> Pose:
        """Return the next target Pose; advance the phase on completion.

        ``plug_tip_xyz`` is the plug tip in base_link when known; the
        planner uses it to translate TCP targets so the *plug*, not the
        TCP itself, lands on the commanded point.
        """
        tcp_to_plug = (
            plug_tip_xyz - tcp_pose_wxyz[:3]
            if plug_tip_xyz is not None
            else np.zeros(3)
        )

        if self._phase == Phase.APPROACH:
            pose, done = self._approach(tcp_pose_wxyz, port, tcp_to_plug)
            if done:
                self._phase = Phase.ALIGN
                self._tick = 0
        elif self._phase == Phase.ALIGN:
            pose, done = self._align(tcp_pose_wxyz, port, plug_tip_xyz, tcp_to_plug)
            if done:
                self._phase = Phase.INSERT
                self._tick = 0
        elif self._phase == Phase.INSERT:
            pose, done = self._insert(tcp_pose_wxyz, port, tcp_to_plug, force_magnitude_n)
            if done:
                self._phase = Phase.DONE
        else:
            pose = self._pose_from_xyzq(
                tcp_pose_wxyz[:3], tcp_pose_wxyz[3:]
            )
        return pose

    # ---- phase implementations --------------------------------------
    def _approach(self, tcp: np.ndarray, port: PortEstimate, tcp_to_plug: np.ndarray):
        plug_target = port.position + self.cfg.approach_height_m * port.axis_base
        target_xyz = plug_target - tcp_to_plug
        target_q = self._gripper_quat_for_port(port, tcp[3:])

        frac = min(1.0, (self._tick + 1) / self.cfg.approach_steps)
        xyz = (1.0 - frac) * tcp[:3] + frac * target_xyz
        q = quaternion_slerp(tcp[3:], target_q, frac)
        self._tick += 1
        done = self._tick >= self.cfg.approach_steps
        return self._pose_from_xyzq(xyz, q), done

    def _align(
        self,
        tcp: np.ndarray,
        port: PortEstimate,
        plug_tip: Optional[np.ndarray],
        tcp_to_plug: np.ndarray,
    ):
        target_q = self._gripper_quat_for_port(port, tcp[3:])
        q = quaternion_slerp(tcp[3:], target_q, 0.25)

        plug_target = port.position + self.cfg.approach_height_m * port.axis_base
        if plug_tip is not None:
            err = plug_target - plug_tip
            err_lat = err - np.dot(err, port.axis_base) * port.axis_base
            step = np.clip(
                self.cfg.xy_servo_gain * err_lat,
                -self.cfg.xy_servo_clip_m,
                self.cfg.xy_servo_clip_m,
            )
            xyz = tcp[:3] + step
        else:
            stand_off = plug_target - tcp_to_plug
            xyz = 0.5 * tcp[:3] + 0.5 * stand_off

        self._tick += 1
        done = self._tick >= self.cfg.align_steps
        return self._pose_from_xyzq(xyz, q), done

    def _insert(
        self,
        tcp: np.ndarray,
        port: PortEstimate,
        tcp_to_plug: np.ndarray,
        force_n: float,
    ):
        self._descent_m += self.cfg.insert_step_m
        plug_target = (
            port.position
            + (self.cfg.approach_height_m - self._descent_m) * port.axis_base
        )
        target_xyz = plug_target - tcp_to_plug
        target_q = self._gripper_quat_for_port(port, tcp[3:])
        q = quaternion_slerp(tcp[3:], target_q, 0.5)

        done = (
            self._descent_m >= (self.cfg.approach_height_m + self.cfg.insertion_depth_m)
            or force_n > self.cfg.force_contact_n
        )
        return self._pose_from_xyzq(target_xyz, q), done

    # ---- helpers -----------------------------------------------------
    @staticmethod
    def _pose_from_xyzq(xyz: np.ndarray, q_wxyz: np.ndarray) -> Pose:
        return Pose(
            position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
            orientation=Quaternion(
                w=float(q_wxyz[0]),
                x=float(q_wxyz[1]),
                y=float(q_wxyz[2]),
                z=float(q_wxyz[3]),
            ),
        )

    @staticmethod
    def _gripper_quat_for_port(port: PortEstimate, current_q_wxyz: np.ndarray) -> np.ndarray:
        """Choose a gripper orientation that points the tool z-axis along
        the port insertion axis (pointing *into* the port). We keep the
        yaw about the port axis as free as possible by projecting the
        current x-axis onto the plane perpendicular to the port axis."""
        z_axis = port.axis_base
        R_cur = quaternion_matrix(current_q_wxyz)[:3, :3]
        x_cur = R_cur[:, 0]
        x_proj = x_cur - np.dot(x_cur, z_axis) * z_axis
        if np.linalg.norm(x_proj) < 1e-6:
            # Degenerate: pick any perpendicular vector.
            x_proj = np.array([1.0, 0.0, 0.0])
            x_proj = x_proj - np.dot(x_proj, z_axis) * z_axis
        x_axis = x_proj / np.linalg.norm(x_proj)
        y_axis = np.cross(z_axis, x_axis)
        R = np.eye(4)
        R[:3, 0] = x_axis
        R[:3, 1] = y_axis
        R[:3, 2] = z_axis
        return quaternion_from_matrix(R)
