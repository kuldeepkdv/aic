#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Vision-guided cable insertion policy.

Perception
----------
At each tick the three wrist cameras publish synchronised images in the
Observation message. A classical port detector runs on each image; all
detections with sufficient confidence feed a linear DLT triangulator
that uses the TF extrinsics (optical frames relative to base_link) and
the camera intrinsics from CameraInfo to produce a 3D port position in
the robot base frame.

Motion planning
---------------
A three-phase planner then drives the TCP:

    APPROACH - move to a stand-off pose above the port
    ALIGN    - rotate the gripper so the plug axis matches the port
               axis, and servo lateral xy error to zero
    INSERT   - descend linearly along the port axis until either the
               configured depth is reached or the force sensor reports
               contact.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from geometry_msgs.msg import Pose
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from aic_example_policies.perception import (
    CameraView,
    PortDetector,
    Triangulator,
)
from aic_example_policies.planning import MotionPlanner, Phase, PortEstimate
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task


# Name-based fallbacks, used only when the Image message doesn't carry a
# usable header.frame_id. The authoritative source is the image message.
_CAMERA_FRAME_FALLBACKS = {
    "left": (
        "left_camera/camera_optical_frame",
        "left_camera/camera_link",
    ),
    "center": (
        "center_camera/camera_optical_frame",
        "center_camera/camera_link",
    ),
    "right": (
        "right_camera/camera_optical_frame",
        "right_camera/camera_link",
    ),
}


class VisionGuidedInsert(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._detector = PortDetector()
        self._planner = MotionPlanner()
        self._task: Optional[Task] = None
        # Per-camera reprojected prior from the most recent good 3D
        # estimate. Primes the detector so it stays locked on the port
        # even when other dark apertures are visible.
        self._prior_px: Dict[str, Tuple[float, float]] = {}
        # Warn once per missing optical frame.
        self._frame_warned: set = set()
        self.get_logger().info("VisionGuidedInsert.__init__()")

    # ---- perception --------------------------------------------------
    def _build_views(
        self, obs: Observation
    ) -> List[Tuple[str, CameraView]]:
        """Detect the port in each camera and pair with its extrinsics."""
        images = {
            "left": (obs.left_image, obs.left_camera_info),
            "center": (obs.center_image, obs.center_camera_info),
            "right": (obs.right_image, obs.right_camera_info),
        }
        views: List[Tuple[str, CameraView]] = []
        for name, (img, info) in images.items():
            if img.width == 0:
                continue
            if len(info.k) < 9 or info.k[0] <= 0.0:
                # Uncalibrated CameraInfo — fx must be positive.
                continue

            det = self._detector.detect(img, prior_px=self._prior_px.get(name))
            if det is None or det.confidence < 0.3:
                continue

            T_base_cam = self._lookup_optical_frame(name, img)
            if T_base_cam is None:
                continue

            K = Triangulator.k_from_camera_info(info)
            views.append(
                (name, CameraView(K=K, T_base_cam=T_base_cam, uv=np.array([det.u, det.v])))
            )
        return views

    def _lookup_optical_frame(self, cam_name: str, img) -> Optional[np.ndarray]:
        """Return 4x4 base_link <- optical frame for this camera.

        Preference order:
          1. img.header.frame_id (source of truth from the driver)
          2. Known optical frame name from the fallback table
          3. Body camera_link frame + an implicit REP-103 optical rotation

        Uses the image timestamp for the TF query when available, with a
        short tolerance; falls back to the latest transform if that
        timestamp isn't yet buffered.
        """
        buf = self._parent_node._tf_buffer
        header_frame = img.header.frame_id
        stamp = img.header.stamp

        frames: List[str] = []
        if header_frame:
            frames.append(header_frame)
        for f in _CAMERA_FRAME_FALLBACKS.get(cam_name, ()):
            if f not in frames:
                frames.append(f)

        for frame in frames:
            T = self._lookup_tf_at_stamp(buf, frame, stamp)
            if T is None:
                continue
            is_body = frame.endswith("camera_link")
            if is_body:
                if cam_name not in self._frame_warned:
                    self.get_logger().warn(
                        f"[{cam_name}] optical frame not published; using "
                        f"'{frame}' + REP-103 body->optical rotation. Verify "
                        f"the assumption or publish a camera_optical_frame."
                    )
                    self._frame_warned.add(cam_name)
                T = T @ _BODY_TO_OPTICAL
            return T
        return None

    @staticmethod
    def _lookup_tf_at_stamp(buf, frame: str, stamp) -> Optional[np.ndarray]:
        """TF lookup at the image stamp with a small timeout; if the
        stamp isn't buffered yet, retry with the latest transform."""
        try:
            tf = buf.lookup_transform(
                "base_link",
                frame,
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.05),
            )
            return _transform_to_matrix(tf.transform)
        except TransformException:
            pass
        try:
            tf = buf.lookup_transform("base_link", frame, Time())
            return _transform_to_matrix(tf.transform)
        except TransformException:
            return None

    def _update_priors(
        self, X_base: np.ndarray, named_views: List[Tuple[str, CameraView]]
    ) -> None:
        """Reproject the latest 3D port estimate into each camera and
        store it as the per-camera pixel prior for the next tick."""
        for name, v in named_views:
            P = Triangulator.projection_matrix(v.K, v.T_base_cam)
            ph = P @ np.append(X_base, 1.0)
            if abs(ph[2]) < 1e-9:
                continue
            self._prior_px[name] = (float(ph[0] / ph[2]), float(ph[1] / ph[2]))

    # ---- state helpers ----------------------------------------------
    def _tcp_pose_array(self, obs: Observation) -> np.ndarray:
        p = obs.controller_state.tcp_pose.position
        q = obs.controller_state.tcp_pose.orientation
        return np.array([p.x, p.y, p.z, q.w, q.x, q.y, q.z], dtype=np.float64)

    def _wrist_force_magnitude(self, obs: Observation) -> float:
        f = obs.wrist_wrench.wrench.force
        return float(np.linalg.norm([f.x, f.y, f.z]))

    def _plug_tip_in_base(self) -> Optional[np.ndarray]:
        """Best-effort lookup of the plug tip; absence is non-fatal."""
        if self._task is None:
            return None
        frame = f"{self._task.cable_name}/{self._task.plug_name}_link"
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", frame, Time()
            )
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z])
        except TransformException:
            return None

    def _port_orientation_guess(self, position: np.ndarray) -> np.ndarray:
        """Without ground truth we assume the port's insertion axis
        points along world +z. Participants with a richer perception
        stack can replace this with a PnP solution on the detected
        rectangle corners."""
        return np.array([1.0, 0.0, 0.0, 0.0])   # identity quaternion (wxyz)

    # ---- main task --------------------------------------------------
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"VisionGuidedInsert.insert_cable() task: {task}")
        self._task = task
        self._planner.reset()

        port: Optional[PortEstimate] = None
        last_good_port: Optional[PortEstimate] = None
        start = self.time_now()
        timeout = Duration(seconds=float(max(task.time_limit, 30)))

        while (self.time_now() - start) < timeout:
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.05)
                continue

            # --- perception: triangulate port position ---
            named_views = self._build_views(obs)
            views = [v for _, v in named_views]
            if len(views) >= 2:
                X = Triangulator.triangulate(views)
                if X is not None:
                    err = Triangulator.reprojection_error(X, views)
                    if err < 6.0:
                        q = self._port_orientation_guess(X)
                        port = PortEstimate(position=X, orientation_wxyz=q)
                        last_good_port = port
                        self._update_priors(X, named_views)
                        d, _ = Triangulator.distance_to_camera(
                            X, views[0].T_base_cam
                        )
                        send_feedback(
                            f"port @ {X[0]:.3f},{X[1]:.3f},{X[2]:.3f} "
                            f"d={d:.3f}m reproj={err:.2f}px n={len(views)}"
                        )
            # Use the most recent confident estimate while perception
            # is briefly unreliable (e.g. motion blur, occlusion).
            if port is None:
                port = last_good_port

            if port is None:
                self.sleep_for(0.05)
                continue

            # --- planning: next TCP pose ---
            tcp = self._tcp_pose_array(obs)
            plug = self._plug_tip_in_base()
            force_n = self._wrist_force_magnitude(obs)
            next_pose: Pose = self._planner.step(
                tcp_pose_wxyz=tcp,
                port=port,
                plug_tip_xyz=plug,
                force_magnitude_n=force_n,
            )
            self.set_pose_target(move_robot=move_robot, pose=next_pose)

            if self._planner.phase == Phase.DONE:
                send_feedback("insertion complete")
                self.sleep_for(2.0)
                return True

            self.sleep_for(0.05)

        self.get_logger().warn("VisionGuidedInsert timed out")
        return self._planner.phase == Phase.DONE


# --- tf helpers ------------------------------------------------------

# Rotation that maps a body frame (x-forward, y-left, z-up) to an
# optical frame (z-forward, x-right, y-down). REP-103.
_BODY_TO_OPTICAL = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _transform_to_matrix(t) -> np.ndarray:
    """geometry_msgs/Transform -> 4x4 homogeneous matrix."""
    from transforms3d._gohlketransforms import quaternion_matrix

    q = (t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z)
    M = quaternion_matrix(q)
    M[0, 3] = t.translation.x
    M[1, 3] = t.translation.y
    M[2, 3] = t.translation.z
    return M
