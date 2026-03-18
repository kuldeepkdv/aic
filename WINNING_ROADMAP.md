# AIC Winning Roadmap
## Computer Vision + AI/ML Solutions for Ubuntu 24.04 + NVIDIA RTX A5000 (16 GB VRAM)

---

## 1. Challenge Summary

The **AI for Industry Challenge (AIC)** requires a UR5e robot arm to insert fiber optic
connectors (SFP modules, SC plugs) into ports on a randomized modular task board, evaluated
entirely inside Gazebo simulation during the Qualification Phase (Mar – May 15, 2026).

### Critical facts about the qualification task

| Item | Detail |
|------|--------|
| Plug already in hand | Robot starts with one plug grasped — **no picking required** |
| Starting distance | Robot arm is already within **a few centimetres** of the target port |
| Trials | 3 per submission: Trial 1 & 2 = SFP→SFP_PORT, Trial 3 = SC_PLUG→SC_PORT |
| Randomisation | Board pose (XY + yaw), NIC card rail + translation + yaw, SC port translation |
| Grasp deviation | ≈ ±2 mm, ±0.04 rad from nominal — policy must be robust to this |
| Time limit | 180 s per trial |
| Evaluation GPU | NVIDIA L4 Tensor Core, 24 GB VRAM |

### Scoring (max ≈ 100 pts/trial)

| Tier | What | Points |
|------|------|--------|
| 1 | Node loads + lifecycle valid | 0–1 |
| 2 | Trajectory smoothness (0-5) + speed (0-10) + efficiency (0-5) − force penalty (−12) − collision penalty (−24) | 0–30 |
| 3 | Full insertion correct port +60, wrong port −10, partial 0–40, proximity 0–25 | −10 to +60 |

**Takeaway:** getting the plug into the port (Tier 3) is worth 60 % of the max score.
Speed (≤ 30 s = ~8/12 pts) and smoothness matter second.

---

## 2. Winning Architecture

A **3-layer hybrid policy** outperforms plain imitation learning:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HybridInsertPolicy                              │
│                                                                        │
│  Layer 1 ─ Perception (CV)                                            │
│    • YOLOv8-s  → detect port bounding box in center camera            │
│    • PnP solver → 3-D port pose from 3 calibrated wrist cameras       │
│    • Output: port_pose_in_base (4×4 transform)                        │
│                                                                        │
│  Layer 2 ─ Coarse motion (Imitation Learning)                         │
│    • ACT policy (dim=512, ResNet34 backbone) — primary                │
│    • Diffusion Policy — fallback / ensemble member                     │
│    • Conditioned on: 3 images + 26-D state + task.cable_type token    │
│    • Output: Cartesian twist velocity (6-D) @ 4 Hz                    │
│                                                                        │
│  Layer 3 ─ Fine insertion (Force-guided state machine)                │
│    • Admittance controller already on the robot                        │
│    • Spiral search if XY error > 1 mm                                 │
│    • Force threshold for seat detection (> 5 N sustained)              │
│    • Abort + re-align if force > 18 N                                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 0 — Environment Setup (2–4 hours)

### 3.1 GPU drivers and CUDA

```bash
# NVIDIA driver ≥ 550 (RTX A5000 is Ampere / GA102)
sudo apt install nvidia-driver-550 -y
sudo reboot
nvidia-smi          # verify: "NVIDIA RTX A5000" + VRAM 16376 MiB

# CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install cuda-toolkit-12-4 -y
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version      # verify: "release 12.4"
```

### 3.2 Docker + NVIDIA Container Toolkit

```bash
# Docker Engine
sudo apt install docker.io -y
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ctk-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ctk-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-ctk.list
sudo apt update && sudo apt install nvidia-container-toolkit -y
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu24.04 nvidia-smi  # verify
```

### 3.3 Distrobox + Pixi

```bash
sudo apt install distrobox -y
curl -fsSL https://pixi.sh/install.sh | sh && source ~/.bashrc
```

### 3.4 Clone repo + install deps

```bash
mkdir -p ~/ws_aic/src && cd ~/ws_aic/src
git clone https://github.com/intrinsic-dev/aic
cd ~/ws_aic/src/aic
pixi install        # downloads ROS 2 Kilted + LeRobot 0.4.3 + MuJoCo 3.5.0 + OpenCV

# Verify GPU visible inside pixi
pixi run python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA RTX A5000
```

### 3.5 Pull evaluation container

```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

### 3.6 Quick-start smoke test

```bash
# Terminal A — start simulator
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal B — run dummy WaveArm policy
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
# Expect: Gazebo opens, 3 trials run, scoring.yaml written to ~/aic_results/
```

---

## 4. Phase 1 — Baseline Validation (Day 1–2)

### 4.1 CheatCode (ground truth ceiling)

```bash
# Terminal A
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true

# Terminal B
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```

Record the score from `~/aic_results/scoring.yaml`.
This is your theoretical ceiling (~85–100 pts/trial). Any ML policy must approach this.

### 4.2 ACT baseline (no ground truth)

```bash
# Terminal A — no ground truth!
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal B
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunACT
# Downloads grkw/aic_act_policy from HuggingFace automatically
```

Note your starting score. Typical baseline: ~20–40 pts/trial.
Your goal: **> 75 pts/trial** with the hybrid approach.

### 4.3 Study CheatCode internals

Key geometry observed (`aic_example_policies/ros/CheatCode.py`):

- Descends Z from +0.20 m above port to −0.015 m (insertion depth)
- Step size: 0.5 mm per control loop
- XY PID correction: `i_gain = 0.15`, `max_windup = 0.05 m`
- Orientation: slerp toward target quaternion
- Insertion confirmed after 5 s stabilisation at depth

These numbers are your ground truth for the fine-insertion state machine in Phase 5.

---

## 5. Phase 2 — Computer Vision Pipeline (Week 2–3)

### 5.1 Camera geometry recap

Three wrist-mounted RGB cameras:

| Camera | ROS topic | Resolution | Rate |
|--------|-----------|------------|------|
| Left | `/left_camera/image` | 1152 × 1024 | 20 Hz |
| Center | `/center_camera/image` | 1152 × 1024 | 20 Hz |
| Right | `/right_camera/image` | 1152 × 1024 | 20 Hz |

All three cameras publish `sensor_msgs/CameraInfo` (intrinsics + distortion).
Left–Center and Center–Right form two stereo pairs.

### 5.2 Synthetic dataset generation

Use the simulation with `ground_truth:=true` to auto-label images:

```python
# record_labeled_images.py  (run inside pixi env)
# Subscribes to /center_camera/image and /tf
# Projects port 3D origin → 2D pixel via camera intrinsics
# Saves: image.jpg + YOLO label  (class_id cx cy w h, normalized)

import rclpy, cv2, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge

class LabelWriter(Node):
    def __init__(self):
        super().__init__('label_writer')
        self.bridge = CvBridge()
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.create_subscription(Image, '/center_camera/image', self.image_cb, 10)
        self.create_subscription(CameraInfo, '/center_camera/camera_info', self.info_cb, 10)
        self.K = None

    def info_cb(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)

    def image_cb(self, msg):
        if self.K is None:
            return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        for port_frame in ['sfp_port_0', 'sfp_port_1', 'sc_port_0', 'sc_port_1']:
            try:
                t = self.tf_buf.lookup_transform('center_camera_optical', port_frame,
                                                  rclpy.time.Time())
                pt = np.array([t.transform.translation.x,
                               t.transform.translation.y,
                               t.transform.translation.z, 1.0])
                # Project: u = fx*(X/Z)+cx, v = fy*(Y/Z)+cy
                u = self.K[0,0] * pt[0]/pt[2] + self.K[0,2]
                v = self.K[1,1] * pt[1]/pt[2] + self.K[1,2]
                # Write YOLO label ...
            except Exception:
                pass
```

Collect **5 000–10 000 labelled frames** across randomised board configurations.

### 5.3 YOLOv8-s training (fits comfortably in 16 GB)

```bash
pip install ultralytics

# Dataset structure
# aic_ports/
#   images/train/   images/val/
#   labels/train/   labels/val/
#   data.yaml

cat > aic_ports/data.yaml << 'EOF'
path: aic_ports
train: images/train
val: images/val
nc: 4
names: [sfp_port, sc_port, nic_card, sfp_module]
EOF

yolo train \
  model=yolov8s.pt \
  data=aic_ports/data.yaml \
  epochs=200 \
  imgsz=640 \
  batch=32 \
  lr0=1e-3 \
  augment=True \
  project=runs/detect \
  name=aic_ports_v1
# Training time: ~3–4 hrs on RTX A5000 16 GB
# Expected mAP50: > 0.90 for port detection
```

### 5.4 3-D port pose via PnP

Obtain the 3-D corner coordinates of each port face from `aic_assets/models/`:

```python
import cv2, numpy as np

# 3-D keypoints for SFP port face (in port model frame, metres)
SFP_PORT_3D = np.array([
    [-0.0055, -0.0045, 0.0],   # top-left
    [ 0.0055, -0.0045, 0.0],   # top-right
    [ 0.0055,  0.0045, 0.0],   # bottom-right
    [-0.0055,  0.0045, 0.0],   # bottom-left
], dtype=np.float64)

def estimate_port_pose(image_points_2d, camera_K, dist_coeffs=None):
    """
    image_points_2d: (4,2) pixel coords of port corners from YOLO + corner refinement
    Returns rvec, tvec in camera frame
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    success, rvec, tvec = cv2.solvePnP(
        SFP_PORT_3D, image_points_2d.astype(np.float64),
        camera_K, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE)
    return rvec, tvec   # transform to robot base via TF
```

### 5.5 Multi-view depth fusion (bonus accuracy)

With both stereo pairs you can triangulate the port centre to sub-mm accuracy:

```python
# Using OpenCV stereo rectification + triangulatePoints
# Left-Center: baseline ≈ known from URDF
left_pts, center_pts = matched_keypoints_from_superpoint()
pts_4d = cv2.triangulatePoints(P_left, P_center, left_pts, center_pts)
pts_3d = (pts_4d[:3] / pts_4d[3]).T   # Euclidean coords in camera frame
```

### 5.6 Visual servoing to approach

```python
def visual_servo_to_port(port_pose_base, move_robot, get_observation):
    """Move gripper TCP toward port using proportional control."""
    Kp_xyz = 0.6   # proportional gain [1/s]
    Kp_rot = 0.4
    max_v  = 0.08  # m/s

    while True:
        obs = get_observation()
        tcp = obs.controller_state.tcp_pose
        err_xyz = port_pose_base[:3] - np.array([tcp.position.x,
                                                   tcp.position.y,
                                                   tcp.position.z])
        v_xyz = np.clip(Kp_xyz * err_xyz, -max_v, max_v)
        if np.linalg.norm(err_xyz) < 0.015:   # within 15 mm → hand off to IL/FSM
            break
        # send velocity command
        send_twist(move_robot, v_xyz, angular=[0,0,0])
```

---

## 6. Phase 3 — Imitation Learning (Week 2–4)

### 6.1 Teleoperation data collection

```bash
# Terminal A — eval env with ground truth (for better annotations)
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=false

# Terminal B — record with keyboard EE control
cd ~/ws_aic/src/aic
pixi run lerobot-record \
  --robot.type=aic_robot_aic \
  --teleop.type=aic_keyboard_ee \
  --dataset.repo_id=YOUR_HF_USERNAME/aic_demos \
  --dataset.num_episodes=300
```

**Keyboard mapping** (`aic_utils/aic_teleoperation`):

| Key | Action | Speed |
|-----|--------|-------|
| W/S | +/− Y (lateral) | Shift=fast 0.1 m/s, normal 0.02 m/s |
| A/D | +/− X (forward) | same |
| R/F | +/− Z (vertical) | same |
| I/K, J/L, U/O | Rotation (roll/pitch/yaw) | 0.1/0.02 rad/s |

**SpaceMouse** (recommended for quality): `--teleop.type=aic_spacemouse`

**Dataset split target:**

| Connector | # episodes | Notes |
|-----------|-----------|-------|
| SFP → SFP_PORT | 150 | Vary NIC rail 0–4, translation, yaw ±10° |
| SC → SC_PORT | 100 | Vary rail translation |
| Recovery demos | 50 | Start misaligned, recover and insert |
| **Total** | **300** | Upload to HuggingFace |

### 6.2 Train ACT (primary policy)

16 GB VRAM allows a full-quality ACT model with ResNet34 backbone:

```bash
pixi run lerobot-train \
  --policy.type=act \
  --dataset.repo_id=YOUR_HF_USERNAME/aic_demos \
  \
  --policy.backbone=resnet34 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=7 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.use_vae=true \
  \
  --training.batch_size=24 \
  --training.lr=1e-5 \
  --training.lr_scheduler=cosine \
  --training.num_epochs=500 \
  --training.use_amp=true \
  --training.grad_accumulation_steps=2 \
  \
  --dataset.video_backend=torchvision \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.color_jitter.brightness=0.3 \
  --dataset.image_transforms.color_jitter.contrast=0.3 \
  --dataset.image_transforms.random_crop.enable=true \
  \
  --output_dir=outputs/act_aic_v1 \
  --wandb.enable=true
```

**Expected training time on RTX A5000 (16 GB):** ≈ 6–10 hours for 500 epochs

**Memory usage:** ≈ 11–13 GB VRAM with batch=24, AMP enabled

### 6.3 Train Diffusion Policy (secondary / ensemble)

```bash
pixi run lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=YOUR_HF_USERNAME/aic_demos \
  \
  --policy.horizon=16 \
  --policy.n_obs_steps=2 \
  --policy.n_action_steps=8 \
  --policy.n_diffusion_steps=100 \
  --policy.down_dims=[256,512,1024] \
  \
  --training.batch_size=32 \
  --training.lr=1e-4 \
  --training.num_epochs=800 \
  --training.use_amp=true \
  \
  --output_dir=outputs/diffusion_aic_v1
```

**Expected time:** ≈ 8–12 hours

### 6.4 Evaluate and compare

```bash
# Run eval on trained policy
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy.ActPolicy \
  -p policy_checkpoint:=/home/user/ws_aic/outputs/act_aic_v1/checkpoints/best.pt
```

Look at `scoring.yaml` and compare against baseline. Target at this stage: **> 45 pts/trial**.

---

## 7. Phase 4 — Reinforcement Learning with IsaacLab (Week 4–6)

### 7.1 Setup IsaacLab

IsaacLab requires NVIDIA Isaac Sim (≈ 20 GB download). For 16 GB VRAM use 32–64 parallel environments.

```bash
# Install Isaac Sim via pip (omniverse-launcher alternative)
pip install 'isaacsim[all]' --extra-index-url https://pypi.nvidia.com
pip install isaaclab

# Verify
python -c "import isaaclab; print(isaaclab.__version__)"
```

### 7.2 Run IsaacLab PPO training (already configured in repo)

```bash
cd ~/ws_aic/src/aic/aic_utils/aic_isaac/aic_isaaclab

python scripts/rsl_rl/train.py \
  --task=AicTask \
  --num_envs=64 \
  --max_iterations=1500 \
  --save_interval=50 \
  --headless \
  --video --video_interval=200
# 64 envs uses ≈ 12–14 GB VRAM; reduce to 32 if needed
```

**Pre-configured PPO hyperparameters** (`agents/rsl_rl_ppo_cfg.py`):

```
Actor/Critic: [512, 256, 128], ELU activation
LR: 1e-3 adaptive, clip=0.2, entropy=0.006
Gamma=0.99, Lambda=0.95, target_KL=0.01
Steps per env: 24, Mini-batches: 4, Epochs: 8
```

### 7.3 Warm-start RL from IL policy (recommended)

Instead of training RL from scratch, initialise the actor network from the ACT-derived state encoder:

```python
# In train.py, after creating agent:
il_checkpoint = torch.load("outputs/act_aic_v1/checkpoints/best.pt")
# Extract state encoder weights → init actor MLP layers
# Fine-tune RL from this warm start → converges 3-5× faster
```

### 7.4 Curriculum schedule

Edit `aic_task_env_cfg.py` events section:

```python
# Week 4: easy (small randomisation)
board_pose_noise_xy = 0.01     # ±1 cm
nic_translation_range = [0.0, 0.03]
nic_orientation_range = [-5.0, 5.0]  # degrees

# Week 5: medium
board_pose_noise_xy = 0.03
nic_translation_range = [0.0, 0.062]
nic_orientation_range = [-10.0, 10.0]  # full spec

# Week 6: hard (full randomisation + lighting noise)
dome_light_intensity = [1500, 3500]
dome_light_color_noise = 0.1
```

### 7.5 MuJoCo alternative (lighter, no Isaac Sim)

If IsaacLab is too heavy, use the MuJoCo pipeline already in the repo:

```bash
cd ~/ws_aic/src/aic/aic_utils/aic_mujoco
# Follow the README:
# 1. Export Gazebo world → /tmp/aic.sdf
# 2. Run: python add_cable_plugin.py (fixes URIs, adds actuators/sensors)
# 3. Launch MuJoCo + ros2_control instead of Gazebo
# Same ROS 2 interfaces — same policy code works unchanged
```

MuJoCo uses ≈ 2–4 GB VRAM for 32 envs. Runs 3–5× faster wall-clock than Gazebo.

---

## 8. Phase 5 — Force-Guided Insertion State Machine (Week 5–6)

This layer executes after the IL policy brings the plug to within ~5 mm of the port.

### 8.1 State machine

```python
import numpy as np
from enum import Enum

class State(Enum):
    APPROACH = 0    # CV visual servo to 15 mm above port
    ALIGN    = 1    # IL policy for fine alignment
    SEARCH   = 2    # spiral search if XY error > 1 mm
    DESCEND  = 3    # controlled Z descent into port
    VERIFY   = 4    # force signature check
    DONE     = 5
    ABORT    = 6    # re-align after failed attempt

class ForceGuidedInsertion:
    FORCE_ABORT    = 18.0   # N — stay below 20 N penalty threshold
    FORCE_CONTACT  =  5.0   # N — plug touching port
    DEPTH_TARGET   = -0.015 # m relative to port entrance
    STEP_Z_FAST    = 0.002  # m/s — descent speed
    STEP_Z_SLOW    = 0.0005 # m/s — final push speed
    SPIRAL_RADIUS  = 0.003  # m — search radius
    SPIRAL_FREQ    = 0.5    # Hz

    def run(self, get_observation, move_robot, send_feedback):
        state = State.APPROACH
        attempt = 0

        while state not in (State.DONE, State.ABORT):
            obs       = get_observation()
            ft        = obs.wrist_wrench.wrench
            fz        = ft.force.z
            state_str = state.name
            send_feedback(f"FSM: {state_str}  Fz={fz:.2f} N")

            if fz > self.FORCE_ABORT:
                state = State.ALIGN     # back off and re-align
                attempt += 1
                if attempt > 3:
                    state = State.ABORT
                continue

            if state == State.APPROACH:
                # CV visual servo (see Phase 2)
                if self._within_approach_zone(obs):
                    state = State.ALIGN

            elif state == State.ALIGN:
                # Run ACT / Diffusion policy for N steps
                if self._aligned(obs):
                    state = State.DESCEND
                elif self._alignment_timeout():
                    state = State.SEARCH

            elif state == State.SEARCH:
                # Lissajous / Archimedean spiral in XY plane
                t = self.time_now()
                dx = self.SPIRAL_RADIUS * np.cos(2 * np.pi * self.SPIRAL_FREQ * t)
                dy = self.SPIRAL_RADIUS * np.sin(2 * np.pi * self.SPIRAL_FREQ * t)
                self._send_xy_offset(move_robot, dx, dy)
                if fz > self.FORCE_CONTACT:
                    state = State.DESCEND   # found port lip!

            elif state == State.DESCEND:
                vz = self.STEP_Z_SLOW if fz > self.FORCE_CONTACT else self.STEP_Z_FAST
                self._send_vz(move_robot, -vz)
                if self._insertion_depth_reached(obs):
                    state = State.VERIFY

            elif state == State.VERIFY:
                # Dwell 0.5 s; mild lateral wiggle to confirm seating
                self.sleep_for(0.5)
                state = State.DONE

        return state == State.DONE
```

### 8.2 Impedance tuning per phase

```python
IMPEDANCE = {
    "APPROACH": dict(
        stiffness=[90,  90,  90,  50, 50, 50],
        damping=  [50,  50,  50,  20, 20, 20],
        wrench_fb=[0.5, 0.5, 0.5,  0,  0,  0]),
    "ALIGN": dict(
        stiffness=[60,  60,  60,  30, 30, 30],
        damping=  [40,  40,  40,  15, 15, 15],
        wrench_fb=[0.8, 0.8, 0.5,  0,  0,  0]),
    "DESCEND": dict(
        stiffness=[50,  50, 200,  30, 30, 30],  # high Z stiffness → pushes into port
        damping=  [40,  40,  60,  15, 15, 15],
        wrench_fb=[0.9, 0.9, 0.2,  0,  0,  0]), # high XY compliance → self-centres
}
```

---

## 9. Phase 6 — Hybrid Integration (Week 7)

### 9.1 Policy skeleton

```python
# my_aic_policy/hybrid_policy.py
from aic_model.policy import Policy
from aic_task_interfaces.msg import Task

class HybridInsertPolicy(Policy):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.yolo      = load_yolo("runs/detect/aic_ports_v1/weights/best.pt")
        self.act       = load_act("outputs/act_aic_v1/checkpoints/best.pt")
        self.diffusion = load_diffusion("outputs/diffusion_aic_v1/checkpoints/best.pt")
        self.fsm       = ForceGuidedInsertion()

    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback):
        cable_type = task.cable_type   # "sfp_sc" or "sc_sfp"
        plug_type  = task.plug_type    # "sfp" or "sc"
        port_class = "sfp_port" if plug_type == "sfp" else "sc_port"

        send_feedback(f"Starting insertion: {plug_type} → {port_class}")

        # --- Layer 1: CV coarse approach ---
        port_pose = self._detect_port(get_observation, port_class)
        if port_pose is not None:
            send_feedback("CV pose acquired, visual servoing...")
            visual_servo_to_port(port_pose, move_robot, get_observation)
        else:
            send_feedback("CV failed, using learned policy only")

        # --- Layer 2: IL fine alignment (ACT + optional Diffusion ensemble) ---
        for _ in range(200):   # up to 50 s @ 4 Hz
            obs = get_observation()
            act_action  = self.act.select_action(obs)
            diff_action = self.diffusion.select_action(obs)
            action = 0.6 * act_action + 0.4 * diff_action   # weighted ensemble
            move_robot(action)
            if self._near_port(obs):
                break

        # --- Layer 3: Force-guided insertion ---
        success = self.fsm.run(get_observation, move_robot, send_feedback)
        send_feedback(f"Insertion {'succeeded' if success else 'failed'}")
        return success
```

### 9.2 Trajectory smoothing for Tier 2

```python
from scipy.signal import savgol_filter

class ActionBuffer:
    """Rolling buffer for Savitzky-Golay smoothing of action output."""
    def __init__(self, window=7, polyorder=3):
        self.buf = []
        self.window = window
        self.polyorder = polyorder

    def push(self, action):
        self.buf.append(action)
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def smoothed(self):
        if len(self.buf) < self.window:
            return self.buf[-1]
        arr = np.array(self.buf)
        return savgol_filter(arr, self.window, self.polyorder, axis=0)[-1]
```

---

## 10. Phase 7 — Packaging & Submission (Week 8–10)

### 10.1 Create participant package

```
my_aic_policy/
├── setup.py
├── package.xml
└── my_aic_policy/
    ├── __init__.py
    ├── hybrid_policy.py       ← HybridInsertPolicy
    ├── cv_port_detector.py    ← YOLO + PnP
    ├── force_fsm.py           ← ForceGuidedInsertion
    └── checkpoints/
        ├── yolo_aic_ports.pt
        ├── act_aic_best.pt
        └── diffusion_aic_best.pt
```

### 10.2 Dockerfile

```dockerfile
FROM ghcr.io/intrinsic-dev/aic/aic_eval:latest

# Extra dependencies
RUN pip install ultralytics==8.3 scipy

# Copy policy package
COPY my_aic_policy /ws_aic/src/my_aic_policy/
RUN cd /ws_aic && colcon build --packages-select my_aic_policy
RUN echo "source /ws_aic/install/setup.bash" >> /etc/bash.bashrc
```

### 10.3 Full local test (3 trials, no ground truth)

```bash
# Terminal A
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal B
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_aic_policy.HybridInsertPolicy

# After all 3 trials:
cat ~/aic_results/scoring.yaml
```

Target before submission: **> 70 pts average across 3 trials**.

### 10.4 Submit

```bash
# Follow docs/submission.md exactly
docker build -t my_aic_submission .
docker tag my_aic_submission <REGISTRY>/<USERNAME>/my_aic_submission:v1
docker push <REGISTRY>/<USERNAME>/my_aic_submission:v1
```

---

## 11. VRAM Budget (RTX A5000 16 GB)

| Component | Inference VRAM | Training VRAM |
|-----------|---------------|---------------|
| YOLOv8-s (640 px) | 0.8 GB | 3 GB (batch 32) |
| ACT dim=512, ResNet34 | 1.8 GB | 11–13 GB (batch 24, AMP) |
| Diffusion Policy (UNet-1024) | 2.1 GB | 10–12 GB (batch 32) |
| FoundationPose (zero-shot 6-DoF) | 3.0 GB | N/A |
| IsaacLab PPO (64 envs) | 12–14 GB | — |
| IsaacLab PPO (32 envs) | 6–8 GB | — |

**Training workflow:** Train one model at a time. 16 GB is sufficient for every step above.
Do not run Gazebo + training simultaneously.

---

## 12. Timeline to May 15, 2026 Deadline

| Week | Milestone | Target score |
|------|-----------|-------------|
| 1 | Setup complete, WaveArm running, CheatCode scored | — |
| 2 | ACT baseline evaluated; data collection started | Baseline noted |
| 3 | 300 demos collected, YOLOv8-s trained | mAP50 > 0.90 |
| 4 | ACT v1 trained (500 epochs) | > 45 pts/trial |
| 5 | Force-FSM + visual servo integrated | > 55 pts/trial |
| 6 | IsaacLab RL fine-tuning (1 500 iterations) | > 65 pts/trial |
| 7 | Diffusion Policy trained; ensemble built | > 70 pts/trial |
| 8 | Hybrid policy integrated and tested | > 75 pts/trial |
| 9 | Trajectory smoothing + speed optimisation | > 80 pts/trial |
| 10 | Docker packaging, submission | Submitted ≤ May 15 |

---

## 13. Key Files in This Repository

| File | Role |
|------|------|
| `aic_model/aic_model/policy.py` | Abstract base class — extend this |
| `aic_example_policies/ros/RunACT.py` | ACT integration pattern (observe → action loop) |
| `aic_example_policies/ros/CheatCode.py` | Ground-truth insertion reference (PD control, geometry) |
| `aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_robot_aic_controller.py` | Obs/action space for data collection |
| `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py` | RL environment + rewards |
| `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |
| `aic_engine/config/sample_config.yaml` | Trial configuration (grasp offsets, board geometry) |
| `docs/qualification_phase.md` | Exact trial definitions + constraints |
| `docs/scoring.md` | Full scoring formula |
| `docs/submission.md` | Container packaging + upload |

---

## 14. Verification Checklist

- [ ] `nvidia-smi` shows RTX A5000 + 16376 MiB VRAM
- [ ] `nvcc --version` shows CUDA 12.4
- [ ] `docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu24.04 nvidia-smi` succeeds
- [ ] `pixi install` completes without errors
- [ ] WaveArm policy runs 3 trials → `scoring.yaml` written
- [ ] CheatCode policy achieves > 80 pts/trial (ground_truth:=true)
- [ ] ACT baseline (grkw/aic_act_policy) loads and scores without errors
- [ ] `lerobot-record` captures demos with 3 cameras + state
- [ ] `lerobot-train` (ACT, batch=24) runs without OOM
- [ ] YOLOv8-s training reaches mAP50 > 0.85 on validation set
- [ ] Hybrid policy completes 3 trials > 70 pts/trial average
- [ ] Docker container builds and passes local 3-trial test
