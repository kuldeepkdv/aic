# AIC Roadmap — Minimum Viable Policy First
## Ubuntu 24.04 + NVIDIA RTX A5000 (16 GB VRAM) | Deadline: May 15, 2026

---

## 1. Challenge Summary

The **AI for Industry Challenge (AIC)** requires a UR5e robot arm to insert fiber optic
connectors (SFP modules, SC plugs) into ports on a randomised modular task board, evaluated
entirely inside Gazebo simulation during the Qualification Phase.

### Critical facts

| Item | Detail |
|------|--------|
| Plug already in hand | Robot starts with plug grasped — **no picking required** |
| Starting distance | Arm is already within **a few centimetres** of the target port |
| Trials | 3 per submission: Trial 1 & 2 = SFP→SFP_PORT, Trial 3 = SC_PLUG→SC_PORT |
| Randomisation | Board pose (XY + yaw), NIC card rail + translation + yaw, SC port translation |
| Grasp deviation | ≈ ±2 mm, ±0.04 rad from nominal |
| Time limit | 180 s per trial |
| Evaluation GPU | NVIDIA L4 Tensor Core, 24 GB VRAM |

### Scoring (max ≈ 100 pts/trial)

| Tier | What | Points |
|------|------|--------|
| 1 | Node loads + lifecycle transitions valid | 0–1 |
| 2 | Smoothness (0–5) + speed (0–10) + efficiency (0–5) − force penalty (−12) − collision penalty (−24) | 0–30 |
| 3 | Full insertion correct port +60, wrong port −10, partial 0–40, proximity 0–25 | −10 to +60 |

**Key insight:** A classical force-guided state machine can achieve Tier 3 insertion without
any machine learning. Measure that score first. Only add CV/IL/RL if the gap justifies
the weeks of work required.

---

## 2. Decision-Gated Architecture

Do not build all layers upfront. Add each layer only when the score gap demands it.

```
┌─────────────────────────────────────────────────────────┐
│  Week 1: Force FSM only                                 │
│    Score >= 80 → SUBMIT. Stop here.                     │
│    Score 55-79 → add Phase 2 (Imitation Learning)       │
│    Score 30-54 → add Phase 2 + Phase 3 (CV servo)       │
│    Score < 30  → diagnose root cause before adding ML   │
└─────────────────────────────────────────────────────────┘
         │ (only if score < 80)
         ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2: ACT imitation learning                        │
│    Score >= 75 → skip Diffusion Policy, skip RL         │
│    Score 65-74 → add CV visual servo (Phase 3)          │
│    Score < 65  → add RL fine-tuning (Phase 4)           │
└─────────────────────────────────────────────────────────┘
         │ (only if score < 75)
         ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3: CV visual servo (coarse approach only)        │
│  Phase 4: RL fine-tuning via MuJoCo (last resort)       │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Phase 0 — Environment Setup (Day 1, ~4 hours)

### 3.1 GPU drivers and CUDA

```bash
sudo apt install nvidia-driver-550 -y && sudo reboot
nvidia-smi          # verify: RTX A5000, 16376 MiB

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install cuda-toolkit-12-4 -y
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version      # verify: release 12.4
```

### 3.2 Docker + NVIDIA Container Toolkit

```bash
sudo apt install docker.io -y
sudo usermod -aG docker $USER && newgrp docker

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

### 3.4 Clone repo and install

```bash
mkdir -p ~/ws_aic/src && cd ~/ws_aic/src
git clone https://github.com/intrinsic-dev/aic
cd ~/ws_aic/src/aic
pixi install

pixi run python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA RTX A5000
```

### 3.5 Pull evaluation container

```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

### 3.6 Smoke test

```bash
# Terminal A
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal B
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
# Expect: Gazebo opens, 3 trials run, ~/aic_results/scoring.yaml written
```

---

## 4. Phase 1 — Force FSM (Week 1) — PRIMARY DELIVERABLE

This is your first real policy. Build and score it before writing a single line of ML code.

### 4.1 Study CheatCode first (1-2 hours)

Read `aic_example_policies/ros/CheatCode.py` in full. Key geometry:

- Descends Z from +0.20 m above port to -0.015 m (insertion depth)
- Step size: 0.5 mm per control loop
- XY PID: `i_gain = 0.15`, `max_windup = 0.05 m`
- Orientation: slerp toward target quaternion
- Insertion confirmed after 5 s stabilisation at depth

Run CheatCode with `ground_truth:=true` to get your theoretical ceiling:

```bash
# Terminal A
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true

# Terminal B
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```

Record the score. This is the upper bound for any approach.

### 4.2 Measure force sensor noise floor before setting thresholds

Do not assume force thresholds. Measure them first:

```python
# noise_floor.py — run while robot is stationary, plug not touching anything
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import numpy as np

class NoiseMeasure(Node):
    def __init__(self):
        super().__init__('noise_measure')
        self.samples = []
        self.create_subscription(WrenchStamped, '/wrist_wrench', self.cb, 10)

    def cb(self, msg):
        self.samples.append(msg.wrench.force.z)
        if len(self.samples) == 200:
            arr = np.array(self.samples)
            self.get_logger().info(
                f"Fz noise: mean={arr.mean():.3f} std={arr.std():.3f} "
                f"max={arr.max():.3f} min={arr.min():.3f}")
            rclpy.shutdown()

rclpy.init()
rclpy.spin(NoiseMeasure())
```

Use the measured values to set:
- `FORCE_CONTACT` = noise_max + 1.5 N safety margin
- `FORCE_ABORT` = 17.0 N (3 N below the 20 N penalty threshold)

### 4.3 Implement the Force FSM

```python
# my_aic_policy/force_fsm.py
import numpy as np
from enum import Enum

class State(Enum):
    APPROACH = 0
    ALIGN    = 1
    SEARCH   = 2
    DESCEND  = 3
    VERIFY   = 4
    DONE     = 5
    ABORT    = 6

class ForceGuidedInsertion:
    # Set FORCE_CONTACT from noise_floor.py measurement — do not guess
    FORCE_ABORT   = 17.0   # N — 3 N margin below 20 N penalty
    FORCE_CONTACT =  4.0   # N — placeholder; replace with measured value
    DEPTH_TARGET  = -0.015 # m relative to port entrance (from CheatCode)
    STEP_Z_FAST   = 0.002  # m/s
    STEP_Z_SLOW   = 0.0005 # m/s
    SPIRAL_RADIUS = 0.005  # m — wider than port tolerance; tighten after testing
    SPIRAL_FREQ   = 0.5    # Hz

    def __init__(self):
        self._fz_window = []  # 3-sample moving average to suppress spikes

    def _fz_filtered(self, fz_raw):
        self._fz_window.append(fz_raw)
        if len(self._fz_window) > 3:
            self._fz_window.pop(0)
        return np.mean(self._fz_window)

    def run(self, get_observation, move_robot, send_feedback):
        state = State.APPROACH
        attempt = 0
        t_search_start = None

        while state not in (State.DONE, State.ABORT):
            obs = get_observation()
            fz_raw = obs.wrist_wrench.wrench.force.z
            fz = self._fz_filtered(fz_raw)
            send_feedback(f"FSM:{state.name} Fz={fz:.2f}N (raw={fz_raw:.2f}N)")

            if fz > self.FORCE_ABORT:
                # Back off 5 mm in Z before re-aligning
                self._send_vz(move_robot, +0.005)
                attempt += 1
                state = State.ALIGN if attempt <= 3 else State.ABORT
                self._fz_window.clear()
                continue

            if state == State.APPROACH:
                if self._within_approach_zone(obs):
                    state = State.ALIGN

            elif state == State.ALIGN:
                if self._aligned(obs):
                    state = State.DESCEND
                elif self._alignment_timeout(obs):
                    t_search_start = self._time_now()
                    state = State.SEARCH

            elif state == State.SEARCH:
                t = self._time_now() - t_search_start
                # Log spiral: radius grows over time to guarantee port coverage
                r = self.SPIRAL_RADIUS * (1 + 0.1 * t)
                dx = r * np.cos(2 * np.pi * self.SPIRAL_FREQ * t)
                dy = r * np.sin(2 * np.pi * self.SPIRAL_FREQ * t)
                self._send_xy_offset(move_robot, dx, dy)
                if fz > self.FORCE_CONTACT:
                    state = State.DESCEND  # found port lip

            elif state == State.DESCEND:
                vz = self.STEP_Z_SLOW if fz > self.FORCE_CONTACT else self.STEP_Z_FAST
                self._send_vz(move_robot, -vz)
                if self._insertion_depth_reached(obs):
                    state = State.VERIFY

            elif state == State.VERIFY:
                self._dwell(0.5)
                state = State.DONE

        return state == State.DONE
```

### 4.4 Score the FSM-only policy

```bash
# Terminal A
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal B
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_aic_policy.FsmOnlyPolicy

cat ~/aic_results/scoring.yaml
```

**Decision point:** If average score >= 80, package and submit. Do not add ML.

---

## 5. Score-Gap Gate

Evaluate your FSM-only score and decide the next step:

| FSM-only avg score | Action |
|--------------------|--------|
| >= 80 | Package FSM as-is and submit |
| 65-79 | Phase 2: collect 150 demos, train ACT only |
| 45-64 | Phase 2 ACT + Phase 3 CV visual servo |
| < 45 | Diagnose failure mode from Section 9 before adding any ML |

Do not skip the diagnosis step. Adding ML on top of a broken FSM makes debugging
exponentially harder.

---

## 6. Phase 2 — Imitation Learning (only if score < 80)

### 6.1 Data collection

Use `ground_truth:=false` to match the evaluation domain. Using `ground_truth:=true`
introduces a domain gap that degrades eval performance.

```bash
# Terminal A
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=false

# Terminal B
cd ~/ws_aic/src/aic
pixi run lerobot-record \
  --robot.type=aic_robot_aic \
  --teleop.type=aic_keyboard_ee \
  --dataset.repo_id=YOUR_HF_USERNAME/aic_demos \
  --dataset.num_episodes=150
```

**Keyboard mapping:**

| Key | Action |
|-----|--------|
| W/S | +/- Y lateral |
| A/D | +/- X forward |
| R/F | +/- Z vertical |
| I/K, J/L, U/O | Rotation (roll/pitch/yaw) |
| Shift | 5x speed |

SpaceMouse strongly recommended: `--teleop.type=aic_spacemouse`

**Staged collection — evaluate ACT at each stage before collecting more:**

| Stage | Episodes | Purpose |
|-------|----------|---------|
| 1 | 50 | Sanity check: does ACT learn anything? |
| 2 | +100 (total 150) | Minimum for ±10 deg yaw generalisation |
| 3 | +150 (total 300) | Add recovery demos (start misaligned, recover, insert) |

If ACT shows no learning after stage 1, fix data quality before collecting stage 2.

### 6.2 Train ACT

Start conservative on batch size — verify VRAM headroom before increasing:

```bash
pixi run lerobot-train \
  --policy.type=act \
  --dataset.repo_id=YOUR_HF_USERNAME/aic_demos \
  --policy.backbone=resnet34 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=7 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.use_vae=true \
  --training.batch_size=16 \
  --training.lr=1e-5 \
  --training.lr_scheduler=cosine \
  --training.num_epochs=300 \
  --training.use_amp=true \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.color_jitter.brightness=0.3 \
  --dataset.image_transforms.color_jitter.contrast=0.3 \
  --dataset.image_transforms.random_crop.enable=true \
  --output_dir=outputs/act_aic_v1 \
  --wandb.enable=true
```

Monitor VRAM during first epoch. If < 13 GB used, increase to `--training.batch_size=24`.
Expected time: ~4-6 hours at batch=16 for 300 epochs.

### 6.3 Evaluate ACT alone before adding anything else

```bash
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_aic_policy.ActFsmPolicy \
  -p policy_checkpoint:=/home/user/ws_aic/outputs/act_aic_v1/checkpoints/best.pt

cat ~/aic_results/scoring.yaml
```

Target at this stage: **> 60 pts/trial**. If score >= 75, skip Diffusion Policy and RL.

### 6.4 Diffusion Policy (only if ACT score < 65)

Do not add Diffusion Policy by default. Naive action-space averaging of two IL policies
often produces worse results than either alone due to mode averaging. Only add if ACT
has clearly identified failure modes that Diffusion handles better.

If justified:

```bash
pixi run lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=YOUR_HF_USERNAME/aic_demos \
  --policy.horizon=16 \
  --policy.n_obs_steps=2 \
  --policy.n_action_steps=8 \
  --policy.n_diffusion_steps=100 \
  --policy.down_dims=[256,512,512] \
  --training.batch_size=16 \
  --training.lr=1e-4 \
  --training.num_epochs=500 \
  --training.use_amp=true \
  --output_dir=outputs/diffusion_aic_v1
```

Note: `down_dims=[256,512,1024]` at batch=32 may OOM on 16 GB. Start with
`[256,512,512]` and batch=16.

If combining ACT and Diffusion, use policy selection rather than action averaging:

```python
# Pick the policy whose validation loss is lower for the current connector type
policy = act if sfp_trial else diffusion
```

---

## 7. Phase 3 — CV Visual Servo (only if score < 70 after IL)

Scope is intentionally narrow: CV for coarse approach only (15 cm to 5 cm of port).
Do not build stereo fusion unless PnP alone proves insufficient.

### 7.1 Auto-label with ground truth TF

```python
# record_labeled_images.py
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
                u = self.K[0,0] * t.transform.translation.x / t.transform.translation.z \
                    + self.K[0,2]
                v = self.K[1,1] * t.transform.translation.y / t.transform.translation.z \
                    + self.K[1,2]
                # write YOLO label: class_id cx cy w h (normalised)
            except Exception:
                pass
```

Collect 3,000-5,000 frames. This is a sim-only eval — no sim-to-real gap to worry about.

### 7.2 YOLOv8-s training

```bash
pip install ultralytics

yolo train \
  model=yolov8s.pt \
  data=aic_ports/data.yaml \
  epochs=150 \
  imgsz=640 \
  batch=32 \
  lr0=1e-3 \
  project=runs/detect \
  name=aic_ports_v1
# ~2-3 hrs on RTX A5000; target mAP50 > 0.88
```

### 7.3 PnP pose estimation

```python
import cv2, numpy as np

SFP_PORT_3D = np.array([
    [-0.0055, -0.0045, 0.0],
    [ 0.0055, -0.0045, 0.0],
    [ 0.0055,  0.0045, 0.0],
    [-0.0055,  0.0045, 0.0],
], dtype=np.float64)

def estimate_port_pose(image_points_2d, camera_K):
    dist_coeffs = np.zeros(5)
    success, rvec, tvec = cv2.solvePnP(
        SFP_PORT_3D, image_points_2d.astype(np.float64),
        camera_K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    return rvec, tvec
```

Only add stereo triangulation if single-camera PnP accuracy proves insufficient.

### 7.4 Visual servo

```python
def visual_servo_to_port(port_pose_base, move_robot, get_observation):
    Kp_xyz = 0.6
    max_v  = 0.08  # m/s

    while True:
        obs = get_observation()
        tcp = obs.controller_state.tcp_pose
        err_xyz = port_pose_base[:3] - np.array([tcp.position.x,
                                                   tcp.position.y,
                                                   tcp.position.z])
        v_xyz = np.clip(Kp_xyz * err_xyz, -max_v, max_v)
        if np.linalg.norm(err_xyz) < 0.015:
            break
        send_twist(move_robot, v_xyz, angular=[0, 0, 0])
```

---

## 8. Phase 4 — RL Fine-Tuning (only if score < 65 after IL + CV)

Use MuJoCo first — already in the repo, runs 3-5x faster than Gazebo, uses 2-4 GB
VRAM for 32 environments. Use IsaacLab only if MuJoCo proves insufficient.

### 8.1 MuJoCo pipeline (preferred)

```bash
cd ~/ws_aic/src/aic/aic_utils/aic_mujoco
# 1. Export Gazebo world to /tmp/aic.sdf
# 2. python add_cable_plugin.py
# 3. Launch MuJoCo + ros2_control
# Same ROS 2 interfaces — same policy code works unchanged
```

### 8.2 IsaacLab (escalation path only)

```bash
pip install 'isaacsim[all]' --extra-index-url https://pypi.nvidia.com
pip install isaaclab

cd ~/ws_aic/src/aic/aic_utils/aic_isaac/aic_isaaclab

python scripts/rsl_rl/train.py \
  --task=AicTask \
  --num_envs=32 \
  --max_iterations=1500 \
  --save_interval=50 \
  --headless
# 32 envs: ~6-8 GB VRAM. Use 64 envs only if 32 converges too slowly.
```

**On warm-starting from ACT:** ACT uses a Transformer encoder; IsaacLab PPO uses an
MLP. These are not architecturally compatible for direct weight transfer. To warm-start,
you must encode training observations through ACT's state encoder, collect the resulting
latent vectors, and train a separate MLP regressor on them as a proxy init for the PPO
actor. This is several days of engineering — only attempt if RL is genuinely needed.

### 8.3 Curriculum schedule

```python
# aic_task_env_cfg.py — validate each stage before progressing
EASY   = dict(board_xy_noise=0.01, nic_range=[0.0, 0.03], nic_yaw=[-5, 5])
MEDIUM = dict(board_xy_noise=0.03, nic_range=[0.0, 0.062], nic_yaw=[-10, 10])
HARD   = dict(board_xy_noise=0.03, nic_range=[0.0, 0.062], nic_yaw=[-10, 10],
              dome_intensity=[1500, 3500])
```

---

## 9. Failure Modes and Solutions

Check this section before adding complexity. A broken FSM plus ML is harder to debug
than a broken FSM alone.

### 9.1 Force-related failures

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| Score loses -12 repeatedly (force penalty) | FORCE_ABORT too high; sensor spike crosses 20 N | Lower FORCE_ABORT to 15 N; apply 3-sample moving average on Fz |
| Robot never detects contact | FORCE_CONTACT set above actual contact force | Re-run noise_floor.py; set threshold = noise_max + 1.5 N |
| Robot stops halfway into port | DEPTH_TARGET too shallow | Read actual insertion depth from CheatCode.py; add 2 mm to DEPTH_TARGET |
| Force spikes on first touch then recovers | Z stiffness too high in DESCEND phase | Lower Z stiffness from 200 to 120; increase XY damping |
| Robot aborts after every attempt | Back-off distance not enough; re-contacts immediately | Increase back-off from 5 mm to 15 mm; clear fz_window before re-entering ALIGN |

### 9.2 Search and alignment failures

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| Spiral search never finds port | Radius too small for board randomisation range | Widen SPIRAL_RADIUS to 6 mm; use log spiral (radius grows with time) |
| Search finds port but robot misses on descent | Spiral frequency too high — overshoots port | Halve SPIRAL_FREQ; dwell 0.2 s when Fz > FORCE_CONTACT before descending |
| Alignment timeout triggers immediately | _aligned() threshold too tight | Loosen XY tolerance from 1 mm to 3 mm; compare to actual port opening width in URDF |
| Robot oscillates during ALIGN | Kp gain too high | Reduce Kp by 40%; add velocity damping term |

### 9.3 Imitation learning failures

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| Training loss NaN after epoch 10 | LR too high or non-finite values in dataset | Drop LR to 5e-6; run `lerobot-check-dataset` to find bad episodes |
| High training score, low eval score | Insufficient demo diversity | Add 50 recovery demos (start deliberately misaligned); verify demos used `ground_truth:=false` |
| ACT produces jerky motion (Tier 2 penalty) | Chunk size too large; action discontinuities | Reduce chunk_size to 50; apply Savitzky-Golay smoothing (window=7, polyorder=3) |
| ACT ignores visual input | Camera topics not in action_features | Verify all 3 camera topics are listed in robot controller action_features |
| Diffusion ensemble worse than ACT alone | Action-space averaging causes mode collapse | Drop ensemble; use policy selection by trial type instead of weighted average |

### 9.4 CV and detection failures

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| YOLO never detects port at close range | Port occluded by plug body | Expand bounding box by 20%; fall back to FSM-only if no detection within 2 s |
| PnP pose estimate is unstable or flipping | Too few corners detected | Use SOLVEPNP_IPPE_SQUARE if port is square; filter outliers with RANSAC |
| Visual servo overshoots at close range | Kp_xyz too high for final approach | Reduce Kp to 0.3; saturate velocity at 0.04 m/s for the last 3 cm |
| mAP50 plateaus below 0.80 | Insufficient lighting variation in data | Add brightness/contrast augmentation; collect 1000 extra frames with dome_intensity varied |

### 9.5 Submission and infrastructure failures

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| Tier 1 score = 0 (node lifecycle invalid) | Missing configure/activate lifecycle transitions | Add on_configure() and on_activate() to policy node; follow docs/submission.md lifecycle checklist |
| scoring.yaml not written after trials | Policy node crashed silently | Check /tmp/aic_model.log; ensure send_feedback() is called at every step without raising exceptions |
| Docker build fails at colcon build | Missing ROS dependency in container | Add `RUN rosdep install --from-paths src --ignore-src -r -y` before colcon build in Dockerfile |
| Docker push rejected | Wrong image tag format | Follow exact naming in docs/submission.md; verify registry credentials with docker login first |
| Policy runs locally but OOM in eval | Eval container has memory limits different from local | Check container resource limits in submission spec; reduce model batch size at inference |
| lerobot-record hangs on first episode | ROS 2 topics not yet published | Add 3 s startup delay; verify all camera topics active with `ros2 topic hz /center_camera/image` |

---

## 10. Hybrid Integration

Only assemble the hybrid policy once each layer has been validated independently.

```python
# my_aic_policy/hybrid_policy.py
from aic_model.policy import Policy
from aic_task_interfaces.msg import Task

class HybridInsertPolicy(Policy):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.yolo = load_yolo("runs/detect/aic_ports_v1/weights/best.pt")
        self.act  = load_act("outputs/act_aic_v1/checkpoints/best.pt")
        self.fsm  = ForceGuidedInsertion()

    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback):
        plug_type  = task.plug_type
        port_class = "sfp_port" if plug_type == "sfp" else "sc_port"

        # Layer 1: CV coarse approach (skip if YOLO returns no detection in 2 s)
        port_pose = self._detect_port(get_observation, port_class, timeout=2.0)
        if port_pose is not None:
            send_feedback("CV pose acquired")
            visual_servo_to_port(port_pose, move_robot, get_observation)
        else:
            send_feedback("CV timeout — FSM only")

        # Layer 2: ACT fine alignment (max 50 s at 4 Hz)
        for _ in range(200):
            obs = get_observation()
            action = self.act.select_action(obs)
            move_robot(action)
            if self._near_port(obs):
                break

        # Layer 3: Force FSM insertion
        success = self.fsm.run(get_observation, move_robot, send_feedback)
        send_feedback(f"{'Success' if success else 'Failed'}")
        return success
```

### Trajectory smoothing for Tier 2 (add only if jerk penalty visible in scoring)

```python
from scipy.signal import savgol_filter
import numpy as np

class ActionSmoother:
    def __init__(self, window=7, polyorder=3):
        self.buf = []
        self.window = window
        self.polyorder = polyorder

    def __call__(self, action):
        self.buf.append(action)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        if len(self.buf) < self.window:
            return action
        return savgol_filter(np.array(self.buf), self.window, self.polyorder, axis=0)[-1]
```

---

## 11. Packaging and Submission

### 11.1 Package structure

```
my_aic_policy/
├── package.xml
├── setup.py
└── my_aic_policy/
    ├── __init__.py
    ├── hybrid_policy.py
    ├── force_fsm.py
    ├── cv_port_detector.py     (only if Phase 3 was used)
    └── checkpoints/
        ├── yolo_aic_ports.pt   (only if Phase 3 was used)
        └── act_aic_best.pt     (only if Phase 2 was used)
```

Minimum viable submission contains only `force_fsm.py`. Do not add other files
unless they are verified to improve the score.

### 11.2 Lifecycle node compliance (Tier 1)

The policy node must implement the managed lifecycle or Tier 1 score = 0:

```python
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

class MyPolicyNode(LifecycleNode):
    def on_configure(self, state):
        # load model weights here
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        # start timers and subscriptions
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS
```

### 11.3 Dockerfile

```dockerfile
FROM ghcr.io/intrinsic-dev/aic/aic_eval:latest

RUN pip install ultralytics==8.3 scipy

COPY my_aic_policy /ws_aic/src/my_aic_policy/
RUN cd /ws_aic \
    && rosdep install --from-paths src --ignore-src -r -y \
    && colcon build --packages-select my_aic_policy
RUN echo "source /ws_aic/install/setup.bash" >> /etc/bash.bashrc
```

### 11.4 Full local test before submission

```bash
# Terminal A
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal B
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_aic_policy.HybridInsertPolicy

cat ~/aic_results/scoring.yaml
# Must average > 70 pts across all 3 trials before submitting
```

### 11.5 Submit

Read `docs/submission.md` for the exact registry URL, image naming convention, and
credential setup before running any of the commands below:

```bash
docker build -t my_aic_submission .
# Use the exact tag format from docs/submission.md — do not guess
docker tag my_aic_submission <REGISTRY_FROM_DOCS>/<USERNAME>/my_aic_submission:v1
docker login <REGISTRY_FROM_DOCS>
docker push <REGISTRY_FROM_DOCS>/<USERNAME>/my_aic_submission:v1
```

---

## 12. VRAM Budget (RTX A5000 16 GB)

| Component | Inference VRAM | Training VRAM | Notes |
|-----------|---------------|---------------|-------|
| YOLOv8-s (640 px) | 0.8 GB | 2-3 GB (batch 32) | Fits comfortably |
| ACT dim=512, ResNet34 | 1.8 GB | 10-12 GB (batch 16, AMP) | Start batch=16; increase to 24 if < 13 GB used |
| Diffusion Policy down=[256,512,512] | 1.5 GB | 8-10 GB (batch 16, AMP) | down=[256,512,1024] at batch=32 may OOM |
| MuJoCo PPO (32 envs) | 2-4 GB | — | Best starting point for RL |
| IsaacLab PPO (32 envs) | 6-8 GB | — | Prefer over 64 envs on 16 GB |
| IsaacLab PPO (64 envs) | 12-14 GB | — | Close to limit; never run alongside Gazebo |

**Rule:** Train one model at a time. Never run Gazebo and GPU training simultaneously.

---

## 13. Timeline (MVP-Gated)

| Week | Milestone | Gate |
|------|-----------|------|
| 1 | Setup + Force FSM implemented + 3 trials scored | **Score >= 80 → submit now** |
| 2 | Score gap measured; go/no-go on ML; start data collection if needed | Score < 80 → continue |
| 3 | 50 demos; ACT sanity check | Poor learning → fix data quality before collecting more |
| 4 | 150 demos; ACT v1 trained and scored | Score >= 75 → skip Diffusion + RL |
| 5 | CV visual servo + FSM integrated | Score >= 70 → skip RL |
| 6 | RL fine-tuning via MuJoCo (if needed) | Score >= 65 → stop RL early |
| 7 | Hybrid policy integrated; smoothing added if needed | Target > 75 pts/trial |
| 8 | Docker packaging + full local 3-trial test | > 70 pts/trial before submit |
| 9-10 | Buffer — re-runs and edge-case fixes | Submitted <= May 15 |

Weeks 9-10 are buffer, not planned work. If you have a passing policy by Week 8,
submit and stop. Do not add features in the final two weeks.

---

## 14. Key Files

| File | Role |
|------|------|
| `aic_model/aic_model/policy.py` | Abstract base — extend this for your policy |
| `aic_example_policies/ros/RunACT.py` | ACT integration pattern (observe → action loop) |
| `aic_example_policies/ros/CheatCode.py` | Ground-truth insertion reference — read before writing FSM |
| `aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_robot_aic_controller.py` | Obs/action space for data collection |
| `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py` | RL environment + rewards |
| `aic_utils/aic_mujoco/` | Lighter RL alternative to IsaacLab |
| `aic_engine/config/sample_config.yaml` | Trial configuration (grasp offsets, board geometry) |
| `docs/qualification_phase.md` | Exact trial definitions and constraints |
| `docs/scoring.md` | Full scoring formula — read before tuning anything |
| `docs/submission.md` | Container packaging + upload — read before building Docker image |

---

## 15. Verification Checklist

**Setup**
- [ ] `nvidia-smi` shows RTX A5000 + 16376 MiB VRAM
- [ ] `nvcc --version` shows CUDA 12.4
- [ ] `docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu24.04 nvidia-smi` succeeds
- [ ] `pixi install` completes without errors
- [ ] WaveArm policy runs 3 trials and `scoring.yaml` is written

**FSM (Week 1 gate)**
- [ ] `noise_floor.py` run; FORCE_CONTACT and FORCE_ABORT set from measurements
- [ ] CheatCode scored with `ground_truth:=true` — ceiling recorded
- [ ] FSM-only policy scored with `ground_truth:=false` — gap to ceiling measured
- [ ] Force penalty (-12) not appearing in any trial's scoring.yaml
- [ ] Collision penalty (-24) not appearing in any trial's scoring.yaml

**IL (if used)**
- [ ] Data collection uses `ground_truth:=false`
- [ ] 50-demo sanity check passed before collecting full dataset
- [ ] `lerobot-train` runs without OOM at batch=16
- [ ] ACT validation loss decreasing by epoch 50
- [ ] ACT scored alone before adding Diffusion Policy

**CV (if used)**
- [ ] YOLO mAP50 > 0.85 on validation set
- [ ] YOLO falls back to FSM-only if no detection within 2 s
- [ ] Visual servo improves score vs FSM-only before being added to hybrid

**Submission**
- [ ] Policy node implements full lifecycle (configure/activate/deactivate/cleanup)
- [ ] `rosdep install` included in Dockerfile before `colcon build`
- [ ] Local 3-trial test averages > 70 pts/trial
- [ ] Image tag matches exact format in `docs/submission.md`
- [ ] `docker login` succeeds before `docker push`
