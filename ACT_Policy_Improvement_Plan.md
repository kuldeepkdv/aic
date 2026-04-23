# Improving the ACT Policy for the AIC Cable‑Insertion Challenge

## Context

The **AI for Industry Challenge (AIC)** qualification task is autonomous
flexible‑cable insertion in Gazebo simulation: a UR5e arm with an already‑grasped
connector must insert it into a randomized task board. Three trials:

| Trial | Connector | Target |
|-------|-----------|--------|
| 1–2 | SFP module | SFP port on NIC card (Zone 1) |
| 3 | SC fiber‑optic plug | SC port on optical patch panel (Zone 2) |

The task board pose, rail positions, and specific target port are randomized per
trial. NIC card translation: ±21.5 mm, rotation ±10°; SC port translation: ±60 mm.
Partial‑insertion detection tolerance: **5 mm x‑y**.

The repo ships a **reference ACT baseline** at
`aic_example_policies/aic_example_policies/ros/RunACT.py` that loads a LeRobot
ACT checkpoint from HuggingFace (`grkw/aic_act_policy`). It demonstrates the
pipeline but scores poorly due to low control rate, missing force sensing,
fixed impedance, and no success detection.

---

## 1. How ACT Works (Action Chunking Transformer)

ACT (Zhao et al., ALOHA 2023) is a behavior‑cloning architecture for fine
manipulation:

### Architecture
- **Vision encoder**: one ResNet18 backbone per camera view → flattened
  spatial features projected to transformer tokens.
- **Proprioceptive token**: flat state vector (TCP pose + vel + error +
  joints) linearly projected and prepended.
- **CVAE latent `z`**: during training, an encoder transformer maps the
  ground‑truth action chunk → `z` (Gaussian). The decoder is conditioned on
  `z` + image tokens + state. At inference, `z = 0` (prior mean) → single
  deterministic mode.
- **Transformer decoder**: cross‑attends to encoder output; produces a chunk
  of **K future actions** in one forward pass (typically K = 50–100).

### Action Chunking
Instead of predicting one action per step (which compounds BC error), ACT
predicts K steps. At each control tick, overlapping chunks from prior ticks
are **temporally ensembled** (exponential moving average with coefficient
`temporal_ensemble_coeff`) → smooth, stable output.

### Loss
`L = L1(predicted_chunk, true_chunk) + kl_weight * KL(z_posterior || z_prior)`

---

## 2. How RunACT Wires ACT in This Repo

### Loading (`RunACT.__init__`, line 55)
- Downloads `config.json`, `model.safetensors`, and normalization stats from
  `grkw/aic_act_policy` via `huggingface_hub.snapshot_download`.
- Strips the `"type"` field from config dict (draccus compat fix, line 76),
  then `draccus.decode(ACTConfig, config_dict)`.
- Loads `ACTPolicy(config)` + safetensors weights onto CUDA.
- Loads per‑key mean/std for 3 images (shape `1,3,1,1`), 26‑dim state
  (shape `1,26`), and 7‑dim action (shape `1,7`).

### Observations (`prepare_observations`, line 169)
Builds a dict of normalized tensors:

| Key | Source | Shape |
|-----|--------|-------|
| `observation.images.{left,center,right}_camera` | ROS Image → resize 0.25× → CHW float/255 → `(x−μ)/σ` | `(1,3,256,288)` |
| `observation.state` | 26‑dim flat vector (below) → `(x−μ)/σ` | `(1,26)` |

**26‑dim state vector** (lines 198–227):
```
TCP position (3) + TCP quaternion xyzw (4) + TCP linear vel (3) +
TCP angular vel (3) + TCP error (6) + joint positions (7) = 26
```

**Critical gap**: `obs_msg.wrist_wrench` (6‑axis F/T at wrist) is available
in the `Observation` message but **not used** by RunACT. The LeRobot adapter
(`aic_robot_aic_controller.py:312–380`) also **does not record** wrench data —
so the training dataset lacks it entirely.

### Inference loop (`insert_cable`, line 237)
```python
policy.reset()  # clear temporal ensemble buffers
for 30 seconds at ~4 Hz:
    obs = get_observation()
    tensors = prepare_observations(obs)
    action[1,7] = policy.select_action(tensors)  # first chunk element
    raw = action * action_std + action_mean       # un-normalize
    twist = Twist(linear=raw[0:3], angular=raw[3:6])  # index 6 unused
    motion_update = MotionUpdate(
        velocity=twist,
        stiffness=diag([100,100,100,50,50,50]),
        damping=diag([40,40,40,15,15,15]),
        wrench_feedback_gains=[0.5,0.5,0.5,0,0,0],
        mode=MODE_VELOCITY
    )
    move_robot(motion_update)
    sleep(0.25 - elapsed)
return True  # unconditional
```

### Training / Data Pipeline (`aic_utils/lerobot_robot_aic/`)
- **Teleop**: `lerobot-teleoperate` with `aic_keyboard_ee` (WASD, 0.02–0.1 m/s),
  `aic_keyboard_joint` (0.02–0.05 rad/s), or `aic_spacemouse` (6DOF, 0.1 scaling,
  0.02 deadband). Defined in `aic_teleop.py`.
- **Record**: `lerobot-record` captures images + state + actions at 20 Hz.
  Arrow keys = next/redo episode, ESC = stop.
- **Train**: `lerobot-train --policy.type=act` trains on HuggingFace dataset;
  pushes checkpoints to Hub.
- **Robot adapter** (`aic_robot_aic_controller.py`): records 26‑dim state +
  3 cameras; sends Cartesian twist with fixed impedance
  `stiffness=[85,85,85,85,85,85]`, `damping=[75,75,75,75,75,75]` (lines 406–407).

---

## 3. Scoring Breakdown (Exact Formulas)

**Max per trial: 100 points. Source: `docs/scoring.md`.**

### Tier 1 — Model Validity (0–1 pt)
Pass/fail: policy loads, accepts `/insert_cable` action, publishes valid
`MotionUpdate` or `JointMotionUpdate` commands. Prerequisite for all other tiers.

### Tier 2 — Performance (net range: +24 to −36 pts)
Only awarded if Tier 3 score > 0 (i.e., plug near target).

| Metric | Min → Max | Formula |
|--------|-----------|---------|
| **Smoothness** | 0–6 pts | Jerk = 0 m/s³ → 6; Jerk ≥ 50 m/s³ → 0; linear interp. Jerk via Savitzky‑Golay (15‑sample window). |
| **Duration** | 0–12 pts | ≤ 5 s → 12; ≥ 60 s → 0; linear interp. |
| **Efficiency** | 0–6 pts | Path length ≤ initial distance → 6; ≥ initial + 1 m → 0; linear interp. |
| **Force penalty** | 0 to −12 | −12 if wrist force > **20 N for > 1 s** cumulative. |
| **Collision penalty** | 0 to −24 | −24 for **any** contact with enclosure/walls/task‑board off‑limits. |

### Tier 3 — Cable Insertion (0–75 pts)
| Outcome | Points |
|---------|--------|
| Full correct insertion | **75** |
| Wrong port insertion | **−12** |
| Partial insertion (in bounding box, ≤5 mm x‑y) | **38–50** (proportional to depth) |
| Proximity (plug near port entrance) | **0–25** (inversely proportional to distance; max distance = ½ initial gap) |

**Key insight**: success (75 pts) dominates. Force/collision penalties can
erase performance gains. Duration matters only after success.

---

## 4. Why the Baseline Underperforms

| Problem | Scoring Impact | Code Location |
|---------|----------------|---------------|
| **4 Hz control rate** vs 20 Hz dataset | High jerk → poor smoothness (−6), temporal‑ensemble mismatch | `RunACT.py:292` (`sleep 0.25`) |
| **No F/T wrench in obs** | Cannot detect contact → force spikes → −12 penalty; cannot sense insertion | `RunACT.py:169` (missing), `aic_robot_aic_controller.py:312` (not recorded) |
| **Fixed impedance** `[100,…,50,…]` | Too soft for approach (drift), too stiff for contact (force spikes) | `RunACT.py:303–305` |
| **No success detection** | Runs full 30 s even after insertion → duration penalty (≤5 s = 12 pts vs 30 s ≈ 7 pts) | `RunACT.py:251` (`while < 30s`) |
| **Raw quaternion** (no sign canon) | `q` and `−q` represent same rotation → BC instability | `RunACT.py:209–212` |
| **Single connector type in training** | Trial 3 (SC) likely fails completely → 0/100 | Dataset scope |
| **Image scaling 0.25×** | 288×256 may lose fine connector/port features needed for sub‑mm alignment | `RunACT.py:131` |
| **MODE_VELOCITY only** | Velocity control accumulates drift; MODE_POSITION with trajectory interpolation is smoother (CheatCode uses position + 50 ms rate, line 236) | `RunACT.py:317` |

---

## 5. Recommended Improvements (Prioritized)

### Phase A — Data & Observations (Highest ROI)

**A1. Add F/T wrench to state vector (26 → 32 dims)**
- In `aic_robot_aic_controller.py`, add `wrist_wrench.wrench.{force,torque}.{x,y,z}`
  to `get_observation()` (after line 380).
- In `RunACT.prepare_observations()`, append 6 wrench dims to `state_np` array
  (line 202). Update normalization stats shape to `(1,32)`.
- **Must retrain** — stats won't match without this.

**A2. Collect diverse demonstrations (≥50 per connector)**
- Use `lerobot-record` with `aic_keyboard_ee` at slow speed (0.02 m/s) for
  precision phases. Record with randomized task‑board poses (re‑launch
  `aic_bringup` between episodes).
- Mix SFP and SC episodes in one dataset. Use `--dataset.single_task="Insert SFP"`
  / `"Insert SC"` to enable task‑conditioned ACT.
- Reject episodes with force > 15 N or collisions.

**A3. Canonicalize quaternions**
- In `prepare_observations`, after reading TCP quaternion: if `w < 0`, negate
  all four components. Prevents sign discontinuity across episodes.

### Phase B — Control Rate & Impedance

**B1. Match control rate to training: 20 Hz**
- Replace `time.sleep(max(0, 0.25 - elapsed))` with `sleep(max(0, 0.05 - elapsed))`
  in `RunACT.py:292`.
- Verify `ACTConfig.temporal_ensemble_coeff` is set (default ~0.01); this
  averages overlapping chunk predictions for stability at high rate.

**B2. Phase‑dependent impedance**
Reference values from example policies:
- **Free‑space approach** (no contact): stiffness `[300,300,300,100,100,100]`,
  damping `[40,40,40,15,15,15]` (cf. WallPresser, line 67).
  MODE_POSITION for smooth interpolation.
- **Contact / insertion phase** (`‖wrench.force‖ > 5 N`): stiffness
  `[50,50,50,20,20,20]`, damping `[40,40,40,20,20,20]` (cf. GentleGiant,
  line 52). MODE_VELOCITY for compliant motion. High damping prevents
  oscillation.
- Gate transition on `wrist_wrench.wrench.force` magnitude from observations
  (now available via A1).

**B3. Wrench feedback gains**
- Current: `[0.5,0.5,0.5,0,0,0]`. During insertion, enable torque feedback
  too: `[0.5,0.5,0.5,0.3,0.3,0.3]` — helps the controller resist
  misalignment torques. Keep gains < 0.95 (controller stability constraint
  per `aic_controller.md`).

### Phase C — Runtime Safety & Early Termination

**C1. Force safety watchdog**
- If `‖wrench.force‖ > 18 N` for > 0.5 s, retract 5 mm along negative
  insertion axis then resume. Avoids crossing the **20 N / 1 s** penalty
  threshold (−12 pts).

**C2. Collision avoidance via velocity clamping**
- Cap twist linear magnitude to 0.05 m/s and angular to 0.2 rad/s.
  Prevents large BC errors from causing wall contact (−24 pts).

**C3. Success detection & early exit**
- Monitor `controller_state.tcp_error`: if position error magnitude < 2 mm
  AND wrench along insertion axis > 3 N (plug seated), hold for 1 s to
  confirm, then `return True`.
- Saves 20–25 s → duration score jumps from ~7 to ~12 pts.

### Phase D — Architecture Tuning

**D1. Increase action chunk size to K=50**
- Modify `ACTConfig.chunk_size=50` before training. Matches ALOHA defaults.
  Longer chunks smooth contact‑phase multimodality.

**D2. Higher image resolution**
- Set `image_scaling=0.5` (576×512) for better connector feature detection.
  Profile inference: must stay < 50 ms/tick at 20 Hz.

**D3. CVAE hyperparameters**
- `use_vae=true`, `kl_weight=10`, `dropout=0.1` to fight overfitting on
  small datasets while preserving multi‑modal action coverage.

**D4. Larger backbone**
- Try `vision_backbone="resnet34"` if GPU memory allows. Monitor
  training/validation L1 gap.

### Phase E — Advanced (Optional)

**E1. Hybrid ACT + scripted fallback**
- If ACT makes no progress (TCP distance to target doesn't decrease) for 3 s,
  switch to a spiral search pattern around current pose (inspired by
  CheatCode's integrator + z‑descent strategy, lines 161, 243), then hand
  back to ACT.

**E2. Residual RL fine‑tuning**
- Use the BC policy as initialization; add PPO/SAC residual on top. Requires
  Isaac Lab or MuJoCo mirror env (mentioned in `docs/qualification_phase.md`).

**E3. Domain randomization**
- Vary lighting, texture, and camera noise in Gazebo during data collection.
  Improves ResNet generalization to evaluation randomization.

---

## 6. Critical Files to Modify

| File | Purpose |
|------|---------|
| `aic_example_policies/aic_example_policies/ros/RunACT.py` | Main policy — rewrite inference loop, observations, impedance |
| `aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_robot_aic_controller.py` | LeRobot adapter — add wrench to `get_observation()` (line 312+) |
| `aic_model/aic_model/policy.py` | Base class reference — `set_pose_target` defaults (line 89), callback protocols |

**Reference files (read‑only)**:
| File | Why |
|------|-----|
| `aic_example_policies/.../CheatCode.py` | Ground‑truth strategy: SLERP orientation, integrator feedback, z‑descent |
| `aic_example_policies/.../GentleGiant.py` | Low‑jerk impedance reference: stiffness `[50,50,50,20,20,20]` |
| `aic_control_interfaces/msg/MotionUpdate.msg` | Full field spec for Cartesian commands |
| `aic_control_interfaces/msg/ControllerState.msg` | TCP pose/vel/error + F/T tare offset |
| `docs/scoring.md` | Exact scoring formulas and thresholds |
| `docs/aic_controller.md` | Impedance control math, MODE_POSITION vs MODE_VELOCITY |

---

## 7. Verification Plan

1. **Lint / type‑check**: `pixi run ruff check` and `pyright` on modified files.
2. **Dataset validation**: after recording with wrench, confirm 32‑dim state
   vectors at 20 Hz; spot‑check image shapes match `(1152,1024,3)` raw →
   `(576,512,3)` at 0.5× or `(288,256,3)` at 0.25×.
3. **Training convergence**: `pixi run lerobot-train --policy.type=act
   --dataset.repo_id=<local>` — validation L1 loss < 0.05 on held‑out 10%
   episodes within 100 epochs.
4. **Single‑trial smoke test**: launch Gazebo + aic_engine with
   `aic_engine/config/sample_config/` SFP trial → confirm:
   - Policy loads (Tier 1 pass)
   - No force spikes > 20 N (check `/fts_broadcaster/wrench` in rosbag)
   - No enclosure contacts (check scoring output)
   - Plug moves toward target port (Tier 3 proximity > 0)
5. **Full 3‑trial scoring**: run SFP trial 1, SFP trial 2, SC trial 3 →
   read `aic_scoring` output. Target: ≥ 60/100 per trial (180/300 total).
6. **Container submission check**: build `docker/aic_model` image; verify
   lifecycle transitions complete within 60 s timeouts per
   `docs/challenge_rules.md`.
