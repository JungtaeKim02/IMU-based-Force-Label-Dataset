# Data format

Joint indices 1–6 correspond to `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper`.

## Raw logs

| File | Contents and units |
|---|---|
| `arm.csv` | Host monotonic timestamps in ns and per-servo STS3215 registers. Position uses 0–4095 counts; velocity uses steps/s; signed load uses 0.1% per count; voltage uses 0.1 V per count; current uses 6.5 mA per count; temperature is °C. `ok` and `status` indicate read validity. |
| `imu.csv` | Host receive time `rx_ns`, device time `cf_ms`, accelerometer specific force `acc_*` in g, and angular velocity `gyro_*` in degrees/s. |
| `att.csv` | Host/device timestamps and the board's reported roll, pitch, yaw in degrees. These reported angles are retained as logs; final labels use the corrected attitude procedure. |
| `meta.json` | Clock anchor, joint order, servo configuration, calibration, and acquisition metadata. |
| `replay.json` | Commanded motion ID, script start on the host monotonic clock, approach interval, and trajectory interval. |

## Synchronized and motion CSVs

Synchronized files retain the original read repairs, clock alignment, delay correction, and CoM acceleration conversion. Sample timing follows the approximately 100 Hz arm stream and is not perfectly uniform. Motion CSVs preserve the model-input cells and replace the force labels with the final corrected labels.

| Columns | Meaning and units |
|---|---|
| `timestamp` | Seconds from the replay-script start in synchronized files; seconds from the first selected trajectory sample in motion files. `replay.json` links script time to the raw host clock. |
| `pos1..6`, `goal_pos1..6` | Measured and commanded model joint coordinates, rad. The calibrated joint map is already applied. |
| `vel1..6`, `load1..6`, `volts1..6`, `current1..6`, `temp1..6` | Servo register values in the raw units listed above, retained for compatibility with model inputs. These columns are not converted to SI units. |
| `phase` | 0: approach; 1: initial stationary interval; 2: commanded trajectory, including its internal holds; 3: final stationary interval. Motion files contain phase 2 only. |
| `ref_time` | Commanded reference time in s; blank outside the trajectory interval. |
| `arm_filled`, `imu_valid` | Repair and validity flags. |
| `imu_acc_*`, `imu_gyro_*`, `cf_roll`, `cf_pitch`, `cf_yaw` | Resampled sensor channels, respectively g, degrees/s, and degrees. |
| `omega_grip_*` | Calibrated angular velocity expressed in the gripper frame, rad/s. |
| `acc_obj_base_*`, `acc_drone_base_*`, `acc_com_base_*` | Original IMU-derived object, board, and combined CoM acceleration in the base frame, m/s², before the final attitude correction. |
| `acc_com_fk_base_*` | Original FK acceleration diagnostic, m/s². |
| `force_*`, `force_raw_*` | Final 5 Hz-filtered and prefilter IMU force labels in motion files, N. In synchronized input files these names retain the pre-correction labels. |
| `force_fkatt_*` | Pre-correction, FK-attitude IMU labels retained in motion files, N. |
| `force_fkref_*` | FK CoM position, central second difference using median sample interval, and a 5 Hz filter, N. Evaluation additionally applies the common 2 Hz filter after next-timestamp alignment. |
| `force_obj_*`, `force_fk_*` | Original object-force and FK diagnostic channels, N; not the final training labels or evaluation reference. |

Force-component suffixes `x`, `y`, `z` use the robot base frame. The mass-weighted object-plus-board CoM is used for the IMU label; the FK comparison uses the gripper's `payload_com` site.

## Model alignment

The feature order is six values each of `goal_pos`, `pos`, `load`, `vel`, `volts`, and `temp`, followed by six `goal_pos − pos` values. Training-only mean/std normalization and clipping to `[−10, 10]` precede window construction. Every trial starts with eight zero frames in normalized coordinates. Input rows `t−8..t` predict force at row `t+1`; therefore each trial contributes one fewer prediction than motion samples. The eight test trials yield 5,949 predictions.

## Trajectories and media

`reference.csv` contains time in s, six joint coordinates in rad, commanded CoM position in m, velocity in m/s, and acceleration in m/s². These are commanded trajectories, not measured robot motion. `motion_000` is the first trajectory and `motion_006` is the seventh. The gripper reference is a placeholder; the recorded grasp is held during acquisition.

`media/representatives.json` links the force-data trial IDs, trajectory files, and force plots.
