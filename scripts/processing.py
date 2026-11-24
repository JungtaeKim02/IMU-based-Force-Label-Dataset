import sys, os, json, glob
import numpy as np, pandas as pd, mujoco
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import butter, filtfilt
G = 9.81
g_B = np.array([0.0, 0.0, -G])
e_z = np.array([0.0, 0.0, 1.0])
(REST1_SKIP_S, REST3_SKIP_S) = (0.5, 0.2)
(b5, a5) = butter(4, 5.0 / 50.0)
(b2, a2) = butter(4, 2.0 / 50.0)

def rotvec_min(u, v):
    ax = np.cross(u, v)
    s = np.linalg.norm(ax)
    return np.zeros(3) if s < 1e-12 else ax / s * np.arctan2(s, float(u @ v))

class FK:

    def __init__(self, scene):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        self.gid = self.m.body('gripper').id
        self.c_G = self.m.site_pos[self.m.site('payload_com').id].copy()

    def R_B_G(self, qpos_rows):
        out = np.empty((len(qpos_rows), 3, 3))
        pos = np.empty((len(qpos_rows), 3))
        for (k, q) in enumerate(qpos_rows):
            self.d.qpos[:] = q
            mujoco.mj_kinematics(self.m, self.d)
            out[k] = self.d.xmat[self.gid].reshape(3, 3)
            pos[k] = self.d.xpos[self.gid] + out[k] @ self.c_G
        return (out, pos)

def relabel(df, fk):
    t = df.timestamp.to_numpy()
    ph = df.phase.to_numpy()
    q = df[['pos%d' % i for i in range(1, 7)]].to_numpy()
    a_B = df[['acc_com_base_x', 'acc_com_base_y', 'acc_com_base_z']].to_numpy()
    w_G = df[['omega_grip_x', 'omega_grip_y', 'omega_grip_z']].to_numpy()
    F_old_raw = df[['force_raw_x', 'force_raw_y', 'force_raw_z']].to_numpy()
    assert np.isfinite(a_B).all() and np.isfinite(w_G).all() and np.isfinite(q).all()
    m = float(np.median(F_old_raw[:, 2] / (-G - a_B[:, 2])))
    (R_fk, p_c_B) = fk.R_B_G(q)
    dt = float(np.median(np.diff(t)))
    a_fk = np.zeros_like(p_c_B)
    a_fk[1:-1] = (p_c_B[2:] - 2 * p_c_B[1:-1] + p_c_B[:-2]) / dt ** 2
    (a_fk[0], a_fk[-1]) = (a_fk[1], a_fk[-2])
    F_fkref = filtfilt(b5, a5, m * (g_B - a_fk), axis=0)
    s_G = np.einsum('nji,nj->ni', R_fk, a_B - g_B)
    assert np.abs(F_old_raw - m * (g_B - a_B)).max() < 0.002
    (p1, p3) = (np.where(ph == 1)[0], np.where(ph == 3)[0])
    k_ref = p1[0]
    n = len(t)
    Q = [None] * n
    Q[k_ref] = Rot.identity()
    for k in range(k_ref, n - 1):
        Q[k + 1] = Q[k] * Rot.from_rotvec(0.5 * (w_G[k] + w_G[k + 1]) * (t[k + 1] - t[k]))
    for k in range(k_ref, 0, -1):
        Q[k - 1] = Q[k] * Rot.from_rotvec(-0.5 * (w_G[k - 1] + w_G[k]) * (t[k] - t[k - 1]))
    R_gyro = Rot.from_matrix(R_fk[k_ref]) * Rot.concatenate(Q)
    up_B = R_gyro.apply(s_G)
    up_B /= np.linalg.norm(up_B, axis=1, keepdims=True)
    w1 = p1[t[p1] >= t[p1[0]] + REST1_SKIP_S]
    w3 = p3[t[p3] >= t[p3[0]] + REST3_SKIP_S]
    u1 = up_B[w1].mean(axis=0)
    u1 /= np.linalg.norm(u1)
    u3 = up_B[w3].mean(axis=0)
    u3 /= np.linalg.norm(u3)
    (phi1, phi3) = (rotvec_min(u1, e_z), rotvec_min(u3, e_z))
    (tc1, tc3) = (t[w1].mean(), t[w3].mean())
    al = np.clip((t - tc1) / (tc3 - tc1), 0.0, 1.0)
    phi = (1 - al)[:, None] * phi1 + al[:, None] * phi3
    R_hat = Rot.from_rotvec(phi) * R_gyro
    F_new_raw = -m * R_hat.apply(s_G)
    F_new = filtfilt(b5, a5, F_new_raw, axis=0)
    F_old = df[['force_x', 'force_y', 'force_z']].to_numpy()
    up_fk = np.einsum('nij,nj->ni', R_fk, s_G)
    up_fk /= np.linalg.norm(up_fk, axis=1, keepdims=True)
    tilt = lambda w: float(np.degrees(np.arccos(np.clip(up_fk[w].mean(axis=0) @ e_z / np.linalg.norm(up_fk[w].mean(axis=0)), -1, 1))))
    info = dict(m=m, tilt_fk_rest1_deg=tilt(w1), tilt_fk_rest3_deg=tilt(w3), drift_deg=float(np.degrees(np.linalg.norm(phi3 - phi1))), span_s=float(tc3 - tc1), heading_change_deg=float(np.degrees(np.abs((Rot.from_matrix(R_fk[w3[-1]]).inv() * R_hat[w3[-1]]).as_rotvec() @ (R_fk[w3[-1]].T @ e_z)))))
    return (F_new, F_new_raw, F_old, F_fkref, info)
