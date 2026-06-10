#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/06/10 18:25
# @Author : HuHanYing
"""
Pico4 -> Panthera VR 遥操作桥接脚本。

已实现：主控制循环、按住跟随（clutch）、超时急停、速度限幅、IK 连续求解、
XR 数据读取 + Panthera 真机通讯。

仿真 ``wrs`` 包优先加载仓库 ``wrs_vr/wrs/``（与 ``wrs_vr/teleop_bridge_panthera.py`` 同源）；
真机仍使用本目录 ``fafu_robot_python/``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prefer bundled simulation stack in wrs_vr/; fallback to monorepo wrs/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRS_VR_ROOT = _REPO_ROOT / "wrs_vr"
if _WRS_VR_ROOT.is_dir():
    _sim_path = str(_WRS_VR_ROOT)
else:
    _sim_path = str(_REPO_ROOT)
if _sim_path not in sys.path:
    sys.path.insert(0, _sim_path)

import argparse
import importlib.util
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from wrs import wd, mgm, rm
from wrs.robot_sim.robots.panthera_ht.panthera_ht import PantheraHTSglArm


@dataclass
class XRState:
    """单帧 XR 输入数据。

    字段说明：
    - timestamp_s: 这一帧数据的时间戳（秒）
    - controller_pos_vr: 控制器在 VR 世界坐标系中的位置
    - controller_quat_vr: 控制器姿态四元数 [x, y, z, w]
    - grip_pressed: 按住时才让机械臂跟随（防误触）
    - grip_value: grip 原始模拟量（0~1），用于迟滞防抖
    - estop_pressed: 软件急停按键
    - button_a_pressed: A 键（用于夹爪打开）
    - button_b_pressed: B 键（用于夹爪闭合）
    """
    timestamp_s: float
    controller_pos_vr: np.ndarray     # shape (3,)
    controller_quat_vr: np.ndarray    # shape (4,), [x, y, z, w]
    grip_pressed: bool
    estop_pressed: bool
    button_a_pressed: bool
    button_b_pressed: bool
    grip_value: float = 0.0


class JsonlReplayReader:
    """从 JSONL 回放 XR 数据，并按录制时间戳节奏输出。"""

    def __init__(self, jsonl_path: str, replay_speed: float = 1.0) -> None:
        self._path = Path(jsonl_path)
        self._replay_speed = max(float(replay_speed), 1e-6)
        self._frames: list[XRState] = []
        self._idx = 0
        self._start_wall: Optional[float] = None
        self._start_ts: Optional[float] = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Replay file not found: {self._path}")
        raw_frames: list[XRState] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                    frame = XRState(
                        timestamp_s=float(obj["timestamp_s"]),
                        controller_pos_vr=np.asarray(obj["controller_pos_vr"], dtype=float),
                        controller_quat_vr=np.asarray(obj["controller_quat_vr"], dtype=float),
                        grip_pressed=bool(obj["grip_pressed"]),
                        grip_value=float(
                            obj.get(
                                "grip_value",
                                1.0 if bool(obj["grip_pressed"]) else 0.0,
                            )
                        ),
                        estop_pressed=bool(obj["estop_pressed"]),
                        button_a_pressed=bool(obj.get("button_a_pressed", False)),
                        button_b_pressed=bool(obj.get("button_b_pressed", False)),
                    )
                    if frame.controller_pos_vr.shape != (3,) or frame.controller_quat_vr.shape != (4,):
                        raise ValueError("pose shape mismatch")
                    raw_frames.append(frame)
                except Exception as exc:
                    print(f"[REPLAY] skip invalid line {line_idx}: {exc}")

        # 清理时间轴：
        # 1) 丢弃 timestamp<=0 的前导无效帧（常见于 SDK 刚启动阶段）；
        # 2) 保证时间戳单调递增，避免 poll 等待异常长时间。
        cleaned: list[XRState] = []
        for frm in raw_frames:
            if frm.timestamp_s <= 0.0:
                continue
            if cleaned and frm.timestamp_s <= cleaned[-1].timestamp_s:
                frm = XRState(
                    timestamp_s=cleaned[-1].timestamp_s + 1e-4,
                    controller_pos_vr=frm.controller_pos_vr,
                    controller_quat_vr=frm.controller_quat_vr,
                    grip_pressed=frm.grip_pressed,
                    grip_value=frm.grip_value,
                    estop_pressed=frm.estop_pressed,
                    button_a_pressed=frm.button_a_pressed,
                    button_b_pressed=frm.button_b_pressed,
                )
            cleaned.append(frm)

        self._frames = cleaned
        if not self._frames:
            raise ValueError(f"No valid replay frame in file: {self._path}")
        print(f"[REPLAY] loaded {len(self._frames)} frames from {self._path}")

    def poll(self) -> Optional[XRState]:
        """到时间就吐出下一帧；没到时间返回 None。"""
        if self._idx >= len(self._frames):
            return None

        frame = self._frames[self._idx]
        now = time.time()
        if self._start_wall is None:
            self._start_wall = now
            self._start_ts = frame.timestamp_s
            self._idx += 1
            return frame

        assert self._start_ts is not None
        target_wall = self._start_wall + (frame.timestamp_s - self._start_ts) / self._replay_speed
        if now >= target_wall:
            self._idx += 1
            return frame
        return None

    def is_finished(self) -> bool:
        return self._idx >= len(self._frames)


_REPLAY_READER: Optional[JsonlReplayReader] = None
_XRT = None
_XR_CONTROLLER_SIDE = "right"
_XR_GRIP_THRESHOLD = 0.5
_XR_ESTOP_BUTTON = "A"
_XR_HOME_BUTTON = "RIGHT_TRIGGER"
_LAST_XR_TIMESTAMP_NS: Optional[int] = None
_XR_LINK_READY = False
_LAST_XR_POLL_WALL_TS = 0.0
# ---------------- Runtime Mode Switch (edit here) ----------------
# 直接在脚\本里切模式：
# - "live"   : 实时 XR 控制
# - "replay" : 离线回放
DEFAULT_RUNTIME_MODE = "live"
# ------------------------------------------------------------------
# ---------------- Robot Control Switch (edit here) ----------------
# - "sim"  : 仅仿真，不连接真机
# - "real" : 连接本目录 fafu_robot_python/ 真机驱动
DEFAULT_CONTROL_MODE = "real"
# ------------------------------------------------------------------
# 为空表示未指定默认回放文件；replay 模式下可在此填写文件名
DEFAULT_REPLAY_FILENAME = "xr_controller_20260520_151710.jsonl"
# ---------------- Calibration Config (edit here) ----------------
# VR世界坐标 -> 机器人基坐标 的旋转欧拉角（度，XYZ顺序）
# Pico / OpenXR：VR+X=右, VR+Y=上, VR+Z=后（向前 = -Z）。
# Panthera 基座：+X=前, +Y=左, +Z=上。
# [90, 0, -90]：让操作者方向与机械臂方向一致
#   VR+X(右)  -> robot-Y(右)
#   VR+Y(上)  -> robot+Z(上)
#   VR+Z(后)  -> robot-X(后)
# 历史值 [90,0,0]：VR+X→robot+X(前), VR+Z→robot-Y(右)，会出现「左右↔前后」对调。
CALIB_RPY_DEG_BR = np.array([90.0, 0.0, -90.0], dtype=float)
# VR世界坐标 -> 机器人基坐标 的平移（米）
CALIB_T_BR = np.array([0.0, 0.0, 0.0], dtype=float)
# ----------------------------------------------------------------


def _xr_safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _as_pose7(value) -> np.ndarray:
    """
    将 SDK 返回的 pose 统一转为 [x,y,z,qx,qy,qz,qw]。
    """
    if value is None:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    try:
        if hasattr(value, "tolist"):
            seq = value.tolist()
        else:
            seq = list(value)
        if len(seq) >= 7:
            return np.asarray(seq[:7], dtype=float)
    except Exception:
        pass
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)


def _get_button_pressed(btn_name: str) -> bool:
    if _XRT is None:
        return False
    btn = str(btn_name).upper()
    if btn == "A":
        return bool(_xr_safe_call(_XRT.get_A_button, 0))
    if btn == "B":
        return bool(_xr_safe_call(_XRT.get_B_button, 0))
    if btn == "X":
        return bool(_xr_safe_call(_XRT.get_X_button, 0))
    if btn == "Y":
        return bool(_xr_safe_call(_XRT.get_Y_button, 0))
    if btn == "RIGHT_AXIS_CLICK":
        return bool(_xr_safe_call(_XRT.get_right_axis_click, 0))
    if btn == "LEFT_AXIS_CLICK":
        return bool(_xr_safe_call(_XRT.get_left_axis_click, 0))
    if btn == "RIGHT_TRIGGER":
        return float(_xr_safe_call(_XRT.get_right_trigger, 0.0)) >= _XR_GRIP_THRESHOLD
    if btn == "LEFT_TRIGGER":
        return float(_xr_safe_call(_XRT.get_left_trigger, 0.0)) >= _XR_GRIP_THRESHOLD
    return False


def _get_estop_pressed() -> bool:
    return _get_button_pressed(_XR_ESTOP_BUTTON)


def _get_home_pressed() -> bool:
    return _get_button_pressed(_XR_HOME_BUTTON)


def _resolve_replay_filename_in_logs(replay_filename: str) -> Path:
    """按文件名在脚本同目录 logs 中解析回放文件路径。"""
    logs_dir = Path(__file__).resolve().parent / "logs"
    return logs_dir / replay_filename


def quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """四元数 [x,y,z,w] -> 3x3 旋转矩阵。"""
    x, y, z, w = quat_xyzw.astype(float)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def rotmat_from_euler_xyz_deg(rpy_deg: np.ndarray) -> np.ndarray:
    """XYZ欧拉角(度) -> 旋转矩阵。"""
    rx, ry, rz = np.radians(np.asarray(rpy_deg, dtype=float))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rzm @ rym @ rxm


def skew(v: np.ndarray) -> np.ndarray:
    """向量转反对称矩阵（用于 Rodrigues 公式）。"""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """轴角 -> 旋转矩阵。"""
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12 or abs(theta) < 1e-12:
        return np.eye(3)
    k = axis / axis_norm
    kx = skew(k)
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def scale_rotation_delta(delta_r: np.ndarray, scale: float) -> np.ndarray:
    """按比例缩放相对旋转（轴不变，角度乘以 scale）。"""
    s = max(float(scale), 0.0)
    if abs(s - 1.0) < 1e-9:
        return np.asarray(delta_r, dtype=float)
    tr = float(np.trace(delta_r))
    cos_theta = float(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    theta = float(math.acos(cos_theta))
    if theta < 1e-9:
        return np.eye(3)
    axis = np.array([
        delta_r[2, 1] - delta_r[1, 2],
        delta_r[0, 2] - delta_r[2, 0],
        delta_r[1, 0] - delta_r[0, 1],
    ], dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return np.eye(3)
    axis = axis / axis_norm
    return rodrigues(axis, theta * s)


def scale_rotation_delta_axes(delta_r: np.ndarray, axis_gain: np.ndarray) -> np.ndarray:
    """按旋转向量分量缩放相对旋转（X/Y/Z 可独立调节灵敏度）。"""
    r = np.asarray(delta_r, dtype=float)
    g = np.asarray(axis_gain, dtype=float).reshape(3,)
    if np.allclose(g, np.ones(3), atol=1e-9):
        return r

    tr = float(np.trace(r))
    cos_theta = float(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    theta = float(math.acos(cos_theta))
    if theta < 1e-9:
        return np.eye(3)

    sin_theta = float(math.sin(theta))
    if abs(sin_theta) < 1e-9:
        return r

    axis = np.array([
        r[2, 1] - r[1, 2],
        r[0, 2] - r[2, 0],
        r[1, 0] - r[0, 1],
    ], dtype=float) / (2.0 * sin_theta)
    rotvec = axis * theta
    rotvec_scaled = rotvec * g

    theta_scaled = float(np.linalg.norm(rotvec_scaled))
    if theta_scaled < 1e-9:
        return np.eye(3)
    axis_scaled = rotvec_scaled / theta_scaled
    return rodrigues(axis_scaled, theta_scaled)


def limit_linear_step(cur: np.ndarray, tgt: np.ndarray, max_step: float) -> np.ndarray:
    """平移限幅：每个控制周期最多移动 max_step 米。"""
    delta = tgt - cur
    dist = np.linalg.norm(delta)
    if dist <= max_step or dist < 1e-12:
        return tgt
    return cur + delta / dist * max_step


def rotation_angle_rad(cur_r: np.ndarray, tgt_r: np.ndarray) -> float:
    """Return the rotation angle (rad) from cur_r to tgt_r."""
    r_err = np.asarray(cur_r, dtype=float).T @ np.asarray(tgt_r, dtype=float)
    trace = float(np.clip((np.trace(r_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(trace))


def limit_angular_step(cur_r: np.ndarray, tgt_r: np.ndarray, max_step_rad: float) -> np.ndarray:
    """姿态限幅：每个控制周期最多旋转 max_step_rad。"""
    r_err = np.asarray(cur_r, dtype=float).T @ np.asarray(tgt_r, dtype=float)
    angle = rotation_angle_rad(cur_r, tgt_r)
    if angle <= max_step_rad or angle < 1e-9:
        return tgt_r

    axis = np.array([
        r_err[2, 1] - r_err[1, 2],
        r_err[0, 2] - r_err[2, 0],
        r_err[1, 0] - r_err[0, 1],
    ])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return cur_r
    axis = axis / axis_norm
    return cur_r @ rodrigues(axis, max_step_rad)


def limit_joint_step(cur_q: np.ndarray, tgt_q: np.ndarray, max_step_rad: float) -> np.ndarray:
    """关节限幅：每个控制周期每个关节最多变化 max_step_rad。"""
    if max_step_rad <= 0:
        return np.asarray(tgt_q, dtype=float)
    cur_q = np.asarray(cur_q, dtype=float)
    tgt_q = np.asarray(tgt_q, dtype=float)
    dq = tgt_q - cur_q
    dq = np.clip(dq, -max_step_rad, max_step_rad)
    return cur_q + dq


# Shoulder ~45deg, moderate elbow; TCP z~0.16m (old [0,40,45,-15,0,0] was z~0.32m).
_TELEOP_READY_HOME_DEG = np.array([0.0, 45.0, 25.0, -10.0, 0.0, 0.0], dtype=float)


def teleop_ready_home_q(
    n_joints: int = 6,
    joints_deg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """VR teleop home (rad): mid-range ready pose, not all zeros."""
    n = max(int(n_joints), 1)
    if joints_deg is not None:
        deg = np.asarray(joints_deg, dtype=float).reshape(-1)
    else:
        deg = _TELEOP_READY_HOME_DEG[: min(n, _TELEOP_READY_HOME_DEG.size)]
    if deg.size < n:
        deg = np.concatenate([deg, np.zeros(n - deg.size, dtype=float)])
    elif deg.size > n:
        deg = deg[:n]
    return np.deg2rad(deg)


def resolve_right_home_q(n_joints: int, args) -> np.ndarray:
    """Default RIGHT home = joint zeros; optional ready pose or custom joints via CLI."""
    custom_deg = getattr(args, "right_home_joints_deg", None)
    if custom_deg is not None:
        return teleop_ready_home_q(n_joints, joints_deg=np.asarray(custom_deg, dtype=float))
    if bool(getattr(args, "right_home_ready_pose", False)):
        return teleop_ready_home_q(n_joints)
    n = max(int(n_joints), 1)
    return np.zeros(n, dtype=float)


def align_joint_solution_to_seed(q_try: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    """Map each joint angle to the 2pi-equivalent value nearest q_seed."""
    q_try = np.asarray(q_try, dtype=float)
    q_seed = np.asarray(q_seed, dtype=float)
    dq = q_try - q_seed
    dq = (dq + np.pi) % (2.0 * np.pi) - np.pi
    return q_seed + dq


def sync_q_seed_from_hardware(
    q_seed: np.ndarray,
    robot_model,
    arm_hw: "PantheraArmController",
    *,
    resync_servo: bool = False,
    tag: str = "SYNC",
) -> np.ndarray:
    """Read measured joints, update IK seed + sim model; optionally pin servo command chain."""
    try:
        q_real = arm_hw.get_joint_values()
        if (
            q_real is None
            or q_real.shape != q_seed.shape
            or not np.all(np.isfinite(q_real))
        ):
            print(f"[{tag}][WARN] hardware joint read failed shape/finite check.")
            return q_seed
        q_seed = np.asarray(q_real, dtype=float).copy()
        robot_model.goto_given_conf(q_seed)
        if resync_servo:
            arm_hw.resync_servo_to_measured()
        return q_seed
    except Exception as exc:
        print(f"[{tag}][WARN] could not sync q_seed from hardware: {exc}")
        return q_seed


def align_tcp_chain_from_q_seed(robot_model, q_seed: np.ndarray):
    """Update sim model and return (tcp_sent_pos, tcp_sent_rot, tcp_target_pos, tcp_target_rot)."""
    robot_model.goto_given_conf(q_seed)
    tcp_pos, tcp_rot = robot_model.fk(q_seed)
    return tcp_pos.copy(), tcp_rot.copy(), tcp_pos.copy(), tcp_rot.copy()


def reanchor_vr_tcp_on_servo_reject(
    *,
    xr,
    xr_rot: np.ndarray,
    tcp_sent_pos: np.ndarray,
    tcp_sent_rot: np.ndarray,
):
    """Reset VR/TCP anchors after servo reject so the next IK step does not re-chase lag."""
    return (
        tcp_sent_pos.copy(),
        tcp_sent_rot.copy(),
        xr.controller_pos_vr.copy(),
        xr_rot.copy(),
        None,
        None,
    )


def recover_q_seed_after_servo_reject(
    q_cmd: np.ndarray,
    arm_hw: "PantheraArmController",
    *,
    recover_step_rad: float,
) -> np.ndarray:
    """Step from measured toward command; avoids full snap-to-measured brake under load."""
    q_cmd = np.asarray(q_cmd, dtype=float)
    try:
        q_meas = np.asarray(arm_hw.get_joint_values(), dtype=float)
        if q_meas.shape != q_cmd.shape or not np.all(np.isfinite(q_meas)):
            raise ValueError("bad measured joints")
        step = max(float(recover_step_rad), math.radians(1.0))
        return limit_joint_step(q_meas, q_cmd, step)
    except Exception:
        try:
            return np.asarray(arm_hw.get_joint_values(), dtype=float).copy()
        except Exception:
            return q_cmd.copy()


def apply_servo_reject_recovery(
    q_seed: np.ndarray,
    robot_model,
    arm_hw: "PantheraArmController",
    *,
    xr,
    xr_rot: np.ndarray,
    q_cmd: Optional[np.ndarray] = None,
    recover_step_rad: float = math.radians(1.5),
):
    """Creep toward last command and re-anchor VR after a servo_j reject."""
    q_target = np.asarray(q_cmd if q_cmd is not None else q_seed, dtype=float)
    q_seed = recover_q_seed_after_servo_reject(
        q_target, arm_hw, recover_step_rad=recover_step_rad
    )
    if robot_model is not None:
        robot_model.goto_given_conf(q_seed)
    tcp_sent_pos, tcp_sent_rot, tcp_target_pos, tcp_target_rot = align_tcp_chain_from_q_seed(
        robot_model, q_seed
    )
    (
        tcp_pos_anchor,
        tcp_rot_anchor,
        xr_anchor_pos,
        xr_anchor_rot,
        last_ik_target_pos,
        last_ik_target_rot,
    ) = reanchor_vr_tcp_on_servo_reject(
        xr=xr,
        xr_rot=xr_rot,
        tcp_sent_pos=tcp_sent_pos,
        tcp_sent_rot=tcp_sent_rot,
    )
    return (
        q_seed,
        tcp_sent_pos,
        tcp_sent_rot,
        tcp_target_pos,
        tcp_target_rot,
        tcp_pos_anchor,
        tcp_rot_anchor,
        xr_anchor_pos,
        xr_anchor_rot,
        last_ik_target_pos,
        last_ik_target_rot,
    )


def _get_real_arm_controller_candidates():
    """Return available real arm controller candidates in priority order."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        (base_dir / "fafu_robot_python" / "fafu_robot_controller.py", "FafuRobotController"),
    ]
    loaded = []
    last_exc = None
    for controller_file, class_name in candidates:
        if not controller_file.exists():
            continue
        module_name = f"{controller_file.parent.name}_{controller_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(controller_file))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        # Register module before execution so decorators (e.g. @dataclass)
        # can resolve cls.__module__ via sys.modules during import.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            last_exc = exc
            sys.modules.pop(spec.name, None)
            continue
        if hasattr(module, class_name):
            loaded.append((controller_file, getattr(module, class_name)))
    if loaded:
        return loaded
    if last_exc is not None:
        raise RuntimeError(
            "真机模式需要 panthera_motor 模块（通常是与 fafu_robot_controller.py / "
            "wrs_panthera_arm_controller.py 同目录下的 panthera_motor*.so / .pyd）。"
            "若暂无上位机驱动，请使用 --control-mode sim 或把 DEFAULT_CONTROL_MODE 设为 sim。"
        ) from last_exc
    raise FileNotFoundError(
        f"Real controller not found: {base_dir / 'fafu_robot_python' / 'fafu_robot_controller.py'}"
    )


def _create_real_arm_controller(
    *,
    cfg_path: str,
    hw_port: Optional[str],
    has_gripper: bool,
    gripper_motor_id: int,
    init_rounds: int = 3,
    round_delay_s: float = 1.0,
    attempt_delay_s: float = 0.5,
):
    """Open one real arm backend with gripper fallback and COM-port-safe retries."""
    candidates = _get_real_arm_controller_candidates()
    controller_file, real_cls = candidates[0]
    gripper_motor_id_int = int(gripper_motor_id)
    gripper_tries: List[bool] = [bool(has_gripper)]
    if has_gripper:
        gripper_tries.append(False)

    init_err: Optional[Exception] = None
    port_label = hw_port if hw_port else "auto"
    for round_idx in range(max(int(init_rounds), 1)):
        if round_idx > 0:
            print(
                f"[HW] retry connect {port_label} "
                f"({round_idx + 1}/{init_rounds}) after {round_delay_s:.1f}s ..."
            )
            time.sleep(max(float(round_delay_s), 0.0))
        for try_has_gripper in gripper_tries:
            try:
                real_kwargs = dict(
                    cfg_path=cfg_path,
                    has_gripper=try_has_gripper,
                    gripper_motor_id=gripper_motor_id_int,
                )
                if hw_port:
                    real_kwargs["port"] = hw_port
                arm_real = real_cls(**real_kwargs)
                gripper_info = (
                    f"gripper M{gripper_motor_id_int}"
                    if try_has_gripper
                    else f"no gripper (skip M{gripper_motor_id_int})"
                )
                print(
                    f"[HW] real controller ready ({controller_file.name}), "
                    f"cfg={cfg_path}, port={port_label}, {gripper_info}"
                )
                if has_gripper and not try_has_gripper:
                    print(
                        f"[HW][WARN] gripper M{gripper_motor_id_int} unavailable; "
                        "continuing with 6-DOF arm only."
                    )
                return arm_real, try_has_gripper
            except Exception as exc:
                init_err = exc
                if try_has_gripper and len(gripper_tries) > 1:
                    print(
                        f"[HW][WARN] failed to init {controller_file.name} with gripper: {exc}; "
                        "retry without gripper."
                    )
                else:
                    print(f"[HW][WARN] failed to init {controller_file.name}: {exc}")
                time.sleep(max(float(attempt_delay_s), 0.0))

    raise RuntimeError(
        f"Failed to initialize real controller on {port_label} after "
        f"{max(int(init_rounds), 1)} round(s). "
        f"Run diag_motors.py --port {port_label} after power-cycling the arm."
    ) from init_err


class PantheraArmController:
    """
    真机控制

    建议最低实现能力：
    - enable/disable: 上下使能
    - emergency_stop: 急停
    - get_joint_values: 读当前关节（用于 IK seed）
    - send_joint_positions: 下发关节目标（建议非阻塞）
    """

    def __init__(
        self,
        control_mode: str = "sim",
        cfg_path: str = "",
        has_gripper: bool = True,
        gripper_motor_id: int = 7,
        joint_speed: int = 15,
        servo_watchdog_ms: int = 350,
        servo_max_vel: float = 3.0,
        servo_max_step_rad: float = math.radians(8.0),
        servo_max_lag_rad: float = math.radians(35.0),
        servo_cmd_max_step_rad: float = math.radians(2.0),
        servo_cmd_lp_alpha: float = 0.25,
        servo_recover_hold_ms: float = 150.0,
        servo_reject_hold_command: bool = True,
        servo_diag_enabled: bool = True,
        servo_rate_hz: float = 100.0,
        servo_feedforward_vel: bool = True,
        servo_lookahead_time_s: float = 0.05,
        gripper_effort: Optional[int] = None,
        gripper_grasp_force: Optional[int] = None,
        gripper_grasp_vel: float = 0.15,
        gripper_grasp_timeout: float = 1.5,
        hw_port: Optional[str] = None,
    ) -> None:
        self._control_mode = control_mode
        self._joint_speed = int(joint_speed)
        self._enabled = False
        self._jaw_open = True
        self._arm_real = None
        self._hw_has_gripper = bool(has_gripper)
        self._gripper_effort = int(gripper_effort) if gripper_effort is not None else None
        self._gripper_grasp_force = int(gripper_grasp_force) if gripper_grasp_force is not None else None
        self._gripper_grasp_vel = max(float(gripper_grasp_vel), 1e-4)
        self._gripper_grasp_timeout = max(float(gripper_grasp_timeout), 0.1)
        self._gripper_op_lock = threading.Lock()
        self._gripper_worker: Optional[threading.Thread] = None
        self._supports_servo_j = False
        self._servo_reject_count = 0
        self._last_servo_warn_t = 0.0
        self._last_servo_cmd_q: Optional[np.ndarray] = None
        self._servo_hold_until_t = 0.0
        self._servo_watchdog_ms = max(int(servo_watchdog_ms), 50)
        self._servo_max_vel = max(float(servo_max_vel), 0.1)
        self._servo_max_step_rad = max(float(servo_max_step_rad), 1e-4)
        # 0 = disable firmware max_lag reject (teach mirror). Do NOT clamp to 1e-3.
        lag_rad = float(servo_max_lag_rad)
        self._servo_max_lag_rad = lag_rad if lag_rad > 0.0 else 0.0
        self._servo_cmd_max_step_rad = max(float(servo_cmd_max_step_rad), 1e-4)
        self._servo_cmd_lp_alpha = float(np.clip(float(servo_cmd_lp_alpha), 0.0, 1.0))
        self._servo_recover_hold_s = max(float(servo_recover_hold_ms), 0.0) / 1000.0
        self._servo_reject_hold_command = bool(servo_reject_hold_command)
        self._servo_diag_enabled = bool(servo_diag_enabled)
        self._servo_rate_hz = max(float(servo_rate_hz), 1.0)
        self._servo_feedforward_vel = bool(servo_feedforward_vel)
        self._servo_lookahead_time_s = max(float(servo_lookahead_time_s), 0.0)
        if self._servo_cmd_max_step_rad > self._servo_max_step_rad:
            self._servo_cmd_max_step_rad = self._servo_max_step_rad
        self._diag_prev_cmd_q: Optional[np.ndarray] = None
        self._diag_prev_meas_q: Optional[np.ndarray] = None
        self._diag_prev_cmd_dq: Optional[np.ndarray] = None
        self._diag_count = 0
        self._diag_cmd_step_sum = 0.0
        self._diag_cmd_step_max = 0.0
        self._diag_cmd_vel_sum = 0.0
        self._diag_cmd_vel_max = 0.0
        self._diag_meas_step_sum = 0.0
        self._diag_meas_step_max = 0.0
        self._diag_track_err_sum = 0.0
        self._diag_track_err_max = 0.0
        self._diag_chatter_count = 0
        self._diag_worst_joint_idx = -1
        self._diag_worst_joint_err = 0.0
        self._diag_last_reject_count = 0
        if self._control_mode == "real":
            if not cfg_path:
                raise ValueError("Real mode requires cfg_path")
            self._arm_real, self._hw_has_gripper = _create_real_arm_controller(
                cfg_path=cfg_path,
                hw_port=hw_port,
                has_gripper=bool(has_gripper),
                gripper_motor_id=int(gripper_motor_id),
            )
            self._supports_servo_j = hasattr(self._arm_real, "servo_j")
        else:
            print("[HW] simulation controller mode (no real hardware connection).")

    def enable(self) -> None:
        if self._arm_real is not None:
            self._arm_real.enable()
            if self._supports_servo_j and hasattr(self._arm_real, "servo_start"):
                try:
                    # Use a more tolerant servo profile for teleop:
                    # - larger watchdog to avoid sporadic brake on loop jitter
                    # - larger max_lag / max_step to avoid reject storms under load
                    # - higher max_vel to help the lagging joint catch up
                    servo_opts = None
                    mod = sys.modules.get(self._arm_real.__class__.__module__)
                    if mod is not None and hasattr(mod, "ServoOpts"):
                        ServoOptsCls = getattr(mod, "ServoOpts")
                        servo_kwargs = dict(
                            watchdog_ms=self._servo_watchdog_ms,
                            max_vel=self._servo_max_vel,
                            max_step_rad=self._servo_max_step_rad,
                            max_lag_rad=self._servo_max_lag_rad,
                            is_radians=True,
                        )
                        # Compatibility with old/new fafu_robot_python:
                        # only pass fields supported by the current ServoOpts.
                        ann = getattr(ServoOptsCls, "__annotations__", {})
                        if "rate_hz" in ann:
                            servo_kwargs["rate_hz"] = self._servo_rate_hz
                        if "feedforward_vel" in ann:
                            servo_kwargs["feedforward_vel"] = self._servo_feedforward_vel
                        if "lookahead_time" in ann:
                            servo_kwargs["lookahead_time"] = self._servo_lookahead_time_s
                        servo_opts = ServoOptsCls(**servo_kwargs)
                    if servo_opts is not None:
                        self._arm_real.servo_start(servo_opts)
                    else:
                        self._arm_real.servo_start()
                    print("[HW] servo_j mode enabled")
                except Exception as exc:
                    print(f"[HW][WARN] servo_start failed, fallback to move_j: {exc}")
                    self._supports_servo_j = False
        self._enabled = True
        print("[HW] enable called")

    def disable(self) -> None:
        self._join_gripper_worker(timeout_s=0.2)
        if self._arm_real is not None:
            try:
                if self._supports_servo_j and hasattr(self._arm_real, "servo_end"):
                    try:
                        self._arm_real.servo_end(finish_mode="hold")
                    except Exception:
                        pass
                self._arm_real.disable()
            finally:
                self._arm_real.close_connection()
        self._enabled = False
        print("[HW] disable called")

    def emergency_stop(self) -> None:
        if self._arm_real is not None:
            self._arm_real.emergency_stop()
        print("[HW] EMERGENCY STOP called")

    def get_joint_values(self) -> np.ndarray:
        if self._arm_real is not None:
            return np.asarray(self._arm_real.get_joint_values(), dtype=float)
        return np.zeros(6)

    def resync_servo_to_measured(self) -> Optional[np.ndarray]:
        """Pin servo command chain to measured pose (after clutch/home re-engage)."""
        if self._arm_real is None:
            return None
        try:
            q_now = np.asarray(self._arm_real.get_joint_values(), dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(q_now)):
            return None
        self._last_servo_cmd_q = q_now.copy()
        self._diag_prev_cmd_q = q_now.copy()
        self._diag_prev_meas_q = q_now.copy()
        self._diag_prev_cmd_dq = np.zeros_like(q_now)
        if self._supports_servo_j:
            try:
                self._arm_real.servo_j(q_now)
            except Exception:
                pass
        return q_now

    def resync_servo_to_command(self, q_rad: np.ndarray) -> Optional[np.ndarray]:
        """Pin servo command chain to an explicit joint target (hold last IK command)."""
        if self._arm_real is None:
            return None
        q_cmd = np.asarray(q_rad, dtype=float).copy()
        if not np.all(np.isfinite(q_cmd)):
            return None
        self._last_servo_cmd_q = q_cmd.copy()
        self._diag_prev_cmd_q = q_cmd.copy()
        self._diag_prev_cmd_dq = np.zeros_like(q_cmd)
        try:
            q_meas = np.asarray(self._arm_real.get_joint_values(), dtype=float)
            if q_meas.shape == q_cmd.shape:
                self._diag_prev_meas_q = q_meas.copy()
        except Exception:
            pass
        if self._supports_servo_j:
            try:
                self._arm_real.servo_j(q_cmd)
            except Exception:
                pass
        return q_cmd

    def feed_idle_servo_hold(self, q_hold: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Stream hold pose during idle. Default: last command; fallback: measured."""
        if q_hold is not None:
            return self.resync_servo_to_command(q_hold)
        if self._last_servo_cmd_q is not None:
            return self.resync_servo_to_command(self._last_servo_cmd_q)
        return self.resync_servo_to_measured()

    def _update_servo_diag(self, q_sent: np.ndarray, dt_s: float) -> None:
        if (not self._servo_diag_enabled) or self._arm_real is None:
            return
        try:
            q_meas = np.asarray(self._arm_real.get_joint_values(), dtype=float)
        except Exception:
            return
        if q_meas.shape != q_sent.shape:
            return

        dt_safe = max(float(dt_s), 1e-4)
        if self._diag_prev_cmd_q is None:
            self._diag_prev_cmd_q = q_sent.copy()
            self._diag_prev_meas_q = q_meas.copy()
            self._diag_prev_cmd_dq = np.zeros_like(q_sent)
            return

        cmd_dq = q_sent - self._diag_prev_cmd_q
        cmd_step = float(np.max(np.abs(cmd_dq)))
        cmd_vel = cmd_step / dt_safe
        if self._diag_prev_meas_q is not None and self._diag_prev_meas_q.shape == q_meas.shape:
            meas_dq = q_meas - self._diag_prev_meas_q
            meas_step = float(np.max(np.abs(meas_dq)))
        else:
            meas_dq = np.zeros_like(q_meas)
            meas_step = 0.0
        track_err = np.abs(q_sent - q_meas)
        track_err_max = float(np.max(track_err))
        worst_idx = int(np.argmax(track_err))
        if track_err_max > self._diag_worst_joint_err:
            self._diag_worst_joint_err = track_err_max
            self._diag_worst_joint_idx = worst_idx

        chatter_thr = np.deg2rad(0.03)
        if (
            self._diag_prev_cmd_dq is not None
            and self._diag_prev_cmd_dq.shape == cmd_dq.shape
            and np.any((self._diag_prev_cmd_dq * cmd_dq) < 0.0)
            and (
                float(np.max(np.abs(self._diag_prev_cmd_dq))) > chatter_thr
                or float(np.max(np.abs(cmd_dq))) > chatter_thr
            )
        ):
            self._diag_chatter_count += 1

        self._diag_count += 1
        self._diag_cmd_step_sum += cmd_step
        self._diag_cmd_step_max = max(self._diag_cmd_step_max, cmd_step)
        self._diag_cmd_vel_sum += cmd_vel
        self._diag_cmd_vel_max = max(self._diag_cmd_vel_max, cmd_vel)
        self._diag_meas_step_sum += meas_step
        self._diag_meas_step_max = max(self._diag_meas_step_max, meas_step)
        self._diag_track_err_sum += track_err_max
        self._diag_track_err_max = max(self._diag_track_err_max, track_err_max)

        self._diag_prev_cmd_q = q_sent.copy()
        self._diag_prev_meas_q = q_meas.copy()
        self._diag_prev_cmd_dq = cmd_dq.copy()

    def pop_servo_diag(self) -> Optional[dict]:
        if self._diag_count <= 0:
            return None
        reject_delta = self._servo_reject_count - self._diag_last_reject_count
        self._diag_last_reject_count = self._servo_reject_count
        data = {
            "count": self._diag_count,
            "cmd_step_avg": self._diag_cmd_step_sum / self._diag_count,
            "cmd_step_max": self._diag_cmd_step_max,
            "cmd_vel_avg": self._diag_cmd_vel_sum / self._diag_count,
            "cmd_vel_max": self._diag_cmd_vel_max,
            "meas_step_avg": self._diag_meas_step_sum / self._diag_count,
            "meas_step_max": self._diag_meas_step_max,
            "track_err_avg": self._diag_track_err_sum / self._diag_count,
            "track_err_max": self._diag_track_err_max,
            "chatter_count": self._diag_chatter_count,
            "reject_delta": reject_delta,
            "worst_joint_idx": self._diag_worst_joint_idx,
            "worst_joint_err": self._diag_worst_joint_err,
        }
        self._diag_count = 0
        self._diag_cmd_step_sum = 0.0
        self._diag_cmd_step_max = 0.0
        self._diag_cmd_vel_sum = 0.0
        self._diag_cmd_vel_max = 0.0
        self._diag_meas_step_sum = 0.0
        self._diag_meas_step_max = 0.0
        self._diag_track_err_sum = 0.0
        self._diag_track_err_max = 0.0
        self._diag_chatter_count = 0
        self._diag_worst_joint_idx = -1
        self._diag_worst_joint_err = 0.0
        return data

    def send_joint_positions(
        self,
        q_rad: np.ndarray,
        dt_s: float,
        *,
        cmd_max_step_rad: Optional[float] = None,
        skip_cmd_lp: bool = False,
    ) -> bool:
        """Return True if command was accepted (or hold-streamed), False if servo_j rejected."""
        if self._arm_real is not None:
            if self._supports_servo_j:
                q_cmd = np.asarray(q_rad, dtype=float).copy()
                now_t = time.time()
                step_limit_rad = self._servo_cmd_max_step_rad
                if cmd_max_step_rad is not None and cmd_max_step_rad > 0.0:
                    step_limit_rad = min(step_limit_rad, float(cmd_max_step_rad))

                # Reject storm recovery: briefly hold at measured pose so lag can shrink.
                if now_t < self._servo_hold_until_t:
                    try:
                        q_now = np.asarray(self._arm_real.get_joint_values(), dtype=float)
                        if q_now.shape == q_cmd.shape:
                            self._arm_real.servo_j(q_now)
                            self._last_servo_cmd_q = q_now
                            self._update_servo_diag(q_now, dt_s)
                            return True
                    except Exception:
                        pass

                # Command-side limiter uses last sent command, so upper-layer jumps do not
                # directly hit firmware step clamp.
                if self._last_servo_cmd_q is not None and self._last_servo_cmd_q.shape == q_cmd.shape:
                    q_cmd = limit_joint_step(self._last_servo_cmd_q, q_cmd, step_limit_rad)

                # Optional first-order low-pass to reduce high-frequency IK noise.
                if (
                    not skip_cmd_lp
                    and self._servo_cmd_lp_alpha < 1.0
                    and self._last_servo_cmd_q is not None
                    and self._last_servo_cmd_q.shape == q_cmd.shape
                ):
                    alpha = self._servo_cmd_lp_alpha
                    q_cmd = self._last_servo_cmd_q + alpha * (q_cmd - self._last_servo_cmd_q)

                ok = bool(self._arm_real.servo_j(q_cmd))
                if not ok:
                    self._servo_reject_count += 1
                    if self._servo_reject_hold_command and self._last_servo_cmd_q is not None:
                        try:
                            self._arm_real.servo_j(self._last_servo_cmd_q)
                            self._update_servo_diag(self._last_servo_cmd_q, dt_s)
                        except Exception:
                            pass
                    elif not self._servo_reject_hold_command:
                        try:
                            q_now = np.asarray(self._arm_real.get_joint_values(), dtype=float)
                            if q_now.shape == q_cmd.shape:
                                self._arm_real.servo_j(q_now)
                                self._last_servo_cmd_q = q_now
                                self._update_servo_diag(q_now, dt_s)
                        except Exception:
                            pass
                    self._servo_hold_until_t = now_t + self._servo_recover_hold_s
                    now_t = time.time()
                    if now_t - self._last_servo_warn_t >= 1.0:
                        hold_msg = (
                            "hold last command; keep servo mode."
                            if self._servo_reject_hold_command
                            else "resync with measured pose; keep servo mode."
                        )
                        print(
                            f"[HW][SERVO] frame rejected x{self._servo_reject_count}, "
                            f"{hold_msg}"
                        )
                        self._last_servo_warn_t = now_t
                    return False
                self._last_servo_cmd_q = q_cmd
                self._update_servo_diag(q_cmd, dt_s)
                return True
            self._arm_real.move_j(
                joint_angles=q_rad,
                is_radians=True,
                speed=self._joint_speed,
                block=False,
            )
            return True
        return False

    def is_servo_streaming(self) -> bool:
        return bool(self._arm_real is not None and self._supports_servo_j)

    def _start_gripper_worker(self, name: str, fn) -> bool:
        with self._gripper_op_lock:
            if self._gripper_worker is not None and self._gripper_worker.is_alive():
                print(f"[HW][GRIPPER] {name} ignored: previous op still running")
                return False
            t = threading.Thread(target=fn, name=f"gripper-{name}", daemon=True)
            self._gripper_worker = t
            t.start()
            return True

    def _join_gripper_worker(self, timeout_s: float = 0.2) -> None:
        t = None
        with self._gripper_op_lock:
            if self._gripper_worker is not None and self._gripper_worker.is_alive():
                t = self._gripper_worker
        if t is not None:
            t.join(timeout=max(float(timeout_s), 0.0))

    def open_gripper(self) -> None:
        if not self._hw_has_gripper:
            return
        if self._arm_real is not None:
            def _do_open() -> None:
                try:
                    kwargs = {"block": False}
                    if self._gripper_effort is not None:
                        kwargs["effort"] = self._gripper_effort
                    self._arm_real.open_gripper(**kwargs)
                except Exception as exc:
                    print(f"[HW][WARN] gripper open failed: {exc}")

            self._start_gripper_worker("open", _do_open)
        if not self._jaw_open:
            self._jaw_open = True
            print("[HW] gripper open")

    def close_gripper(self) -> None:
        if not self._hw_has_gripper:
            return
        if self._arm_real is not None:
            def _do_close() -> None:
                # 优先使用 fafu 的力感知 grasp()，避免“闭合到底硬顶”。
                if self._gripper_grasp_force is not None and hasattr(self._arm_real, "grasp"):
                    try:
                        result = self._arm_real.grasp(
                            force_threshold=self._gripper_grasp_force,
                            effort=self._gripper_effort,
                            vel=self._gripper_grasp_vel,
                            timeout=self._gripper_grasp_timeout,
                        )
                        print(
                            f"[HW] gripper grasp: grasped={getattr(result, 'grasped', False)} "
                            f"reason={getattr(result, 'reason', 'unknown')} "
                            f"peak_torque={getattr(result, 'peak_torque_raw', 'n/a')}"
                        )
                        return
                    except Exception as exc:
                        print(f"[HW][WARN] grasp failed, fallback close_gripper: {exc}")
                try:
                    kwargs = {"block": False}
                    if self._gripper_effort is not None:
                        kwargs["effort"] = self._gripper_effort
                    self._arm_real.close_gripper(**kwargs)
                except Exception as exc:
                    print(f"[HW][WARN] gripper close failed: {exc}")

            self._start_gripper_worker("close", _do_close)
        if self._jaw_open:
            self._jaw_open = False
            print("[HW] gripper close")

    def get_joint_limit_margin_deg(self, q_rad: Optional[np.ndarray] = None) -> Optional[dict]:
        """Return per-joint distance to lower/upper soft limits in degrees."""
        if self._arm_real is None:
            return None
        if not hasattr(self._arm_real, "get_limit") or not hasattr(self._arm_real, "joint_motor_ids"):
            return None
        try:
            joint_ids = list(getattr(self._arm_real, "joint_motor_ids"))
        except Exception:
            return None
        if not joint_ids:
            return None
        if q_rad is None:
            try:
                q_rad = np.asarray(self._arm_real.get_joint_values(), dtype=float)
            except Exception:
                return None
        q_rad = np.asarray(q_rad, dtype=float)
        if q_rad.size != len(joint_ids):
            return None

        margins = []
        min_margin_deg = float("inf")
        min_margin_joint = -1
        for i, mid in enumerate(joint_ids):
            lim = self._arm_real.get_limit(int(mid), is_radians=True)
            if lim is None:
                continue
            lo, hi = float(lim[0]), float(lim[1])
            cur = float(q_rad[i])
            d_lo = cur - lo
            d_hi = hi - cur
            near_deg = math.degrees(min(d_lo, d_hi))
            if near_deg < min_margin_deg:
                min_margin_deg = near_deg
                min_margin_joint = i + 1
            margins.append((i + 1, math.degrees(d_lo), math.degrees(d_hi), near_deg))

        if not margins:
            return None
        return {
            "margins": margins,
            "min_margin_deg": min_margin_deg,
            "min_margin_joint": min_margin_joint,
        }

    def q_rad_within_limits(self, q_rad: np.ndarray, min_margin_deg: float = 0.5) -> bool:
        """Check whether every joint stays inside hardware soft limits."""
        info = self.get_joint_limit_margin_deg(q_rad)
        if info is None:
            return True
        return float(info["min_margin_deg"]) >= float(min_margin_deg)


def _xr_touch_heartbeat() -> bool:
    """SDK 时间戳可读即视为链路存活（用于超时，不要求时间戳递增）。"""
    global _XR_LINK_READY, _LAST_XR_POLL_WALL_TS
    if _XRT is None:
        return False
    timestamp_ns = _xr_safe_call(_XRT.get_time_stamp_ns, None)
    if timestamp_ns is None or int(timestamp_ns) <= 0:
        return False
    _XR_LINK_READY = True
    _LAST_XR_POLL_WALL_TS = time.time()
    return True


def get_xr_state() -> Optional[XRState]:
    """
    TODO: 对接 XRoboToolkit 数据流。

    返回约定：
    - SDK 可读：返回 XRState
    - SDK 不可读：返回 None

    坐标约定：
    - controller_pos_vr / controller_quat_vr 均在 VR 世界坐标系下。
    """
    global _LAST_XR_TIMESTAMP_NS, _XR_LINK_READY, _LAST_XR_POLL_WALL_TS

    # 优先走离线回放源
    if _REPLAY_READER is not None:
        return _REPLAY_READER.poll()

    # 实时模式：直接读取 XRoboToolkit SDK
    if _XRT is None:
        return None

    timestamp_ns = _xr_safe_call(_XRT.get_time_stamp_ns, None)
    if timestamp_ns is None:
        return None
    timestamp_ns = int(timestamp_ns)
    if timestamp_ns <= 0:
        return None
    _LAST_XR_TIMESTAMP_NS = timestamp_ns
    _XR_LINK_READY = True
    _LAST_XR_POLL_WALL_TS = time.time()
    timestamp_s = float(timestamp_ns) / 1e9

    left_pose = _as_pose7(_xr_safe_call(_XRT.get_left_controller_pose, None))
    right_pose = _as_pose7(_xr_safe_call(_XRT.get_right_controller_pose, None))

    if _XR_CONTROLLER_SIDE == "right":
        pose = right_pose
        grip_val = float(_xr_safe_call(_XRT.get_right_grip, 0.0))
    else:
        pose = left_pose
        grip_val = float(_xr_safe_call(_XRT.get_left_grip, 0.0))

    return XRState(
        timestamp_s=timestamp_s,
        controller_pos_vr=np.asarray(pose[:3], dtype=float),
        controller_quat_vr=np.asarray(pose[3:7], dtype=float),
        grip_pressed=grip_val >= _XR_GRIP_THRESHOLD,
        grip_value=grip_val,
        estop_pressed=_get_estop_pressed(),
        button_a_pressed=bool(_xr_safe_call(_XRT.get_A_button, 0)),
        button_b_pressed=bool(_xr_safe_call(_XRT.get_B_button, 0)),
    )


def _build_side_xr_state(
    *,
    timestamp_s: float,
    pose7: np.ndarray,
    grip_val: float,
    button_a: bool,
    button_b: bool,
    estop_pressed: bool,
    grip_threshold: float,
) -> XRState:
    return XRState(
        timestamp_s=timestamp_s,
        controller_pos_vr=np.asarray(pose7[:3], dtype=float),
        controller_quat_vr=np.asarray(pose7[3:7], dtype=float),
        grip_pressed=grip_val >= grip_threshold,
        grip_value=grip_val,
        estop_pressed=estop_pressed,
        button_a_pressed=bool(button_a),
        button_b_pressed=bool(button_b),
    )


def get_xr_dual_states(grip_threshold: float) -> Optional[Tuple[XRState, XRState, bool]]:
    """Poll both controllers. Returns (right, left, estop_pressed) or None."""
    global _LAST_XR_TIMESTAMP_NS, _XR_LINK_READY, _LAST_XR_POLL_WALL_TS

    if _REPLAY_READER is not None or _XRT is None:
        return None

    timestamp_ns = _xr_safe_call(_XRT.get_time_stamp_ns, None)
    if timestamp_ns is None:
        return None
    timestamp_ns = int(timestamp_ns)
    if timestamp_ns <= 0:
        return None
    _LAST_XR_TIMESTAMP_NS = timestamp_ns
    _XR_LINK_READY = True
    _LAST_XR_POLL_WALL_TS = time.time()
    timestamp_s = float(timestamp_ns) / 1e9
    estop_pressed = _get_estop_pressed()

    right_pose = _as_pose7(_xr_safe_call(_XRT.get_right_controller_pose, None))
    left_pose = _as_pose7(_xr_safe_call(_XRT.get_left_controller_pose, None))
    right = _build_side_xr_state(
        timestamp_s=timestamp_s,
        pose7=right_pose,
        grip_val=float(_xr_safe_call(_XRT.get_right_grip, 0.0)),
        button_a=bool(_xr_safe_call(_XRT.get_A_button, 0)),
        button_b=bool(_xr_safe_call(_XRT.get_B_button, 0)),
        estop_pressed=estop_pressed,
        grip_threshold=grip_threshold,
    )
    left = _build_side_xr_state(
        timestamp_s=timestamp_s,
        pose7=left_pose,
        grip_val=float(_xr_safe_call(_XRT.get_left_grip, 0.0)),
        button_a=bool(_xr_safe_call(_XRT.get_X_button, 0)),
        button_b=bool(_xr_safe_call(_XRT.get_Y_button, 0)),
        estop_pressed=estop_pressed,
        grip_threshold=grip_threshold,
    )
    return right, left, estop_pressed


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。

    常用启动：
    - 调试：--dry-run --visualize
    - 上真机：去掉 --dry-run
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "replay"], default=DEFAULT_RUNTIME_MODE,
                        help="Data source mode: live XR stream or offline replay.")
    parser.add_argument("--control-mode", choices=["sim", "real"], default=DEFAULT_CONTROL_MODE,
                        help="Robot control mode: simulation-only or real hardware.")
    parser.add_argument("--hw-cfg", type=str, default="",
                        help="Real mode: path to robot.cfg (empty uses local default).")
    parser.add_argument("--dual-arm", action="store_true", default=False,
                        help="Control two arms: right controller -> --hw-port-right, "
                             "left controller -> --hw-port-left.")
    parser.add_argument("--hw-port-right", type=str, default="COM5",
                        help="Dual-arm mode: serial port for the right-hand controlled arm.")
    parser.add_argument("--hw-port-left", type=str, default="COM6",
                        help="Dual-arm mode: serial port for the left-hand controlled arm.")
    parser.add_argument("--no-gripper-left", action="store_true", default=False,
                        help="Dual-arm mode: skip left-arm gripper (e.g. COM6 M7 offline).")
    parser.add_argument("--no-gripper-right", action="store_true", default=False,
                        help="Dual-arm mode: skip right-arm gripper.")
    parser.add_argument(
        "--left-joint-sign",
        type=float,
        nargs=6,
        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Dual-arm: multiply left-arm hardware joint angles by +/-1 before IK. "
             "Try 1 1 -1 1 1 1 only if J3 direction looks inverted.",
    )
    parser.add_argument(
        "--right-joint-sign",
        type=float,
        nargs=6,
        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Dual-arm: multiply right-arm hardware joint angles by +/-1 before IK.",
    )
    parser.add_argument("--hw-cfg-left", type=str, default="",
                        help="Dual-arm: optional separate robot.cfg for left arm "
                             "(default: same as --hw-cfg / robot.cfg).")
    parser.add_argument(
        "--left-home-to-zero",
        action="store_true",
        default=False,
        help="Dual-arm: LEFT_TRIGGER homes left arm to joint zeros (default: startup pose, avoids lifting).",
    )
    parser.add_argument(
        "--left-j3-headroom-deg",
        type=float,
        default=35.0,
        help="Dual-arm: target J3 soft-limit margin (deg) for left arm before teleop; "
             "auto-unfold on enable when folded tighter than this.",
    )
    parser.add_argument(
        "--no-left-auto-j3-unfold",
        action="store_true",
        default=False,
        help="Dual-arm: do not auto-unfold left J3 on enable when startup margin is too small.",
    )
    parser.add_argument(
        "--left-full-teleop",
        action="store_true",
        default=False,
        help="Dual-arm: left grip VR follow (default: left X=extend, Y=home preset only).",
    )
    parser.add_argument(
        "--left-preset-raise-deltas-deg",
        type=float,
        nargs=6,
        default=[0.0, 70.0, 70.0, 0.0, 0.0, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Dual-arm preset left: phase-1 joint deltas from home (deg). "
             "Default J2+28 J3+26 = 抬肘.",
    )
    parser.add_argument(
        "--left-preset-extend-deltas-deg",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Dual-arm preset left: phase-2 joint deltas added after raise (deg). "
             "Default J4+28 = 腕部上抬（J4 负向会下垂，请用正增量）.",
    )
    parser.add_argument(
        "--left-preset-auto-calibrate",
        action="store_true",
        default=False,
        help="Dual-arm preset left: enable FK auto-calibration (off by default; "
             "sim FK often mismatches left-arm mounting).",
    )
    parser.add_argument(
        "--left-preset-forward-axis",
        type=str,
        default="neg_y",
        choices=("neg_y", "-y", "pos_x", "+x", "neg_x", "-x", "pos_y", "+y"),
        help="(legacy) only used if auto-calibrate is reworked; preset defaults to joint deltas.",
    )
    parser.add_argument(
        "--no-left-preset-auto-calibrate",
        action="store_true",
        default=False,
        help="Dual-arm preset left: force-disable FK auto-calibration (default is already off).",
    )
    parser.add_argument(
        "--left-preset-lift-m",
        type=float,
        default=0.08,
        help="Dual-arm preset left: minimum TCP lift (m) during auto raise calibration.",
    )
    parser.add_argument(
        "--left-preset-reach-m",
        type=float,
        default=0.10,
        help="Dual-arm preset left: minimum TCP horizontal reach (m) during auto extend calibration.",
    )
    parser.add_argument(
        "--left-preset-step-deg",
        type=float,
        default=0.5,
        help="Dual-arm preset left: joint step per cycle (deg) for slow extend/home.",
    )
    parser.add_argument("--hw-gripper-id", type=int, default=7,
                        help="Real mode: gripper motor id in robot.cfg.")
    parser.add_argument("--hw-joint-speed", type=int, default=15,
                        help="Real mode: move_j speed percentage (1-100).")
    parser.add_argument("--servo-watchdog-ms", type=int, default=350,
                        help="servo_j watchdog in ms (larger tolerates loop jitter).")
    parser.add_argument("--servo-max-vel", type=float, default=3.0,
                        help="servo_j max velocity in rad/s.")
    parser.add_argument("--servo-max-step-deg", type=float, default=8.0,
                        help="servo_j firmware max step per cycle in deg.")
    parser.add_argument("--servo-max-lag-deg", type=float, default=35.0,
                        help="servo_j firmware max tracking lag in deg.")
    parser.add_argument("--servo-cmd-max-step-deg", type=float, default=2.0,
                        help="Upper-layer command limiter per cycle in deg (should be <= servo-max-step-deg).")
    parser.add_argument("--servo-cmd-lp-alpha", type=float, default=0.25,
                        help="Upper-layer joint low-pass alpha in [0,1], smaller is smoother.")
    parser.add_argument("--servo-recover-hold-ms", type=float, default=150.0,
                        help="After servo reject, hold measured pose for this long to recover lag.")
    parser.add_argument("--servo-rate-hz", type=float, default=100.0,
                        help="Nominal servo_j rate for feedforward/lookahead dt.")
    parser.add_argument("--servo-lookahead-s", type=float, default=0.05,
                        help="Servo lookahead time constant in seconds.")
    parser.add_argument("--servo-feedforward-vel", action="store_true", default=True,
                        help="Enable servo velocity feedforward in fafu servo_j.")
    parser.add_argument("--no-servo-feedforward-vel", action="store_false", dest="servo_feedforward_vel",
                        help="Disable servo velocity feedforward in fafu servo_j.")
    parser.add_argument("--servo-diag", action="store_true", default=True,
                        help="Enable periodic servo command/measurement diagnostics.")
    parser.add_argument("--no-servo-diag", action="store_false", dest="servo_diag",
                        help="Disable periodic servo diagnostics.")
    parser.add_argument("--loop-hz", type=float, default=100.0)
    parser.add_argument(
        "--timeout-ms",
        type=float,
        default=2000.0,
        help="XR link loss timeout (ms). Triggers when SDK timestamp cannot be read.",
    )
    parser.add_argument("--max-lin-speed", type=float, default=0.08, help="m/s")
    parser.add_argument("--max-ang-speed-deg", type=float, default=80.0, help="deg/s")
    parser.add_argument("--rot-scale", type=float, default=0.1,
                        help="Scale VR relative rotation before IK (0 disables rotation, 1 keeps full).")
    parser.add_argument("--rot-axis-gain", type=float, nargs=3, default=[1.0, 1.6, 1.0],
                        metavar=("GX", "GY", "GZ"),
                        help="Per-axis gain for relative rotation vector in base frame (X/Y/Z).")
    parser.add_argument("--max-joint-step-deg", type=float, default=1.0,
                        help="Max per-cycle joint delta for each joint (deg, <=0 disables).")
    parser.add_argument("--home-joint-step-deg", type=float, default=0.5,
                        help="Max per-cycle joint delta while home hold is active (deg, <=0 disables). "
                             "At 100Hz, 0.75deg ~= 75deg/s per joint. Home bypasses servo cmd LP.")
    parser.add_argument("--vr-yaw-offset-deg", type=float, default=0.0,
                        help="Additional yaw offset for VR->robot mapping (use 180 only if X/Y are mirrored).")
    parser.add_argument("--pos-scale", type=float, default=0.55,
                        help="VR displacement scale (hand move 1m -> robot ~pos_scale m).")
    parser.add_argument("--dry-run", action="store_true", help="do not send real HW command")
    parser.add_argument("--visualize", action="store_true",
                        help="enable Panda3D visualization window")
    # 默认关闭可视化，降低真机模式下的计算与渲染开销。
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="disable Panda3D visualization window")
    parser.set_defaults(visualize=False)
    parser.add_argument("--replay-jsonl", type=str, default="", help="Offline replay jsonl file path.")
    parser.add_argument("--replay-filename", type=str, default=DEFAULT_REPLAY_FILENAME,
                        help="Offline replay filename under <this_script_dir>/logs. Empty means live XR mode.")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Replay speed multiplier.")
    parser.add_argument("--xr-controller-side", choices=["left", "right"], default="right",
                        help="Live XR mode: use left or right controller.")
    parser.add_argument("--xr-grip-threshold", type=float, default=0.5,
                        help="Live XR mode: grip value >= threshold engages clutch follow.")
    parser.add_argument("--xr-grip-release-threshold", type=float, default=0.40,
                        help="Live XR mode: while clutch is active, release when grip value "
                             "drops below this (hysteresis; must be < --xr-grip-threshold).")
    parser.add_argument("--xr-estop-button", choices=["A", "B", "X", "Y", "RIGHT_AXIS_CLICK", "LEFT_AXIS_CLICK", "RIGHT_TRIGGER", "LEFT_TRIGGER"],
                        default="LEFT_AXIS_CLICK", help="Live XR mode: which button triggers software estop.")
    parser.add_argument("--xr-home-button", choices=["A", "B", "X", "Y", "RIGHT_AXIS_CLICK", "LEFT_AXIS_CLICK", "RIGHT_TRIGGER", "LEFT_TRIGGER"],
                        default="RIGHT_TRIGGER",
                        help="Live XR mode: hold to move robot toward home pose; release to stop immediately.")
    parser.add_argument("--jaw-open-width", type=float, default=-1.0,
                        help="Gripper open width in meters (<0 means use gripper jaw max).")
    parser.add_argument("--jaw-close-width", type=float, default=0.0,
                        help="Gripper close width in meters.")
    parser.add_argument("--gripper-effort", type=int, default=100,
                          help="Real mod"
                             ""
                             ""
                             ""
                             "e: gripper effort cap (raw int16). None keeps controller default.")
    parser.add_argument("--gripper-grasp-force", type=int, default=50,
                        help="Real mode: use force-aware grasp() on close with this force threshold; "
                             "set <0 to disable and fallback to close_gripper.")
    parser.add_argument("--gripper-grasp-vel", type=float, default=0.15,
                        help="Real mode: grasp() closing velocity in turns/s.")
    parser.add_argument("--gripper-grasp-timeout", type=float, default=1.5,
                        help="Real mode: grasp() timeout in seconds.")
    parser.add_argument("--invert-gripper-buttons", action="store_true", default=False,
                        help="Swap A/B gripper actions when hardware open/close direction is opposite.")
    parser.add_argument("--position-only", action="store_true",
                        help="Track position only (keep orientation at clutch anchor) to improve IK robustness.")
    parser.add_argument("--full-pose", action="store_false", dest="position_only",
                        help="Track both position and orientation (harder IK, for advanced tuning).")
    parser.set_defaults(position_only=True)
    parser.add_argument("--max-delta-pos", type=float, default=0.7,
                        help="Clamp displacement from clutch anchor in meters (<=0 disables).")
    parser.add_argument("--vis-update-div", type=int, default=3,
                        help="Visualization refresh divisor (>=1). Lower is smoother but heavier.")
    parser.add_argument("--trail-step", type=int, default=0,
                        help="Add one TCP trail point every N accepted IK frames (<=0 disables trail).")
    parser.add_argument("--trail-radius", type=float, default=0.006, help="TCP trail sphere radius (m).")
    parser.add_argument("--ik-retry", action="store_true", default=True,
                        help="Enable IK fallback retries with smaller Cartesian steps.")
    parser.add_argument("--no-ik-retry", action="store_false", dest="ik_retry",
                        help="Disable IK fallback retries.")
    parser.add_argument("--ik-fail-warn-threshold", type=int, default=20,
                        help="Print diagnostics when consecutive IK failures reach this number.")
    parser.add_argument("--ik-max-joint-jump-deg", type=float, default=26.0,
                        help="Reject IK solutions whose max joint delta vs current q_seed exceeds "
                             "this value (deg). Prevents wrist-singularity flips that cause "
                             "servo_j reject storms. Set <=0 to disable.")
    parser.add_argument("--ik-adaptive-ease", action="store_true", default=True,
                        help="On consecutive IK failures, temporarily widen jump threshold and "
                             "reduce rotation scale to reduce stutter.")
    parser.add_argument("--no-ik-adaptive-ease", action="store_false", dest="ik_adaptive_ease",
                        help="Disable adaptive IK easing.")
    parser.add_argument("--ik-adapt-fail-streak", type=int, default=3,
                        help="Consecutive IK failures before adaptive ease kicks in.")
    parser.add_argument("--ik-adapt-jump-boost-deg", type=float, default=0.0,
                        help="Extra jump threshold (deg) added per adaptive ease level "
                             "(0=only shrink rotation on fail; >0 can cause erratic jumps).")
    parser.add_argument("--clutch-min-margin-deg", type=float, default=8.0,
                        help="Block clutch engage when any joint soft-limit margin is below this (deg).")
    parser.add_argument("--clutch-reengage-cooldown-s", type=float, default=0.35,
                        help="Min seconds after clutch release before re-engage (reduces anchor-reset jerks).")
    parser.add_argument("--grip-engage-stable-frames", type=int, default=8,
                        help="Grip >= engage threshold for N frames before clutch starts.")
    parser.add_argument("--grip-release-stable-frames", type=int, default=4,
                        help="Grip < release threshold for N frames before clutch releases.")
    parser.add_argument("--clutch-engage-snap-max-mm", type=float, default=35.0,
                        help="On clutch engage, if measured TCP is within this distance (mm) of "
                             "the last commanded TCP, keep the command anchor instead of snapping "
                             "to measured FK (avoids gravity-sag droop on re-engage). "
                             "Set <=0 to always snap to measured FK.")
    parser.add_argument("--clutch-engage-grace-frames", type=int, default=10,
                        help="After clutch engage, ignore VR motion for N control frames "
                             "(reduces grip-squeeze twitch).")
    parser.add_argument("--home-stop-margin-deg", type=float, default=35.0,
                        help="Soft-limit floor (deg) when clamping home target joints.")
    parser.add_argument("--home-arrive-deg", type=float, default=3.0,
                        help="Stop homing when max joint error to home target is below this (deg).")
    parser.add_argument(
        "--right-home-joints-deg",
        type=float,
        nargs=6,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="RIGHT arm home joints (deg). Default: all zeros.",
    )
    parser.add_argument(
        "--right-home-ready-pose",
        action="store_true",
        default=False,
        help="RIGHT home = teleop ready pose (J2~45deg) instead of joint zeros; "
             "reduces J2 soft-limit edge jitter.",
    )
    parser.add_argument("--servo-reject-hold-command", action="store_true", default=True,
                        help="On servo reject, hold last command instead of snapping to measured.")
    parser.add_argument("--no-servo-reject-hold-command", action="store_false",
                        dest="servo_reject_hold_command",
                        help="Legacy: snap to measured on servo reject.")
    parser.add_argument("--ik-adapt-rot-scale-factor", type=float, default=0.65,
                        help="Rotation scale multiplier per adaptive ease level (<1 shrinks).")
    parser.add_argument("--ik-adapt-rot-scale-min", type=float, default=0.25,
                        help="Minimum effective rotation scale under adaptive ease.")
    parser.add_argument("--ik-adapt-recover-streak", type=int, default=10,
                        help="Consecutive IK successes to restore base jump/rot settings.")
    parser.add_argument("--ik-adapt-j6-margin-deg", type=float, default=8.0,
                        help="When J6 is within this margin (deg) to limit, scale down wrist rotation.")
    parser.add_argument("--self-collision-filter", action="store_true", default=True,
                        help="Reject IK solutions that are self-collided.")
    parser.add_argument("--no-self-collision-filter", action="store_false", dest="self_collision_filter",
                        help="Disable self-collision filtering for IK solutions.")
    parser.add_argument("--soft-collision-start", action="store_true", default=True,
                        help="If init pose is collided, start in soft mode and switch to strict after recovery.")
    parser.add_argument("--no-soft-collision-start", action="store_false", dest="soft_collision_start",
                        help="Disable soft-start behavior for collision filtering.")
    parser.add_argument("--ik-max-attempts-per-cycle", type=int, default=8,
                        help="Maximum IK candidate attempts per control cycle (<=0 means unlimited).")
    parser.add_argument("--ik-allow-flip-candidate", action="store_true", default=False,
                        help="Allow 180deg wrist-flip orientation candidates during full-pose IK retry. "
                             "Can fix rare unreachable targets but often causes jitter.")
    parser.add_argument("--tcp-log-interval", type=float, default=0.2,
                        help="Seconds between TCP diag logs.")
    parser.add_argument("--tcp-pos-log-interval", type=float, default=1.0,
                        help="Seconds between accepted TCP position logs.")
    parser.add_argument("--collision-log-interval", type=float, default=1.0,
                        help="Seconds between collision status logs (<=0 disables periodic logs).")
    parser.add_argument("--joint-limit-log-interval", type=float, default=1.0,
                        help="Seconds between joint-limit margin logs in real mode (<=0 disables).")
    parser.add_argument("--workspace-clamp", action="store_true", default=False,
                        help="Clamp TCP target into a predefined workspace box.")
    parser.add_argument("--no-workspace-clamp", action="store_false", dest="workspace_clamp",
                        help="Disable workspace clamping.")
    parser.add_argument("--ws-x-min", type=float, default=0.10)
    parser.add_argument("--ws-x-max", type=float, default=0.45)
    parser.add_argument("--ws-y-min", type=float, default=-0.30)
    parser.add_argument("--ws-y-max", type=float, default=0.30)
    parser.add_argument("--ws-z-min", type=float, default=0.02)
    parser.add_argument("--ws-z-max", type=float, default=0.45)
    return parser


def main() -> None:
    global _REPLAY_READER, _XRT, _XR_CONTROLLER_SIDE, _XR_GRIP_THRESHOLD, _XR_ESTOP_BUTTON, _XR_HOME_BUTTON
    global _LAST_XR_TIMESTAMP_NS, _XR_LINK_READY, _LAST_XR_POLL_WALL_TS
    args = build_parser().parse_args()
    if args.dual_arm:
        if args.mode != "live":
            raise ValueError("--dual-arm requires live XR mode (--mode live).")
        from teleop_dual_arm import run_dual_arm_teleop

        run_dual_arm_teleop(args)
        return
    dt = 1.0 / args.loop_hz
    max_lin_step = args.max_lin_speed * dt
    max_ang_step = math.radians(args.max_ang_speed_deg) * dt
    max_joint_step_rad = math.radians(max(float(args.max_joint_step_deg), 0.0))
    home_joint_step_rad = math.radians(max(float(args.home_joint_step_deg), 0.0))
    servo_max_step_rad = math.radians(max(float(args.servo_max_step_deg), 0.0))
    servo_max_lag_rad = math.radians(max(float(args.servo_max_lag_deg), 0.0))
    servo_cmd_max_step_rad = math.radians(max(float(args.servo_cmd_max_step_deg), 0.0))
    timeout_s = args.timeout_ms / 1000.0

    replay_path = args.replay_jsonl.strip()
    replay_filename = args.replay_filename.strip()
    if args.mode == "replay":
        if replay_filename:
            replay_path = str(_resolve_replay_filename_in_logs(replay_filename))
            print(f"[REPLAY] use filename in logs: {replay_path}")
        if not replay_path:
            raise ValueError(
                "Replay mode requires replay file. "
                "Please provide --replay-jsonl <path> or --replay-filename <name>."
            )
        _REPLAY_READER = JsonlReplayReader(replay_path, replay_speed=args.replay_speed)
        print(f"[REPLAY] enabled, speed={args.replay_speed}x")
    else:
        if replay_path or replay_filename:
            print("[XR] mode=live: ignore replay file arguments.")
        # 实时模式：初始化 XRoboToolkit SDK
        try:
            import xrobotoolkit_sdk as xrt
        except Exception as exc:
            raise RuntimeError(
                "Live XR mode requires xrobotoolkit_sdk. "
                "Please run in XRoboToolkit Python environment or use --replay-jsonl."
            ) from exc
        _XRT = xrt
        _XR_CONTROLLER_SIDE = args.xr_controller_side
        _XR_GRIP_THRESHOLD = float(args.xr_grip_threshold)
        grip_release_thr = float(args.xr_grip_release_threshold)
        if grip_release_thr >= _XR_GRIP_THRESHOLD:
            grip_release_thr = max(_XR_GRIP_THRESHOLD - 0.08, 0.0)
            print(
                f"[XR][WARN] grip release threshold >= engage threshold; "
                f"auto clamp release -> {grip_release_thr:.2f}"
            )
        args.xr_grip_release_threshold = grip_release_thr
        _XR_ESTOP_BUTTON = args.xr_estop_button
        _XR_HOME_BUTTON = args.xr_home_button
        if _XR_ESTOP_BUTTON.upper() == _XR_HOME_BUTTON.upper():
            fallback_btn = "LEFT_AXIS_CLICK" if _XR_HOME_BUTTON.upper() != "LEFT_AXIS_CLICK" else "RIGHT_AXIS_CLICK"
            print(
                f"[XR][WARN] estop button equals home button ({_XR_HOME_BUTTON}), "
                f"auto remap estop -> {fallback_btn}"
            )
            _XR_ESTOP_BUTTON = fallback_btn
        _LAST_XR_TIMESTAMP_NS = None
        _XR_LINK_READY = False
        _LAST_XR_POLL_WALL_TS = 0.0
        _XRT.init()
        print(
            f"[XR] live mode enabled, side={_XR_CONTROLLER_SIDE}, "
            f"grip_engage={_XR_GRIP_THRESHOLD:.2f}, grip_release={float(args.xr_grip_release_threshold):.2f}, "
            f"estop={_XR_ESTOP_BUTTON}, home={_XR_HOME_BUTTON}"
        )

    # ---------------- 坐标标定（必须做） ----------------
    # r_br, t_br 表示：VR 世界坐标 -> 机器人基坐标
    # 直接在文件顶部 CALIB_RPY_DEG_BR / CALIB_T_BR 修改。
    r_br = rotmat_from_euler_xyz_deg(CALIB_RPY_DEG_BR)
    t_br = CALIB_T_BR.copy()
    r_vr_fix = rm.rotmat_from_euler(0.0, 0.0, np.deg2rad(float(args.vr_yaw_offset_deg)))
    print(f"[CALIB] RPY(deg) BR = {CALIB_RPY_DEG_BR}")
    print(f"[CALIB] t_br(m) = {t_br}")
    print(f"[CALIB] vr_yaw_offset(deg) = {float(args.vr_yaw_offset_deg):.1f}")

    # ---------------- 仿真模型（用于 IK） ----------------
    robot_model = PantheraHTSglArm(enable_cc=True)
    q_seed = robot_model.get_jnt_values()
    q_home = resolve_right_home_q(q_seed.shape[0], args)
    robot_model.goto_given_conf(q_seed)
    tcp_pos_anchor, tcp_rot_anchor = robot_model.fk(q_seed)
    print(f"[INIT] q_seed(rad) = {q_seed}")
    print(f"[INIT] tcp_pos_anchor(m) = {tcp_pos_anchor}")
    print(f"[INIT] tcp_rot_anchor =\n{tcp_rot_anchor}")

    # ---------------- 真机/仿真控制器 ----------------
    hw_cfg_path = args.hw_cfg.strip()
    if args.control_mode == "real" and not hw_cfg_path:
        hw_cfg_path = str(Path(__file__).resolve().parent / "fafu_robot_python" / "robot.cfg")
    arm_hw = PantheraArmController(
        control_mode=args.control_mode,
        cfg_path=hw_cfg_path,
        has_gripper=True,
        gripper_motor_id=args.hw_gripper_id,
        joint_speed=args.hw_joint_speed,
        servo_watchdog_ms=args.servo_watchdog_ms,
        servo_max_vel=args.servo_max_vel,
        servo_max_step_rad=servo_max_step_rad,
        servo_max_lag_rad=servo_max_lag_rad,
        servo_cmd_max_step_rad=servo_cmd_max_step_rad,
        servo_cmd_lp_alpha=args.servo_cmd_lp_alpha,
        servo_recover_hold_ms=args.servo_recover_hold_ms,
        servo_reject_hold_command=args.servo_reject_hold_command,
        servo_diag_enabled=args.servo_diag,
        servo_rate_hz=args.servo_rate_hz,
        servo_feedforward_vel=args.servo_feedforward_vel,
        servo_lookahead_time_s=args.servo_lookahead_s,
        gripper_effort=args.gripper_effort,
        gripper_grasp_force=(None if args.gripper_grasp_force is None or args.gripper_grasp_force < 0
                             else args.gripper_grasp_force),
        gripper_grasp_vel=args.gripper_grasp_vel,
        gripper_grasp_timeout=args.gripper_grasp_timeout,
    )
    arm_hw.enable()
    print(
        f"[HOME] right home q (deg) = {np.round(np.rad2deg(q_home), 1)}; "
        f"home_arrive={args.home_arrive_deg:.0f}deg"
    )

    # CRITICAL: sync q_seed with the actual hardware pose right after enable.
    #
    # Without this, q_seed defaults to robot_model.home_conf (typically all
    # zeros), but the real motors after motor_reset are usually NOT at zero
    # (e.g. J4 may be -55deg, J1 +10deg). The mismatch causes:
    #   1) startup: send_joint_positions(q_seed=0) drags every motor toward
    #      zero, generating large servo lag and reject storms.
    #   2) IK seed mismatch: TracIK sees q_seed=0 as the seed but the real
    #      joints are far from there, so IK solutions can flip across wrist
    #      singularity (cmd_step jumps 50-90deg, J5 swings violently).
    # Reading the real pose and pinning the WRS simulation to it eliminates
    # both issues; servo_j then commands "stay where you are" and the user's
    # first VR motion drives a smooth, local IK trajectory.
    if args.control_mode == "real" and not args.dry_run:
        q_seed = sync_q_seed_from_hardware(
            q_seed, robot_model, arm_hw, resync_servo=True, tag="INIT"
        )
        tcp_pos_anchor, tcp_rot_anchor = robot_model.fk(q_seed)
        print(f"[INIT] q_seed synced from hardware (deg) = {np.rad2deg(q_seed)}")
        print(f"[INIT] tcp_pos_anchor synced(m) = {tcp_pos_anchor}")

    # ---------------- 可视化调试（可选） ----------------
    base = None
    robot_vis_model = None
    vis_frame_id = 0
    trail_frame_id = 0
    trail_points = []
    gripper_vis_dirty = False
    if args.visualize:
        base = wd.World(cam_pos=[1.2, 1.2, 0.8], lookat_pos=[0.0, 0.0, 0.2])
        mgm.gen_frame().attach_to(base)
        robot_vis_model = robot_model.gen_meshmodel(alpha=0.8, toggle_tcp_frame=True)
        robot_vis_model.attach_to(base)

    jaw_open_width = None
    jaw_close_width = None
    if getattr(robot_model, "end_effector", None) is not None and hasattr(robot_model.end_effector, "jaw_range"):
        jaw_min = float(robot_model.end_effector.jaw_range[0])
        jaw_max = float(robot_model.end_effector.jaw_range[1])
        jaw_open_width = jaw_max if args.jaw_open_width < 0 else float(np.clip(args.jaw_open_width, jaw_min, jaw_max))
        jaw_close_width = float(np.clip(args.jaw_close_width, jaw_min, jaw_max))
        print(f"[GRIPPER] sim jaw_range=[{jaw_min:.4f}, {jaw_max:.4f}] m")
        print(f"[GRIPPER] button A open={jaw_open_width:.4f} m, button B close={jaw_close_width:.4f} m")
    if args.invert_gripper_buttons:
        print("[GRIPPER] button mapping inverted: A->close, B->open")
    else:
        print("[GRIPPER] button mapping normal: A->open, B->close")
    print(
        f"[GRIPPER] hw effort={args.gripper_effort}, "
        f"grasp_force={args.gripper_grasp_force}, "
        f"grasp_vel={args.gripper_grasp_vel}, "
        f"grasp_timeout={args.gripper_grasp_timeout}"
    )

    # ---------------- 运行时状态 ----------------
    # clutch_active: 是否处于“按住跟随”状态
    # xr_anchor_*:   按下 grip 瞬间的 XR 参考位姿
    # tcp_target_*:  本周期想要到达的 TCP 目标（限幅后）
    # tcp_sent_*:    最近一次已成功求解/下发的 TCP 状态
    clutch_active = False
    grip_engage_thr = float(getattr(args, "xr_grip_threshold", _XR_GRIP_THRESHOLD))
    grip_release_thr = float(
        getattr(args, "xr_grip_release_threshold", max(grip_engage_thr - 0.1, 0.0))
    )
    grip_engage_streak = 0
    grip_release_streak = 0
    clutch_release_t = 0.0
    clutch_cooldown_warned = False
    clutch_block_warned = False
    clutch_engage_grace_remain = 0
    engage_stable_frames = max(int(args.grip_engage_stable_frames), 1)
    release_stable_frames = max(int(args.grip_release_stable_frames), 1)
    xr_anchor_pos = np.zeros(3)
    xr_anchor_rot = np.eye(3)
    tcp_target_pos = tcp_pos_anchor.copy()
    tcp_target_rot = tcp_rot_anchor.copy()
    tcp_sent_pos = tcp_pos_anchor.copy()
    tcp_sent_rot = tcp_rot_anchor.copy()
    ik_try_count = 0
    ik_success_count = 0
    collision_reject_count = 0
    collision_bypass_count = 0
    last_ik_stat_print_t = time.time()
    last_tcp_diag_print_t = 0.0
    last_tcp_pos_log_t = 0.0
    last_collision_log_t = 0.0
    last_joint_limit_log_t = 0.0
    workspace_clamp_active = bool(args.workspace_clamp)
    ik_zero_streak = 0
    collision_filter_strict = bool(args.self_collision_filter)
    ik_jump_reject_count = 0
    ik_limit_reject_count = 0
    tcp_down_stall_warn_count = 0
    ik_jump_thr_rad = math.radians(max(float(args.ik_max_joint_jump_deg), 0.0))
    ik_base_jump_thr_rad = ik_jump_thr_rad
    ik_effective_jump_thr_rad = ik_jump_thr_rad
    ik_effective_rot_scale = float(args.rot_scale)
    ik_adapt_level = 0
    ik_adapt_recover_count = 0
    last_ik_adapt_log_t = 0.0
    last_ik_target_pos: Optional[np.ndarray] = None
    last_ik_target_rot: Optional[np.ndarray] = None
    ik_target_pos_eps_m = 5e-5
    ik_target_rot_eps_rad = math.radians(0.08)
    prev_a_pressed = False
    prev_b_pressed = False
    prev_home_pressed = False
    # ---------------- 性能统计 ----------------
    perf_window_start_t = time.perf_counter()
    perf_loop_count = 0
    perf_loop_sum = 0.0
    perf_loop_max = 0.0
    perf_compute_sum = 0.0
    perf_sleep_req_sum = 0.0
    perf_sleep_actual_sum = 0.0
    perf_overrun_count = 0
    perf_xr_sum = 0.0
    perf_ik_sum = 0.0
    perf_hw_sum = 0.0

    print("Teleop bridge started. Hold grip to follow. Press estop to stop.")
    print("WARNING: verify software estop and hardware estop before real motion.")
    print(f"[LOOP] target_hz={args.loop_hz:.1f}, target_dt={dt * 1000.0:.3f}ms")
    if max_joint_step_rad > 0:
        print(
            f"[JOINT] per-cycle limit={args.max_joint_step_deg:.3f} deg "
            f"({max_joint_step_rad:.5f} rad)"
        )
    else:
        print("[JOINT] per-cycle limit disabled")
    if home_joint_step_rad > 0:
        print(
            f"[HOME] per-cycle limit={args.home_joint_step_deg:.3f} deg "
            f"({home_joint_step_rad:.5f} rad)"
        )
    else:
        print("[HOME] per-cycle limit disabled (instant snap)")
    if args.control_mode == "real":
        print(
            "[SERVO] "
            f"watchdog={int(args.servo_watchdog_ms)}ms, "
            f"max_vel={float(args.servo_max_vel):.2f}rad/s, "
            f"max_step={float(args.servo_max_step_deg):.2f}deg, "
            f"max_lag={float(args.servo_max_lag_deg):.2f}deg, "
            f"cmd_step={float(args.servo_cmd_max_step_deg):.2f}deg, "
            f"lp_alpha={float(args.servo_cmd_lp_alpha):.2f}, "
            f"recover_hold={float(args.servo_recover_hold_ms):.0f}ms, "
            f"rate={float(args.servo_rate_hz):.0f}Hz, "
            f"feedforward={'on' if bool(args.servo_feedforward_vel) else 'off'}, "
            f"lookahead={float(args.servo_lookahead_s) * 1000.0:.0f}ms"
        )
        print(f"[SERVO] diag={'on' if bool(args.servo_diag) else 'off'}")
        if float(args.joint_limit_log_interval) > 0.0:
            print(f"[JOINT] limit-margin log interval={float(args.joint_limit_log_interval):.2f}s")
    if args.self_collision_filter and robot_model.cc is not None:
        init_collided = bool(robot_model.is_collided())
        if init_collided and args.soft_collision_start:
            collision_filter_strict = False
            print("[COLLISION] init pose collided -> soft-start enabled.")
        else:
            collision_filter_strict = True
            print(f"[COLLISION] strict filter enabled (init_collided={init_collided}).")
    print(f"[TELEOP] pos_scale={float(args.pos_scale):.2f}, max_lin_speed={float(args.max_lin_speed):.2f}m/s")
    print(
        f"[ANTI-JUMP] grip_stable={engage_stable_frames}/{release_stable_frames}, "
        f"cooldown={float(args.clutch_reengage_cooldown_s):.2f}s, "
        f"snap_max={float(args.clutch_engage_snap_max_mm):.1f}mm, "
        f"grace_frames={max(int(args.clutch_engage_grace_frames), 0)}"
    )
    if args.position_only:
        print("[IK] position-only mode enabled (TCP orientation locked at clutch anchor).")
    else:
        print("[IK] full-pose mode: VR wrist rotation drives TCP; may look 'pointing down'.")
        print(f"[IK] rotation scale={float(args.rot_scale):.2f}")
        print(f"[IK] rotation axis gain XYZ={np.asarray(args.rot_axis_gain, dtype=float)}")
        print(f"[IK] flip candidate={'on' if args.ik_allow_flip_candidate else 'off'}")
    if args.ik_adaptive_ease:
        print(
            "[IK] adaptive ease enabled: "
            f"fail_streak={int(args.ik_adapt_fail_streak)}, "
            f"jump_boost={float(args.ik_adapt_jump_boost_deg):.1f}deg/level, "
            f"rot_min={float(args.ik_adapt_rot_scale_min):.2f}, "
            f"recover={int(args.ik_adapt_recover_streak)}"
        )
    if workspace_clamp_active:
        print(
            "[IK] workspace clamp enabled: "
            f"x[{args.ws_x_min:.2f},{args.ws_x_max:.2f}] "
            f"y[{args.ws_y_min:.2f},{args.ws_y_max:.2f}] "
            f"z[{args.ws_z_min:.2f},{args.ws_z_max:.2f}]"
        )

    loop_deadline = time.perf_counter()

    def update_ik_adapt_settings(*, reason: str) -> None:
        nonlocal ik_effective_jump_thr_rad, ik_effective_rot_scale, last_ik_adapt_log_t
        if ik_base_jump_thr_rad > 0.0:
            boost_rad = math.radians(
                max(float(args.ik_adapt_jump_boost_deg), 0.0) * ik_adapt_level
            )
            ik_effective_jump_thr_rad = ik_base_jump_thr_rad + boost_rad
        else:
            ik_effective_jump_thr_rad = 0.0
        if ik_adapt_level <= 0:
            ik_effective_rot_scale = float(args.rot_scale)
        else:
            factor = float(np.clip(float(args.ik_adapt_rot_scale_factor), 0.05, 1.0))
            ik_effective_rot_scale = max(
                float(args.rot_scale) * (factor ** ik_adapt_level),
                float(args.ik_adapt_rot_scale_min),
            )
        now_adapt_t = time.time()
        if now_adapt_t - last_ik_adapt_log_t >= 0.5:
            if ik_adapt_level > 0:
                print(
                    f"[IK][ADAPT] {reason}: level={ik_adapt_level}, "
                    f"jump_thr={np.rad2deg(ik_effective_jump_thr_rad):.1f}deg, "
                    f"rot_scale={ik_effective_rot_scale:.2f}"
                )
            elif reason != "init":
                print(
                    f"[IK][ADAPT] {reason}: restored base "
                    f"jump_thr={np.rad2deg(ik_effective_jump_thr_rad):.1f}deg, "
                    f"rot_scale={ik_effective_rot_scale:.2f}"
                )
            last_ik_adapt_log_t = now_adapt_t

    def filter_ik_candidate(q_try: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Validate one raw IK solution and return joint-limited command, or None."""
        nonlocal collision_filter_strict, ik_jump_reject_count, ik_limit_reject_count
        nonlocal collision_reject_count, collision_bypass_count
        if q_try is None:
            return None
        q_try = align_joint_solution_to_seed(np.asarray(q_try, dtype=float), q_seed)
        if ik_effective_jump_thr_rad > 0.0:
            dq_abs = np.abs(q_try - q_seed)
            dq_max = float(np.max(dq_abs))
            if dq_max > ik_effective_jump_thr_rad:
                worst_j_idx = int(np.argmax(dq_abs))
                ik_jump_reject_count += 1
                if ik_jump_reject_count == 1 or ik_jump_reject_count % 20 == 0:
                    print(
                        f"[IK][JUMP] reject solution: J{worst_j_idx + 1} jump="
                        f"{np.rad2deg(dq_max):.2f}deg (threshold="
                        f"{np.rad2deg(ik_effective_jump_thr_rad):.1f}deg, total_reject="
                        f"{ik_jump_reject_count}). Wrist singularity? "
                        f"Try --position-only."
                    )
                return None
        if args.control_mode == "real" and not args.dry_run:
            if not arm_hw.q_rad_within_limits(q_try, min_margin_deg=0.0):
                ik_limit_reject_count += 1
                if ik_limit_reject_count == 1 or ik_limit_reject_count % 20 == 0:
                    lim_bad = arm_hw.get_joint_limit_margin_deg(q_try)
                    if lim_bad is not None:
                        print(
                            f"[IK][LIMIT] reject solution: J{lim_bad['min_margin_joint']} "
                            f"margin={lim_bad['min_margin_deg']:.1f}deg "
                            f"(total_reject={ik_limit_reject_count})."
                        )
                    else:
                        print(
                            f"[IK][LIMIT] reject solution "
                            f"(total_reject={ik_limit_reject_count})."
                        )
                return None
        if args.self_collision_filter and robot_model.cc is not None:
            robot_model.goto_given_conf(q_try)
            candidate_collided, candidate_contacts = robot_model.is_collided(toggle_contacts=True)
            candidate_collided = bool(candidate_collided)
            candidate_contact_count = len(candidate_contacts)
            robot_model.goto_given_conf(q_seed)
            if candidate_collided:
                if collision_filter_strict:
                    collision_reject_count += 1
                    if collision_reject_count == 1 or collision_reject_count % 20 == 0:
                        print(
                            f"[COLLISION] reject IK solution, total_reject={collision_reject_count}, "
                            f"contacts={candidate_contact_count}"
                        )
                    return None
                collision_bypass_count += 1
                if collision_bypass_count == 1 or collision_bypass_count % 50 == 0:
                    print(
                        "[COLLISION] soft bypass collided solution, "
                        f"total_bypass={collision_bypass_count}, "
                        f"contacts={candidate_contact_count}"
                    )
            elif not collision_filter_strict:
                collision_filter_strict = True
                print("[COLLISION] recovered to collision-free state, strict filter enabled.")
        return limit_joint_step(q_seed, q_try, max_joint_step_rad)

    try:
        while True:
            loop_t_perf = time.perf_counter()
            xr_read_t0 = time.time()
            xr = get_xr_state()
            perf_xr_sum += (time.time() - xr_read_t0)
            if xr is not None:
                xr_rot = quat_xyzw_to_rotmat(xr.controller_quat_vr)

                if xr.estop_pressed:
                    arm_hw.emergency_stop()
                    print("[SAFE] estop pressed, stopping loop.")
                    break

                home_pressed = _get_home_pressed() if _REPLAY_READER is None else False
                if home_pressed and (not prev_home_pressed):
                    print("[CTRL] home hold engaged")
                    # Sync seed + servo baseline to measured pose so the first home-step
                    # is relative to hardware, not stale clutch/idle q_seed. Without this
                    # the firmware sees a >max_lag command, rejects frames, and SERVO_DIAG
                    # shows a 50+deg "jump" while the arm freezes.
                    if args.control_mode == "real" and not args.dry_run:
                        q_seed = sync_q_seed_from_hardware(
                            q_seed, robot_model, arm_hw, resync_servo=True, tag="HOME_ENGAGE"
                        )
                        tcp_pos_anchor, tcp_rot_anchor = robot_model.fk(q_seed)
                        tcp_target_pos, tcp_target_rot = tcp_pos_anchor.copy(), tcp_rot_anchor.copy()
                        tcp_sent_pos, tcp_sent_rot = tcp_pos_anchor.copy(), tcp_rot_anchor.copy()
                if (not home_pressed) and prev_home_pressed:
                    print("[CTRL] home hold released")
                    if args.control_mode == "real" and not args.dry_run:
                        q_seed = sync_q_seed_from_hardware(
                            q_seed, robot_model, arm_hw, resync_servo=True, tag="HOME"
                        )
                        tcp_pos_anchor, tcp_rot_anchor = robot_model.fk(q_seed)
                        tcp_target_pos, tcp_target_rot = tcp_pos_anchor.copy(), tcp_rot_anchor.copy()
                        tcp_sent_pos, tcp_sent_rot = tcp_pos_anchor.copy(), tcp_rot_anchor.copy()
                prev_home_pressed = home_pressed

                # A 开夹爪，B 闭夹爪（按键上升沿触发，避免按住重复下发）
                if xr.button_a_pressed and not prev_a_pressed:
                    do_open = not bool(args.invert_gripper_buttons)
                    if do_open:
                        if jaw_open_width is not None:
                            robot_model.change_jaw_width(jaw_open_width)
                            gripper_vis_dirty = True
                            print(f"[GRIPPER] A pressed -> sim open -> {jaw_open_width:.4f} m")
                        if not args.dry_run:
                            arm_hw.open_gripper()
                    else:
                        if jaw_close_width is not None:
                            robot_model.change_jaw_width(jaw_close_width)
                            gripper_vis_dirty = True
                            print(f"[GRIPPER] A pressed -> sim close -> {jaw_close_width:.4f} m")
                        if not args.dry_run:
                            arm_hw.close_gripper()
                if xr.button_b_pressed and not prev_b_pressed:
                    do_open = bool(args.invert_gripper_buttons)
                    if do_open:
                        if jaw_open_width is not None:
                            robot_model.change_jaw_width(jaw_open_width)
                            gripper_vis_dirty = True
                            print(f"[GRIPPER] B pressed -> sim open -> {jaw_open_width:.4f} m")
                        if not args.dry_run:
                            arm_hw.open_gripper()
                    else:
                        if jaw_close_width is not None:
                            robot_model.change_jaw_width(jaw_close_width)
                            gripper_vis_dirty = True
                            print(f"[GRIPPER] B pressed -> sim close -> {jaw_close_width:.4f} m")
                        if not args.dry_run:
                            arm_hw.close_gripper()
                prev_a_pressed = xr.button_a_pressed
                prev_b_pressed = xr.button_b_pressed

                if gripper_vis_dirty and args.visualize and base is not None:
                    if robot_vis_model is not None:
                        robot_vis_model.detach()
                    robot_vis_model = robot_model.gen_meshmodel(alpha=0.8, toggle_tcp_frame=True)
                    robot_vis_model.attach_to(base)
                    gripper_vis_dirty = False

                if home_pressed:
                    clutch_active = False
                    q_cmd = limit_joint_step(q_seed, q_home, home_joint_step_rad)
                    q_seed = q_cmd
                    robot_model.goto_given_conf(q_seed)
                    tcp_pos_anchor, tcp_rot_anchor = robot_model.fk(q_seed)
                    tcp_target_pos, tcp_target_rot = tcp_pos_anchor.copy(), tcp_rot_anchor.copy()
                    tcp_sent_pos, tcp_sent_rot = tcp_pos_anchor.copy(), tcp_rot_anchor.copy()
                    if not args.dry_run:
                        hw_t0 = time.time()
                        home_cmd_step = home_joint_step_rad if home_joint_step_rad > 0.0 else None
                        arm_hw.send_joint_positions(
                            q_seed, dt, cmd_max_step_rad=home_cmd_step, skip_cmd_lp=True
                        )
                        perf_hw_sum += (time.time() - hw_t0)
                    if args.visualize and base is not None:
                        vis_frame_id += 1
                        vis_div = max(int(args.vis_update_div), 1)
                        if vis_frame_id % vis_div == 0:
                            if robot_vis_model is not None:
                                robot_vis_model.detach()
                            robot_vis_model = robot_model.gen_meshmodel(alpha=0.8, toggle_tcp_frame=True)
                            robot_vis_model.attach_to(base)
                else:
                    if xr.grip_value >= grip_engage_thr:
                        grip_engage_streak += 1
                    else:
                        grip_engage_streak = 0
                    if xr.grip_value < grip_release_thr:
                        grip_release_streak += 1
                    else:
                        grip_release_streak = 0
                    if clutch_active:
                        grip_follow = grip_release_streak < release_stable_frames
                    else:
                        grip_follow = grip_engage_streak >= engage_stable_frames
                    if grip_follow and not clutch_active:
                        cooldown_s = max(float(args.clutch_reengage_cooldown_s), 0.0)
                        since_rel = time.time() - clutch_release_t
                        if clutch_release_t > 0.0 and since_rel < cooldown_s:
                            if not clutch_cooldown_warned:
                                print(
                                    f"[CTRL] grip 重按过快 ({since_rel:.2f}s < {cooldown_s:.2f}s)，"
                                    "稍停稳再跟以免乱窜。"
                                )
                                clutch_cooldown_warned = True
                        else:
                            clutch_cooldown_warned = False
                            lim_now = None
                            if args.control_mode == "real" and not args.dry_run:
                                lim_now = arm_hw.get_joint_limit_margin_deg(q_seed)
                            min_margin_thr = float(args.clutch_min_margin_deg)
                            if (
                                lim_now is not None
                                and float(lim_now["min_margin_deg"]) < min_margin_thr
                            ):
                                if not clutch_block_warned:
                                    j_idx = int(lim_now["min_margin_joint"])
                                    margin = float(lim_now["min_margin_deg"])
                                    print(
                                        f"[CTRL] grip 跟随被阻止: J{j_idx} margin={margin:.1f}deg "
                                        f"(需要 >= {min_margin_thr:.0f}deg)。"
                                        "请先归位或换姿态。"
                                    )
                                    clutch_block_warned = True
                            else:
                                clutch_block_warned = False
                                clutch_active = True
                                hold_tcp_pos = tcp_sent_pos.copy()
                                hold_tcp_rot = tcp_sent_rot.copy()
                                hold_q = q_seed.copy()
                                preserved_command_tcp = False
                                if args.control_mode == "real" and not args.dry_run:
                                    snap_max_m = (
                                        max(float(args.clutch_engage_snap_max_mm), 0.0) / 1000.0
                                    )
                                    try:
                                        q_meas = np.asarray(
                                            arm_hw.get_joint_values(), dtype=float
                                        )
                                        if (
                                            q_meas.shape == q_seed.shape
                                            and np.all(np.isfinite(q_meas))
                                        ):
                                            robot_model.goto_given_conf(q_meas)
                                            fk_meas_pos, fk_meas_rot = robot_model.fk(q_meas)
                                            tcp_err_m = float(
                                                np.linalg.norm(fk_meas_pos - hold_tcp_pos)
                                            )
                                            if snap_max_m > 0.0 and tcp_err_m <= snap_max_m:
                                                # 保留指令 TCP 锚点，但 IK/servo 从实测关节起步，
                                                # 避免把 hold_q 硬灌给真机造成接合瞬间拉扯。
                                                q_seed = q_meas.copy()
                                                robot_model.goto_given_conf(q_seed)
                                                arm_hw.resync_servo_to_measured()
                                                tcp_pos_anchor = hold_tcp_pos.copy()
                                                tcp_rot_anchor = hold_tcp_rot.copy()
                                                preserved_command_tcp = True
                                                if tcp_err_m > 1e-4:
                                                    print(
                                                        "[CTRL] clutch engage: preserve command TCP "
                                                        f"(meas err={tcp_err_m * 1000.0:.1f}mm "
                                                        f"< {snap_max_m * 1000.0:.1f}mm)"
                                                    )
                                            else:
                                                q_seed = q_meas.copy()
                                                arm_hw.resync_servo_to_measured()
                                                tcp_pos_anchor = fk_meas_pos.copy()
                                                tcp_rot_anchor = fk_meas_rot.copy()
                                                if snap_max_m > 0.0 and tcp_err_m > snap_max_m:
                                                    print(
                                                        "[CTRL] clutch engage: snap to measured TCP "
                                                        f"(err={tcp_err_m * 1000.0:.1f}mm)"
                                                    )
                                        else:
                                            q_seed = sync_q_seed_from_hardware(
                                                q_seed,
                                                robot_model,
                                                arm_hw,
                                                resync_servo=True,
                                                tag="CLUTCH",
                                            )
                                            tcp_pos_anchor, tcp_rot_anchor = robot_model.fk(
                                                q_seed
                                            )
                                    except Exception as exc:
                                        print(f"[CLUTCH][WARN] engage sync failed: {exc}")
                                        q_seed = sync_q_seed_from_hardware(
                                            q_seed,
                                            robot_model,
                                            arm_hw,
                                            resync_servo=True,
                                            tag="CLUTCH",
                                        )
                                        tcp_pos_anchor, tcp_rot_anchor = robot_model.fk(q_seed)
                                else:
                                    tcp_pos_anchor, tcp_rot_anchor = robot_model.fk(q_seed)
                                tcp_target_pos = tcp_pos_anchor.copy()
                                tcp_target_rot = tcp_rot_anchor.copy()
                                tcp_sent_pos = tcp_pos_anchor.copy()
                                tcp_sent_rot = tcp_rot_anchor.copy()
                                xr_anchor_pos = xr.controller_pos_vr.copy()
                                xr_anchor_rot = xr_rot.copy()
                                last_ik_target_pos = None
                                last_ik_target_rot = None
                                ik_adapt_level = 0
                                ik_adapt_recover_count = 0
                                ik_zero_streak = 0
                                clutch_engage_grace_remain = max(
                                    int(args.clutch_engage_grace_frames), 0
                                )
                                update_ik_adapt_settings(reason="clutch engage")
                                print(
                                    f"[CTRL] clutch engaged (grip={xr.grip_value:.2f}, "
                                    f"q_seed_deg={np.rad2deg(q_seed)})"
                                )

                    elif (not grip_follow) and clutch_active:
                        clutch_active = False
                        clutch_release_t = time.time()
                        clutch_engage_grace_remain = 0
                        print(f"[CTRL] clutch released (grip={xr.grip_value:.2f})")
                        if args.control_mode == "real" and not args.dry_run:
                            arm_hw.resync_servo_to_command(q_seed)
                            (
                                tcp_sent_pos,
                                tcp_sent_rot,
                                tcp_target_pos,
                                tcp_target_rot,
                            ) = align_tcp_chain_from_q_seed(robot_model, q_seed)

                # Keep feeding servo_j in idle state; otherwise the firmware
                # watchdog may expire and motors feel like "brake/jerk" on release.
                if (
                    (not args.dry_run)
                    and (not clutch_active)
                    and (not home_pressed)
                    and arm_hw.is_servo_streaming()
                ):
                    hw_t0 = time.time()
                    arm_hw.feed_idle_servo_hold(q_hold=q_seed)
                    perf_hw_sum += (time.time() - hw_t0)

                if clutch_active and (not home_pressed):
                    # 1) 计算 VR 相对位移
                    dpos_vr = (xr.controller_pos_vr - xr_anchor_pos) * args.pos_scale
                    dR_vr = xr_rot @ xr_anchor_rot.T
                    if clutch_engage_grace_remain > 0:
                        grace_total = max(int(args.clutch_engage_grace_frames), 1)
                        progressed = grace_total - clutch_engage_grace_remain
                        grace_blend = min(1.0, max(0.0, progressed / grace_total))
                        clutch_engage_grace_remain -= 1
                        dpos_vr = dpos_vr * grace_blend
                        dR_vr = scale_rotation_delta(dR_vr, grace_blend)
                    # 2) 映射到机器人基坐标
                    dpos_base = r_vr_fix @ (r_br @ dpos_vr)
                    # 可选：限制相对位移幅度，避免目标跑到可达空间外
                    if args.max_delta_pos > 0:
                        dnorm = np.linalg.norm(dpos_base)
                        if dnorm > args.max_delta_pos and dnorm > 1e-12:
                            dpos_base = dpos_base / dnorm * args.max_delta_pos

                    # 3) 计算 VR 相对旋转，再映射到机器人基坐标
                    dR_base = r_vr_fix @ (r_br @ dR_vr @ r_br.T) @ r_vr_fix.T
                    dR_base = scale_rotation_delta_axes(dR_base, np.asarray(args.rot_axis_gain, dtype=float))
                    rot_scale_eff = ik_effective_rot_scale
                    if args.control_mode == "real" and not args.dry_run:
                        lim_rot = arm_hw.get_joint_limit_margin_deg(q_seed)
                        j6_margin_thr = max(float(args.ik_adapt_j6_margin_deg), 0.0)
                        if (
                            lim_rot is not None
                            and int(lim_rot["min_margin_joint"]) == 6
                            and float(lim_rot["min_margin_deg"]) < j6_margin_thr
                        ):
                            rot_scale_eff *= max(
                                float(lim_rot["min_margin_deg"]) / j6_margin_thr,
                                0.15,
                            )
                    dR_base = scale_rotation_delta(dR_base, rot_scale_eff)

                    # 4) 生成“原始目标”TCP
                    raw_tgt_pos = tcp_pos_anchor + dpos_base + t_br
                    # 真机已顶到关节软限位时，继续下压只会拉大跟踪误差（日志里常见 J2 margin<0）。
                    if args.control_mode == "real" and not args.dry_run:
                        lim_now = arm_hw.get_joint_limit_margin_deg(q_seed)
                        if (
                            lim_now is not None
                            and float(lim_now["min_margin_deg"]) < 0.0
                            and float(raw_tgt_pos[2]) < float(tcp_sent_pos[2]) - 1e-4
                        ):
                            tcp_down_stall_warn_count += 1
                            if tcp_down_stall_warn_count == 1 or tcp_down_stall_warn_count % 25 == 0:
                                j_idx = int(lim_now["min_margin_joint"])
                                print(
                                    f"[TCP][STALL] 无法继续下探: J{j_idx} 已超出软限位 "
                                    f"{-float(lim_now['min_margin_deg']):.1f}deg。"
                                    f"目标Z={float(raw_tgt_pos[2]):.4f}m, 当前Z={float(tcp_sent_pos[2]):.4f}m。"
                                    f"请先松开 grip 抬高肘部或换姿态后重新 clutch。"
                                )
                            raw_tgt_pos[2] = float(tcp_sent_pos[2])
                    if args.position_only:
                        raw_tgt_rot = tcp_rot_anchor
                    else:
                        raw_tgt_rot = dR_base @ tcp_rot_anchor

                    # 4.1) 工作空间硬限幅：先把目标点夹到可达区域盒子内，再做后续限速与 IK
                    if workspace_clamp_active:
                        raw_tgt_pos = np.array([
                            np.clip(raw_tgt_pos[0], args.ws_x_min, args.ws_x_max),
                            np.clip(raw_tgt_pos[1], args.ws_y_min, args.ws_y_max),
                            np.clip(raw_tgt_pos[2], args.ws_z_min, args.ws_z_max),
                        ])

                    # 5) 限速/限角速度，避免跳变
                    tcp_target_pos = limit_linear_step(tcp_sent_pos, raw_tgt_pos, max_lin_step)
                    tcp_target_rot = limit_angular_step(tcp_sent_rot, raw_tgt_rot, max_ang_step)

                    # 诊断日志：同时打印“目标TCP”和“当前执行TCP(FK from q_seed)”。
                    # 低频输出避免刷屏，同时足够看出“目标在动但执行卡住”的情况。
                    now_diag_t = time.time()
                    if now_diag_t - last_tcp_diag_print_t >= max(float(args.tcp_log_interval), 0.0):
                        tcp_exec_pos, _ = robot_model.fk(q_seed)
                        tcp_err = float(np.linalg.norm(tcp_target_pos - tcp_exec_pos))
                        print(
                            f"[TCP][DIAG] target(m)={tcp_target_pos}, "
                            f"exec(m)={tcp_exec_pos}, err={tcp_err:.4f}"
                        )
                        last_tcp_diag_print_t = now_diag_t
                    if (
                        args.self_collision_filter
                        and robot_model.cc is not None
                        and float(args.collision_log_interval) > 0.0
                        and (now_diag_t - last_collision_log_t) >= float(args.collision_log_interval)
                    ):
                        cur_collided, cur_contacts = robot_model.is_collided(toggle_contacts=True)
                        print(
                            f"[COLLISION][STATE] collided={bool(cur_collided)}, "
                            f"contacts={len(cur_contacts)}"
                        )
                        last_collision_log_t = now_diag_t
                    if (
                        args.control_mode == "real"
                        and float(args.joint_limit_log_interval) > 0.0
                        and (now_diag_t - last_joint_limit_log_t) >= float(args.joint_limit_log_interval)
                    ):
                        lim_info = arm_hw.get_joint_limit_margin_deg(q_seed)
                        if lim_info is not None:
                            margin_text = ", ".join(
                                f"J{j}:lo={dlo:.1f}deg,hi={dhi:.1f}deg"
                                for j, dlo, dhi, _ in lim_info["margins"]
                            )
                            print(
                                f"[JOINT][LIMIT] near=J{lim_info['min_margin_joint']} "
                                f"margin={lim_info['min_margin_deg']:.1f}deg | {margin_text}"
                            )
                        else:
                            print("[JOINT][LIMIT] limit info unavailable from current backend.")
                        last_joint_limit_log_t = now_diag_t

                    # 6) IK 求解（用上一帧关节作为 seed，保证连续）
                    ik_target_moved = True
                    if last_ik_target_pos is not None and last_ik_target_rot is not None:
                        pos_delta = float(np.linalg.norm(tcp_target_pos - last_ik_target_pos))
                        rot_delta = rotation_angle_rad(last_ik_target_rot, tcp_target_rot)
                        ik_target_moved = (
                            pos_delta >= ik_target_pos_eps_m
                            or rot_delta >= ik_target_rot_eps_rad
                        )

                    last_ik_target_pos = tcp_target_pos.copy()
                    last_ik_target_rot = tcp_target_rot.copy()

                    if not ik_target_moved:
                        if (
                            (not args.dry_run)
                            and arm_hw.is_servo_streaming()
                        ):
                            hw_t0 = time.time()
                            ok = arm_hw.send_joint_positions(q_seed, dt)
                            if not ok:
                                (
                                    q_seed,
                                    tcp_sent_pos,
                                    tcp_sent_rot,
                                    tcp_target_pos,
                                    tcp_target_rot,
                                    tcp_pos_anchor,
                                    tcp_rot_anchor,
                                    xr_anchor_pos,
                                    xr_anchor_rot,
                                    last_ik_target_pos,
                                    last_ik_target_rot,
                                ) = apply_servo_reject_recovery(
                                    q_seed,
                                    robot_model,
                                    arm_hw,
                                    xr=xr,
                                    xr_rot=xr_rot,
                                    q_cmd=q_seed,
                                    recover_step_rad=max(max_joint_step_rad * 0.5, math.radians(1.0)),
                                )
                            perf_hw_sum += (time.time() - hw_t0)
                    else:
                        ik_t0 = time.time()
                        ik_try_count += 1
                        q_sol = None
                        pos_try = tcp_target_pos
                        rot_try = tcp_target_rot

                        q_fast = robot_model.ik(
                            tgt_pos=pos_try,
                            tgt_rotmat=rot_try,
                            seed_jnt_values=q_seed,
                            toggle_dbg=False,
                        )
                        q_sol = filter_ik_candidate(q_fast)

                        if q_sol is None and args.ik_retry:
                            rot_candidates = [tcp_target_rot]
                            if not args.position_only:
                                for ang in (
                                    np.deg2rad(8), np.deg2rad(-8),
                                    np.deg2rad(16), np.deg2rad(-16),
                                ):
                                    rot_candidates.append(
                                        tcp_target_rot @ rm.rotmat_from_axangle(tcp_target_rot[:, 2], ang)
                                    )
                                if args.ik_allow_flip_candidate:
                                    yaw_base = float(np.arctan2(tcp_target_rot[1, 0], tcp_target_rot[0, 0]))
                                    for dyaw in (
                                        0.0, np.deg2rad(15), np.deg2rad(-15),
                                        np.deg2rad(30), np.deg2rad(-30),
                                    ):
                                        rot_candidates.append(
                                            rm.rotmat_from_euler(np.pi, 0.0, yaw_base + dyaw)
                                        )

                            pos_candidates = [tcp_target_pos]
                            dpos = tcp_target_pos - tcp_sent_pos
                            for s in (0.5, 0.25):
                                pos_candidates.append(tcp_sent_pos + dpos * s)

                            seed_candidates = [q_seed]
                            if not args.position_only:
                                seed_candidates.append(q_seed + 0.05 * np.sin(np.arange(q_seed.size)))

                            max_attempts = int(args.ik_max_attempts_per_cycle)
                            attempt_count = 0
                            stop_search = False
                            for seed_try in seed_candidates:
                                if q_sol is not None:
                                    break
                                if stop_search:
                                    break
                                for pos_c in pos_candidates:
                                    if q_sol is not None:
                                        break
                                    if stop_search:
                                        break
                                    for rot_c in rot_candidates:
                                        if max_attempts > 0 and attempt_count >= max_attempts:
                                            stop_search = True
                                            break
                                        attempt_count += 1
                                        q_try = robot_model.ik(
                                            tgt_pos=pos_c,
                                            tgt_rotmat=rot_c,
                                            seed_jnt_values=seed_try,
                                            toggle_dbg=False,
                                        )
                                        q_sol = filter_ik_candidate(q_try)
                                        if q_sol is not None:
                                            pos_try = pos_c
                                            rot_try = rot_c
                                            break

                        if q_sol is not None:
                            tcp_target_pos = pos_try
                            tcp_target_rot = rot_try

                        if q_sol is not None:
                            ik_success_count += 1
                            ik_zero_streak = 0
                            if args.ik_adaptive_ease and ik_adapt_level > 0:
                                ik_adapt_recover_count += 1
                                if ik_adapt_recover_count >= max(int(args.ik_adapt_recover_streak), 1):
                                    ik_adapt_level = 0
                                    ik_adapt_recover_count = 0
                                    update_ik_adapt_settings(reason="recovered")
                            q_seed = q_sol
                            robot_model.goto_given_conf(q_seed)
                            tcp_sent_pos, tcp_sent_rot = robot_model.fk(q_seed)
                            now_pos_log_t = time.time()
                            if now_pos_log_t - last_tcp_pos_log_t >= max(float(args.tcp_pos_log_interval), 0.0):
                                print(f"[TCP] pos(m) = {tcp_sent_pos}")
                                last_tcp_pos_log_t = now_pos_log_t

                            if not args.dry_run:
                                hw_t0 = time.time()
                                ok = arm_hw.send_joint_positions(q_seed, dt)
                                if not ok:
                                    (
                                        q_seed,
                                        tcp_sent_pos,
                                        tcp_sent_rot,
                                        tcp_target_pos,
                                        tcp_target_rot,
                                        tcp_pos_anchor,
                                        tcp_rot_anchor,
                                        xr_anchor_pos,
                                        xr_anchor_rot,
                                        last_ik_target_pos,
                                        last_ik_target_rot,
                                    ) = apply_servo_reject_recovery(
                                        q_seed,
                                        robot_model,
                                        arm_hw,
                                        xr=xr,
                                        xr_rot=xr_rot,
                                        q_cmd=q_seed,
                                        recover_step_rad=max(
                                            max_joint_step_rad * 0.5, math.radians(1.0)
                                        ),
                                    )
                                perf_hw_sum += (time.time() - hw_t0)

                            if args.visualize and base is not None:
                                vis_frame_id += 1
                                trail_frame_id += 1
                                vis_div = max(int(args.vis_update_div), 1)
                                trail_step = int(args.trail_step)

                                if vis_frame_id % vis_div == 0:
                                    if robot_vis_model is not None:
                                        robot_vis_model.detach()
                                    robot_vis_model = robot_model.gen_meshmodel(alpha=0.8, toggle_tcp_frame=True)
                                    robot_vis_model.attach_to(base)

                                if trail_step > 0 and trail_frame_id % trail_step == 0:
                                    pt = mgm.gen_sphere(
                                        pos=tcp_sent_pos,
                                        radius=args.trail_radius,
                                        rgb=np.array([1.0, 0.2, 0.2]),
                                        alpha=0.7,
                                    )
                                    pt.attach_to(base)
                                    trail_points.append(pt)
                        else:
                            ik_zero_streak += 1
                            ik_adapt_recover_count = 0
                            if (
                                args.ik_adaptive_ease
                                and ik_zero_streak >= max(int(args.ik_adapt_fail_streak), 1)
                            ):
                                fail_step = max(int(args.ik_adapt_fail_streak), 1)
                                new_level = min(ik_zero_streak // fail_step, 3)
                                if new_level != ik_adapt_level:
                                    ik_adapt_level = new_level
                                    update_ik_adapt_settings(reason="consecutive fail")
                            if ik_zero_streak == max(int(args.ik_fail_warn_threshold), 1):
                                drift = float(np.linalg.norm(tcp_target_pos - tcp_sent_pos))
                                print(
                                    f"[IK][WARN] consecutive_fail={ik_zero_streak}, "
                                    f"target_vs_last_success_dist={drift:.4f} m, "
                                    f"target={tcp_target_pos}, last_success={tcp_sent_pos}"
                                )
                            if (
                                (not args.dry_run)
                                and arm_hw.is_servo_streaming()
                            ):
                                hw_t0 = time.time()
                                ok = arm_hw.send_joint_positions(q_seed, dt)
                                if not ok:
                                    (
                                        q_seed,
                                        tcp_sent_pos,
                                        tcp_sent_rot,
                                        tcp_target_pos,
                                        tcp_target_rot,
                                        tcp_pos_anchor,
                                        tcp_rot_anchor,
                                        xr_anchor_pos,
                                        xr_anchor_rot,
                                        last_ik_target_pos,
                                        last_ik_target_rot,
                                    ) = apply_servo_reject_recovery(
                                        q_seed,
                                        robot_model,
                                        arm_hw,
                                        xr=xr,
                                        xr_rot=xr_rot,
                                        q_cmd=q_seed,
                                        recover_step_rad=max(
                                            max_joint_step_rad * 0.5, math.radians(1.0)
                                        ),
                                    )
                                perf_hw_sum += (time.time() - hw_t0)

                        perf_ik_sum += (time.time() - ik_t0)

                    # 如果启用 workspace clamp 后长期 0 成功，自动关闭 clamp 防止全程卡死。
                    if workspace_clamp_active and ik_success_count == 0 and ik_try_count >= 30:
                        workspace_clamp_active = False
                        print("[IK] workspace clamp auto-disabled (0 success in early stage).")

                    # 每秒打印一次 IK 成功率，便于判断“没轨迹”是否由 IK 失败导致。
                    now_t = time.time()
                    if now_t - last_ik_stat_print_t >= 1.0:
                        ratio = 0.0 if ik_try_count == 0 else (ik_success_count / ik_try_count * 100.0)
                        print(
                            f"[IK] success={ik_success_count}/{ik_try_count} ({ratio:.1f}%), "
                            f"collision_reject={collision_reject_count}, "
                            f"collision_bypass={collision_bypass_count}, "
                            f"limit_reject={ik_limit_reject_count}, "
                            f"strict={collision_filter_strict}"
                        )
                        last_ik_stat_print_t = now_t

            # 回放数据结束：正常退出，不触发超时急停。
            if _REPLAY_READER is not None and _REPLAY_READER.is_finished():
                print("[REPLAY] finished all frames, stop loop.")
                break

            if args.visualize and base is not None:
                base.taskMgr.step()
            if _REPLAY_READER is None:
                _xr_touch_heartbeat()

            if (
                _REPLAY_READER is None
                and _XR_LINK_READY
                and (time.time() - _LAST_XR_POLL_WALL_TS > timeout_s)
            ):
                arm_hw.emergency_stop()
                print(
                    f"[SAFE] XR timeout ({(time.time() - _LAST_XR_POLL_WALL_TS) * 1000:.0f} ms "
                    f"> {timeout_s * 1000:.0f} ms), emergency stop."
                )
                break

            loop_deadline += dt
            sleep_t_req = loop_deadline - time.perf_counter()
            sleep_t_actual = 0.0
            if sleep_t_req > 0:
                sleep_start_t = time.perf_counter()
                time.sleep(sleep_t_req)
                sleep_t_actual = time.perf_counter() - sleep_start_t
            else:
                perf_overrun_count += 1
                # Overrun: re-anchor deadline to avoid accumulating delay.
                loop_deadline = time.perf_counter()
            loop_end_perf = time.perf_counter()
            loop_elapsed = loop_end_perf - loop_t_perf
            compute_elapsed = max(loop_elapsed - sleep_t_actual, 0.0)
            perf_loop_count += 1
            perf_loop_sum += loop_elapsed
            perf_loop_max = max(perf_loop_max, loop_elapsed)
            perf_compute_sum += compute_elapsed
            perf_sleep_req_sum += max(sleep_t_req, 0.0)
            perf_sleep_actual_sum += sleep_t_actual

            perf_window_elapsed = loop_end_perf - perf_window_start_t
            if perf_window_elapsed >= 1.0 and perf_loop_count > 0:
                avg_loop_ms = perf_loop_sum / perf_loop_count * 1000.0
                avg_compute_ms = perf_compute_sum / perf_loop_count * 1000.0
                avg_sleep_req_ms = perf_sleep_req_sum / perf_loop_count * 1000.0
                avg_sleep_actual_ms = perf_sleep_actual_sum / perf_loop_count * 1000.0
                avg_xr_ms = perf_xr_sum / perf_loop_count * 1000.0
                avg_ik_ms = perf_ik_sum / perf_loop_count * 1000.0
                avg_hw_ms = perf_hw_sum / perf_loop_count * 1000.0
                fps = perf_loop_count / perf_window_elapsed
                print(
                    f"[PERF] fps={fps:.1f}, "
                    f"loop_avg={avg_loop_ms:.2f}ms, loop_max={perf_loop_max * 1000.0:.2f}ms, "
                    f"compute_avg={avg_compute_ms:.2f}ms, "
                    f"sleep_req_avg={avg_sleep_req_ms:.2f}ms, sleep_actual_avg={avg_sleep_actual_ms:.2f}ms, "
                    f"overrun={perf_overrun_count}/{perf_loop_count}, "
                    f"xr_avg={avg_xr_ms:.2f}ms, ik_avg={avg_ik_ms:.2f}ms, hw_avg={avg_hw_ms:.2f}ms"
                )
                if args.control_mode == "real" and args.servo_diag:
                    servo_diag = arm_hw.pop_servo_diag()
                    if servo_diag is not None:
                        print(
                            "[SERVO][DIAG] "
                            f"n={servo_diag['count']}, "
                            f"cmd_step_avg={np.rad2deg(servo_diag['cmd_step_avg']):.3f}deg, "
                            f"cmd_step_max={np.rad2deg(servo_diag['cmd_step_max']):.3f}deg, "
                            f"cmd_vel_avg={np.rad2deg(servo_diag['cmd_vel_avg']):.1f}deg/s, "
                            f"cmd_vel_max={np.rad2deg(servo_diag['cmd_vel_max']):.1f}deg/s, "
                            f"meas_step_avg={np.rad2deg(servo_diag['meas_step_avg']):.3f}deg, "
                            f"meas_step_max={np.rad2deg(servo_diag['meas_step_max']):.3f}deg, "
                            f"track_err_avg={np.rad2deg(servo_diag['track_err_avg']):.3f}deg, "
                            f"track_err_max={np.rad2deg(servo_diag['track_err_max']):.3f}deg, "
                            f"chatter={servo_diag['chatter_count']}, "
                            f"reject_delta={servo_diag['reject_delta']}, "
                            f"worst_j={servo_diag['worst_joint_idx'] + 1}, "
                            f"worst_err={np.rad2deg(servo_diag['worst_joint_err']):.3f}deg"
                        )
                perf_window_start_t = loop_end_perf
                perf_loop_count = 0
                perf_loop_sum = 0.0
                perf_loop_max = 0.0
                perf_compute_sum = 0.0
                perf_sleep_req_sum = 0.0
                perf_sleep_actual_sum = 0.0
                perf_overrun_count = 0
                perf_xr_sum = 0.0
                perf_ik_sum = 0.0
                perf_hw_sum = 0.0
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if _XRT is not None:
            _xr_safe_call(_XRT.close, None)
            _XRT = None
        _LAST_XR_TIMESTAMP_NS = None
        _XR_LINK_READY = False
        _LAST_XR_POLL_WALL_TS = 0.0
        final_ratio = 0.0 if ik_try_count == 0 else (ik_success_count / ik_try_count * 100.0)
        print(f"[IK][FINAL] success={ik_success_count}/{ik_try_count} ({final_ratio:.1f}%)")
        arm_hw.disable()
        print("Teleop bridge exited.")


if __name__ == "__main__":
    main()

