#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/06/10 18:25
# @Author : HuHanYing
"""Dual-arm VR teleop: right controller -> right arm, left controller -> left arm."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

import teleop_bridge_panthera as tbp
from wrs import rm
from wrs.robot_sim.robots.panthera_ht.panthera_ht import PantheraHTSglArm


def _parse_joint_sign(values, label: str) -> np.ndarray:
    sign = np.asarray(values, dtype=float).reshape(-1)
    if sign.size != 6:
        raise ValueError(f"{label} joint sign requires 6 values, got {sign.size}")
    sign = np.sign(sign)
    sign[sign == 0.0] = 1.0
    return sign


def _parse_preset_joint_deltas(label: str, values, default) -> np.ndarray:
    if values is None:
        arr = np.asarray(default, dtype=float)
    else:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != 6:
            raise ValueError(f"{label} requires 6 joint deltas, got {arr.size}")
    return np.deg2rad(arr)


class ArmTeleopChannel:
    """One arm + one XR controller channel."""

    def __init__(
        self,
        *,
        label: str,
        hw_port: str,
        home_button: str,
        gripper_open_label: str,
        gripper_close_label: str,
        has_gripper: bool,
        joint_sign: np.ndarray,
        hw_cfg_path: str,
        args,
        dt: float,
        r_br: np.ndarray,
        t_br: np.ndarray,
        r_vr_fix: np.ndarray,
        max_lin_step: float,
        max_ang_step: float,
        max_joint_step_rad: float,
        home_joint_step_rad: float,
        servo_max_step_rad: float,
        servo_max_lag_rad: float,
        servo_cmd_max_step_rad: float,
    ) -> None:
        self.label = label
        self.home_button = home_button
        self.home_to_zero = (label != "LEFT") or bool(getattr(args, "left_home_to_zero", False))
        self.gripper_open_label = gripper_open_label
        self.gripper_close_label = gripper_close_label
        self.has_gripper = bool(has_gripper)
        self.joint_sign = np.asarray(joint_sign, dtype=float).copy()
        self.args = args
        self.dt = dt
        self.r_br = r_br
        self.t_br = t_br
        self.r_vr_fix = r_vr_fix
        self.max_lin_step = max_lin_step
        self.max_ang_step = max_ang_step
        self.max_joint_step_rad = max_joint_step_rad
        self.home_joint_step_rad = home_joint_step_rad

        self.robot_model = PantheraHTSglArm(enable_cc=True)
        self.q_seed = self.robot_model.get_jnt_values()
        if self.home_to_zero and self.label == "RIGHT":
            self.q_home = tbp.resolve_right_home_q(self.q_seed.shape[0], args)
        else:
            self.q_home = np.asarray(
                getattr(self.robot_model, "home_conf", np.zeros_like(self.q_seed)), dtype=float
            ).copy()
        if self.q_home.shape != self.q_seed.shape:
            self.q_home = np.zeros_like(self.q_seed)
        self.robot_model.goto_given_conf(self.q_seed)
        self.tcp_pos_anchor, self.tcp_rot_anchor = self.robot_model.fk(self.q_seed)

        hw_cfg_path = hw_cfg_path.strip()
        if not hw_cfg_path:
            hw_cfg_path = args.hw_cfg.strip()
        if args.control_mode == "real" and not hw_cfg_path:
            hw_cfg_path = str(
                Path(tbp.__file__).resolve().parent / "fafu_robot_python" / "robot.cfg"
            )
        self.arm_hw = tbp.PantheraArmController(
            control_mode=args.control_mode,
            cfg_path=hw_cfg_path,
            has_gripper=self.has_gripper,
            gripper_motor_id=args.hw_gripper_id,
            hw_port=hw_port,
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
            gripper_grasp_force=(
                None
                if args.gripper_grasp_force is None or args.gripper_grasp_force < 0
                else args.gripper_grasp_force
            ),
            gripper_grasp_vel=args.gripper_grasp_vel,
            gripper_grasp_timeout=args.gripper_grasp_timeout,
        )

        self.jaw_open_width = None
        self.jaw_close_width = None
        if (
            getattr(self.robot_model, "end_effector", None) is not None
            and hasattr(self.robot_model.end_effector, "jaw_range")
        ):
            jaw_min = float(self.robot_model.end_effector.jaw_range[0])
            jaw_max = float(self.robot_model.end_effector.jaw_range[1])
            self.jaw_open_width = (
                jaw_max if args.jaw_open_width < 0 else float(np.clip(args.jaw_open_width, jaw_min, jaw_max))
            )
            self.jaw_close_width = float(np.clip(args.jaw_close_width, jaw_min, jaw_max))

        self.clutch_active = False
        self.grip_engage_thr = float(args.xr_grip_threshold)
        self.grip_release_thr = float(args.xr_grip_release_threshold)
        self.xr_anchor_pos = np.zeros(3)
        self.xr_anchor_rot = np.eye(3)
        self.tcp_target_pos = self.tcp_pos_anchor.copy()
        self.tcp_target_rot = self.tcp_rot_anchor.copy()
        self.tcp_sent_pos = self.tcp_pos_anchor.copy()
        self.tcp_sent_rot = self.tcp_rot_anchor.copy()

        self.ik_try_count = 0
        self.ik_success_count = 0
        self.collision_reject_count = 0
        self.collision_bypass_count = 0
        self.ik_jump_reject_count = 0
        self.ik_limit_reject_count = 0
        self.tcp_down_stall_warn_count = 0
        self.ik_zero_streak = 0
        self.ik_adapt_level = 0
        self.ik_adapt_recover_count = 0
        self.last_ik_adapt_log_t = 0.0
        self.last_ik_stat_print_t = time.time()
        self.last_tcp_diag_print_t = 0.0
        self.last_joint_limit_log_t = 0.0
        self.last_ik_target_pos: Optional[np.ndarray] = None
        self.last_ik_target_rot: Optional[np.ndarray] = None
        self.ik_target_pos_eps_m = 5e-5
        self.ik_target_rot_eps_rad = math.radians(0.08)
        self.prev_a_pressed = False
        self.prev_b_pressed = False
        self.prev_home_pressed = False
        self._clutch_block_warned = False
        self._clutch_release_t = 0.0
        self._clutch_cooldown_warned = False
        self._grip_engage_streak = 0
        self._grip_release_streak = 0
        self._home_stop_logged = False
        self._last_home_progress_log_t = 0.0
        self._q_home_fixed: Optional[np.ndarray] = None
        self.preset_only = (label == "LEFT") and not bool(
            getattr(args, "left_full_teleop", False)
        )
        self._q_home_ready: Optional[np.ndarray] = None
        self._q_raise_target: Optional[np.ndarray] = None
        self._q_extend_target: Optional[np.ndarray] = None
        self._preset_active: Optional[str] = None
        self._preset_chain: Optional[str] = None
        self._last_preset_progress_log_t = 0.0

        self.workspace_clamp_active = bool(args.workspace_clamp)
        self.collision_filter_strict = bool(args.self_collision_filter)
        ik_jump_thr_rad = math.radians(max(float(args.ik_max_joint_jump_deg), 0.0))
        self.ik_base_jump_thr_rad = ik_jump_thr_rad
        self.ik_effective_jump_thr_rad = ik_jump_thr_rad
        self.ik_effective_rot_scale = float(args.rot_scale)

        self._log(
            f"port={hw_port}, home={home_button}, "
            f"gripper={'on' if self.has_gripper else 'off'} "
            f"({gripper_open_label}=open {gripper_close_label}=close), "
            f"joint_sign={self.joint_sign.astype(int).tolist()}"
            + (", mode=preset(X=extend,Y=home)" if self.preset_only else "")
        )

    def _log(self, msg: str) -> None:
        print(f"[{self.label}] {msg}")

    def _hw_to_model(self, q_hw: np.ndarray) -> np.ndarray:
        return np.asarray(q_hw, dtype=float) * self.joint_sign

    def _model_to_hw(self, q_model: np.ndarray) -> np.ndarray:
        return np.asarray(q_model, dtype=float) * self.joint_sign

    def _sync_q_from_hw(self, *, resync_servo: bool = False, tag: str = "SYNC") -> np.ndarray:
        try:
            q_hw = np.asarray(self.arm_hw.get_joint_values(), dtype=float)
            if q_hw.shape != self.q_seed.shape or not np.all(np.isfinite(q_hw)):
                self._log(f"[{tag}][WARN] hardware joint read failed shape/finite check.")
                return self.q_seed
            self.q_seed = self._hw_to_model(q_hw).copy()
            self.robot_model.goto_given_conf(self.q_seed)
            if resync_servo:
                self.arm_hw.resync_servo_to_measured()
            return self.q_seed
        except Exception as exc:
            self._log(f"[{tag}][WARN] could not sync q_seed from hardware: {exc}")
            return self.q_seed

    def _align_tcp_chain_to_q_seed(self) -> None:
        self.robot_model.goto_given_conf(self.q_seed)
        self.tcp_sent_pos, self.tcp_sent_rot = self.robot_model.fk(self.q_seed)
        self.tcp_target_pos = self.tcp_sent_pos.copy()
        self.tcp_target_rot = self.tcp_sent_rot.copy()

    def _recover_from_servo_reject(self, q_cmd_model: np.ndarray) -> None:
        if bool(getattr(self.args, "servo_reject_hold_command", True)):
            self.arm_hw.resync_servo_to_command(self._model_to_hw(self.q_seed))
            return
        q_cmd_model = self._clamp_left_j3_q(q_cmd_model)
        recover_step = max(self.max_joint_step_rad * 0.5, math.radians(1.0))
        try:
            q_hw_meas = np.asarray(self.arm_hw.get_joint_values(), dtype=float)
            q_meas = self._hw_to_model(q_hw_meas)
            self.q_seed = tbp.limit_joint_step(q_meas, q_cmd_model, recover_step)
            self.q_seed = self._clamp_left_j3_q(self.q_seed)
        except Exception:
            self._sync_q_from_hw(resync_servo=False, tag="SERVO_REJ")
        self.arm_hw.resync_servo_to_command(self._model_to_hw(self.q_seed))
        self._align_tcp_chain_to_q_seed()

    def _send_q(self, q_model: np.ndarray, dt: float, **kwargs) -> bool:
        q_model = self._clamp_left_j3_q(q_model)
        q_hw = self._model_to_hw(q_model)
        ok = self.arm_hw.send_joint_positions(q_hw, dt, **kwargs)
        if not ok:
            self._recover_from_servo_reject(q_model)
        return ok

    def _send_q_follow(self, q_model: np.ndarray, dt: float, xr, xr_rot: np.ndarray, **kwargs) -> bool:
        ok = self._send_q(q_model, dt, **kwargs)
        if not ok:
            self.tcp_pos_anchor = self.tcp_sent_pos.copy()
            self.tcp_rot_anchor = self.tcp_sent_rot.copy()
            self.xr_anchor_pos = xr.controller_pos_vr.copy()
            self.xr_anchor_rot = xr_rot.copy()
            self.last_ik_target_pos = None
            self.last_ik_target_rot = None
        return ok

    def _feed_idle_servo(self) -> None:
        """Hold last commanded pose on servo_j; do not snap to measured (avoids sag on release)."""
        q_hw = self._model_to_hw(self.q_seed)
        self.arm_hw.feed_idle_servo_hold(q_hold=q_hw)

    def _model_limit_margin(self, q_model: Optional[np.ndarray] = None) -> Optional[dict]:
        if self.arm_hw._arm_real is None:
            return None
        if not hasattr(self.arm_hw._arm_real, "get_limit"):
            return None
        try:
            joint_ids = list(self.arm_hw._arm_real.joint_motor_ids)
        except Exception:
            return None
        if not joint_ids:
            return None
        q_model = np.asarray(self.q_seed if q_model is None else q_model, dtype=float)
        if q_model.size != len(joint_ids):
            return None

        margins = []
        min_margin_deg = float("inf")
        min_margin_joint = -1
        for i, mid in enumerate(joint_ids):
            lim = self.arm_hw._arm_real.get_limit(int(mid), is_radians=True)
            if lim is None:
                continue
            lo_hw, hi_hw = float(lim[0]), float(lim[1])
            sign = float(self.joint_sign[i])
            if sign >= 0.0:
                lo_m, hi_m = lo_hw, hi_hw
            else:
                lo_m, hi_m = -hi_hw, -lo_hw
            cur = float(q_model[i])
            d_lo = cur - lo_m
            d_hi = hi_m - cur
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

    def _limit_margin(self, q_model: Optional[np.ndarray] = None) -> Optional[dict]:
        return self._model_limit_margin(q_model)

    def _within_limits(self, q_model: np.ndarray, min_margin_deg: float = 0.0) -> bool:
        info = self._model_limit_margin(q_model)
        if info is None:
            return True
        return float(info["min_margin_deg"]) >= float(min_margin_deg)

    def _effective_home_step_rad(self) -> float:
        base = self.home_joint_step_rad
        if base <= 0.0:
            return base
        lim = self._limit_margin()
        if lim is None or float(lim["min_margin_deg"]) >= 0.0:
            return base
        # Outside soft limits: speed up homing (up to 4x) so teleop can start sooner.
        boost = min(max(-float(lim["min_margin_deg"]) / 25.0, 1.0), 4.0)
        return base * boost

    def _resolve_home_target(self) -> np.ndarray:
        home_target = self.q_home if self.home_to_zero else self._q_home_fixed
        if home_target is None:
            home_target = self.q_home
        return self._clamp_q_to_soft_limits(home_target, min_margin_deg=5.0)

    def _home_max_error_deg(self, q_model: np.ndarray, home_target: np.ndarray) -> float:
        dq = np.abs(np.asarray(q_model, dtype=float) - np.asarray(home_target, dtype=float))
        return float(np.max(np.rad2deg(dq)))

    def _estimate_home_seconds(self) -> Optional[float]:
        lim = self._limit_margin()
        step = self._effective_home_step_rad()
        if lim is None or step <= 0.0 or float(lim["min_margin_deg"]) >= 0.0:
            return None
        worst_deg = max(-float(lim["min_margin_deg"]), 0.0)
        per_joint_deg = math.degrees(step)
        if per_joint_deg <= 1e-6:
            return None
        return worst_deg / per_joint_deg / max(float(self.args.loop_hz), 1.0)

    def _log_limit_state(self, *, tag: str = "INIT") -> None:
        if self.args.control_mode != "real" or self.args.dry_run:
            return
        try:
            q_hw = np.asarray(self.arm_hw.get_joint_values(), dtype=float)
            self._log(f"[{tag}] q_hw (deg) = {np.round(np.rad2deg(q_hw), 3)}")
        except Exception:
            pass
        self._log(f"[{tag}] q_model (deg) = {np.round(np.rad2deg(self.q_seed), 3)}")
        lim = self._limit_margin()
        if lim is None:
            return
        j_idx = int(lim["min_margin_joint"])
        margin = float(lim["min_margin_deg"])
        if margin < 0.0:
            eta = self._estimate_home_seconds()
            eta_text = f"，预计归位约 {eta:.0f}s" if eta is not None else ""
            self._log(
                f"[JOINT][WARN] J{j_idx} 超出软限位 margin={margin:.1f}deg{eta_text}。"
                f"请先长按 {self.home_button} 归位，进入限位后再按 grip 跟随。"
            )
            if j_idx == 3 and float(self.joint_sign[2]) > 0.0:
                self._log(
                    "[JOINT][HINT] 若 J3 方向反了，可试 --left-joint-sign 1 1 -1 1 1 1"
                )
        else:
            self._log(f"[JOINT] limit OK, nearest J{j_idx} margin={margin:.1f}deg")
            min_thr = float(getattr(self.args, "clutch_min_margin_deg", 8.0))
            if margin < min_thr and tag == "INIT":
                self._log(
                    f"[JOINT][WARN] J{j_idx} 余量仅 {margin:.1f}deg (<{min_thr:.0f}deg)，"
                    f"小幅 VR 也可能乱动。请先长按 {self.home_button} 归位到中间姿态。"
                )

    def _joint_model_limits_rad(self, joint_idx: int) -> Optional[tuple[float, float]]:
        if self.arm_hw._arm_real is None or not hasattr(self.arm_hw._arm_real, "get_limit"):
            return None
        try:
            joint_ids = list(self.arm_hw._arm_real.joint_motor_ids)
        except Exception:
            return None
        if joint_idx < 0 or joint_idx >= len(joint_ids):
            return None
        lim = self.arm_hw._arm_real.get_limit(int(joint_ids[joint_idx]), is_radians=True)
        if lim is None:
            return None
        lo_hw, hi_hw = float(lim[0]), float(lim[1])
        sign = float(self.joint_sign[joint_idx])
        if sign >= 0.0:
            return lo_hw, hi_hw
        return -hi_hw, -lo_hw

    def _clamp_left_j3_q(
        self,
        q_model: np.ndarray,
        *,
        min_margin_deg: float = 5.0,
    ) -> np.ndarray:
        if self.label != "LEFT":
            return np.asarray(q_model, dtype=float)
        lim = self._joint_model_limits_rad(2)
        if lim is None:
            return np.asarray(q_model, dtype=float)
        lo_m, _hi_m = lim
        floor = lo_m + math.radians(max(float(min_margin_deg), 0.0))
        q_out = np.asarray(q_model, dtype=float).copy()
        if q_out[2] < floor:
            q_out[2] = floor
        return q_out

    def _clamp_q_to_soft_limits(
        self,
        q_model: np.ndarray,
        *,
        min_margin_deg: float = 5.0,
    ) -> np.ndarray:
        if self.arm_hw._arm_real is None or not hasattr(self.arm_hw._arm_real, "get_limit"):
            return np.asarray(q_model, dtype=float)
        try:
            joint_ids = list(self.arm_hw._arm_real.joint_motor_ids)
        except Exception:
            return np.asarray(q_model, dtype=float)
        q_out = np.asarray(q_model, dtype=float).copy()
        if q_out.size != len(joint_ids):
            return q_out
        margin_rad = math.radians(max(float(min_margin_deg), 0.0))
        for i, mid in enumerate(joint_ids):
            lim = self.arm_hw._arm_real.get_limit(int(mid), is_radians=True)
            if lim is None:
                continue
            lo_hw, hi_hw = float(lim[0]), float(lim[1])
            sign = float(self.joint_sign[i])
            if sign >= 0.0:
                lo_m, hi_m = lo_hw, hi_hw
            else:
                lo_m, hi_m = -hi_hw, -lo_hw
            lo_safe = lo_m + margin_rad
            hi_safe = hi_m - margin_rad
            if lo_safe <= hi_safe:
                q_out[i] = float(np.clip(q_out[i], lo_safe, hi_safe))
        if self.label == "LEFT":
            q_out = self._clamp_left_j3_q(q_out, min_margin_deg=min_margin_deg)
        return q_out

    def _j3_margin_deg(self) -> Optional[float]:
        lim = self._limit_margin()
        if lim is None:
            return None
        for j_idx, _d_lo, _d_hi, near_deg in lim["margins"]:
            if int(j_idx) == 3:
                return float(near_deg)
        return None

    def _prepare_left_j3_headroom(self) -> None:
        if self.label != "LEFT" or self.home_to_zero or self.preset_only:
            return
        headroom_deg = max(float(getattr(self.args, "left_j3_headroom_deg", 35.0)), 0.0)
        if headroom_deg <= 0.0:
            return
        j3_margin = self._j3_margin_deg()
        if j3_margin is None or j3_margin >= headroom_deg:
            return

        delta_deg = headroom_deg - j3_margin
        j3_idx = 2
        target_j3 = float(self.q_seed[j3_idx]) + math.radians(delta_deg)
        if self._q_home_fixed is not None:
            self._q_home_fixed[j3_idx] = target_j3
            self.q_home = self._q_home_fixed.copy()
        self._log(
            f"[J3] 折叠余量仅 {j3_margin:.1f}deg，"
            f"就绪/归位 J3 上调 {delta_deg:.1f}deg -> {math.degrees(target_j3):.1f}deg"
        )

        if bool(getattr(self.args, "no_left_auto_j3_unfold", False)):
            self._log(
                f"[J3] 请长按 {self.home_button} 展开约 {delta_deg:.0f}deg 后再 grip；"
                "或去掉 --no-left-auto-j3-unfold 启用上电自动展开。"
            )
            return

        step_rad = max(self.home_joint_step_rad, math.radians(0.75))
        steps = max(int(math.ceil(delta_deg / math.degrees(step_rad))), 1)
        self._log(f"[J3] 自动展开余量 ({steps} steps, {math.degrees(step_rad):.2f}deg/step)...")
        for _ in range(steps):
            self.q_seed[j3_idx] = min(float(self.q_seed[j3_idx]) + step_rad, target_j3)
            self.robot_model.goto_given_conf(self.q_seed)
            self._send_q(self.q_seed, self.dt, cmd_max_step_rad=step_rad, skip_cmd_lp=True)
            time.sleep(self.dt)
        self._sync_q_from_hw(resync_servo=True, tag="J3_READY")
        if self._q_home_fixed is not None:
            self._q_home_fixed = self.q_seed.copy()
            self.q_home = self._q_home_fixed.copy()
        self.tcp_pos_anchor, self.tcp_rot_anchor = self.robot_model.fk(self.q_seed)
        self.tcp_target_pos = self.tcp_pos_anchor.copy()
        self.tcp_target_rot = self.tcp_rot_anchor.copy()
        self.tcp_sent_pos = self.tcp_pos_anchor.copy()
        self.tcp_sent_rot = self.tcp_rot_anchor.copy()
        j3_after = self._j3_margin_deg()
        if j3_after is not None:
            self._log(f"[J3] 就绪 margin={j3_after:.1f}deg")

    def _setup_preset_targets(self) -> None:
        if not self.preset_only:
            return
        self._q_home_ready = np.asarray(self.q_seed, dtype=float).copy()
        if self._q_home_fixed is not None:
            self._q_home_ready = np.asarray(self._q_home_fixed, dtype=float).copy()
        self._q_raise_target, self._q_extend_target = self._compute_preset_pose_targets(
            self._q_home_ready
        )
        self._log(
            "[PRESET] X=先抬高再前伸, Y=归位; grip/trigger 无功能"
        )
        self._log(
            f"[PRESET] home q (deg) = {np.round(np.rad2deg(self._q_home_ready), 2)}"
        )
        self._log(
            f"[PRESET] raise q (deg) = {np.round(np.rad2deg(self._q_raise_target), 2)}"
        )
        self._log(
            f"[PRESET] extend q (deg) = {np.round(np.rad2deg(self._q_extend_target), 2)}"
        )
        try:
            p_home, _ = self.robot_model.fk(self._q_home_ready)
            p_ext, _ = self.robot_model.fk(self._q_extend_target)
            dp = np.asarray(p_ext, dtype=float) - np.asarray(p_home, dtype=float)
            self._log(
                f"[PRESET] extend tcp delta (m) = {np.round(dp, 4)} "
                "(sim FK 仅供参考; 真机方向以关节增量为准)"
            )
        except Exception:
            pass

    def _preset_fk_pos(self, q_model: np.ndarray) -> np.ndarray:
        self.robot_model.goto_given_conf(q_model)
        pos, _rot = self.robot_model.fk(q_model)
        return np.asarray(pos, dtype=float)

    def _preset_forward_axis_info(self) -> tuple[int, float, str]:
        raw = str(getattr(self.args, "left_preset_forward_axis", "neg_y")).strip().lower()
        mapping = {
            "pos_x": (0, 1.0, "+X"),
            "+x": (0, 1.0, "+X"),
            "x": (0, 1.0, "+X"),
            "neg_x": (0, -1.0, "-X"),
            "-x": (0, -1.0, "-X"),
            "pos_y": (1, 1.0, "+Y"),
            "+y": (1, 1.0, "+Y"),
            "y": (1, 1.0, "+Y"),
            "neg_y": (1, -1.0, "-Y"),
            "-y": (1, -1.0, "-Y"),
        }
        return mapping.get(raw, (1, -1.0, "-Y"))

    def _auto_calibrate_preset_pose_targets(
        self,
        q_home: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_home = np.asarray(q_home, dtype=float).copy()
        p0 = self._preset_fk_pos(q_home)
        lift_m = max(float(getattr(self.args, "left_preset_lift_m", 0.08)), 0.0)
        reach_m = max(float(getattr(self.args, "left_preset_reach_m", 0.10)), 0.0)

        best_raise = None
        for j2_deg in range(0, 32, 2):
            for j3_deg in range(0, 36, 2):
                delta = np.deg2rad([0.0, float(j2_deg), float(j3_deg), 0.0, 0.0, 0.0])
                q_try = self._apply_joint_deltas(q_home, delta)
                if not self._within_limits(q_try, min_margin_deg=-2.0):
                    continue
                dp = self._preset_fk_pos(q_try) - p0
                if lift_m > 0.0 and float(dp[2]) < lift_m:
                    continue
                score = float(dp[2]) - 0.9 * abs(float(dp[0])) - 0.9 * abs(float(dp[1]))
                if best_raise is None or score > best_raise[0]:
                    best_raise = (score, q_try, j2_deg, j3_deg)

        if best_raise is None:
            q_raise = self._apply_joint_deltas(
                q_home, np.deg2rad([0.0, 28.0, 26.0, 0.0, 0.0, 0.0])
            )
            self._log("[PRESET][WARN] auto raise 未找到合适目标，使用默认 J2+28 J3+26")
        else:
            q_raise = best_raise[1]
            if int(best_raise[3]) == 0:
                self._log(f"[PRESET] auto raise: J2+{best_raise[2]:.0f}deg")
            else:
                self._log(
                    f"[PRESET] auto raise: J2+{best_raise[2]:.0f}deg, J3+{best_raise[3]:.0f}deg"
                )

        p_raise = self._preset_fk_pos(q_raise)
        best_ext = None
        for j4_deg in range(6, 34, 2):
            delta = np.deg2rad([0.0, 0.0, 0.0, float(j4_deg), 0.0, 0.0])
            q_try = self._apply_joint_deltas(q_raise, delta)
            if not self._within_limits(q_try, min_margin_deg=-2.0):
                continue
            dpe = self._preset_fk_pos(q_try) - p_raise
            horiz = math.hypot(float(dpe[0]), float(dpe[1]))
            if reach_m > 0.0 and horiz < reach_m * 0.4:
                continue
            score = float(dpe[2]) + 0.25 * horiz - 0.4 * max(-float(dpe[2]), 0.0)
            if best_ext is None or score > best_ext[0]:
                best_ext = (score, q_try, j4_deg, dpe, horiz)

        if best_ext is None:
            q_extend = self._apply_joint_deltas(
                q_raise, np.deg2rad([0.0, 0.0, 0.0, 28.0, 0.0, 0.0])
            )
            self._log("[PRESET][WARN] auto extend 未找到合适目标，使用默认 J4+28 (腕上抬)")
        else:
            q_extend = best_ext[1]
            self._log(
                f"[PRESET] auto extend: J4+{best_ext[2]:.0f}deg, "
                f"dZ={float(best_ext[3][2]):+.3f}m, reach_xy={best_ext[4]:.3f}m"
            )
        return q_raise, q_extend

    def _apply_joint_deltas(self, q_base: np.ndarray, delta_rad: np.ndarray) -> np.ndarray:
        q_out = np.asarray(q_base, dtype=float).copy()
        q_out += np.asarray(delta_rad, dtype=float)
        return self._clamp_left_j3_q(q_out)

    def _compute_preset_pose_targets(
        self,
        q_home: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        args = self.args
        q_home = np.asarray(q_home, dtype=float).copy()
        use_auto = bool(getattr(args, "left_preset_auto_calibrate", False))
        if bool(getattr(args, "no_left_preset_auto_calibrate", False)):
            use_auto = False
        if use_auto:
            return self._auto_calibrate_preset_pose_targets(q_home)

        raise_deg = getattr(args, "left_preset_raise_deltas_deg", None)
        extend_deg = getattr(args, "left_preset_extend_deltas_deg", None)
        raise_delta = _parse_preset_joint_deltas(
            "left preset raise",
            raise_deg,
            default=[0.0, 28.0, 26.0, 0.0, 0.0, 0.0],
        )
        extend_delta = _parse_preset_joint_deltas(
            "left preset extend",
            extend_deg,
            default=[0.0, 0.0, 0.0, 28.0, 0.0, 0.0],
        )
        q_raise = self._apply_joint_deltas(q_home, raise_delta)
        q_extend = self._apply_joint_deltas(q_raise, extend_delta)
        self._log(
            "[PRESET] 关节 preset: raise "
            f"{np.round(np.rad2deg(raise_delta), 1).tolist()}, extend "
            f"{np.round(np.rad2deg(extend_delta), 1).tolist()}"
        )
        if not self._within_limits(q_raise, min_margin_deg=-3.0):
            self._log("[PRESET][WARN] raise 目标接近限位，可按需调 --left-preset-raise-deltas-deg")
        if not self._within_limits(q_extend, min_margin_deg=-3.0):
            self._log("[PRESET][WARN] extend 目标接近限位，可按需调 --left-preset-extend-deltas-deg")
        return q_raise, q_extend

    def _preset_step_rad(self) -> float:
        step_deg = max(float(getattr(self.args, "left_preset_step_deg", 0.5)), 0.05)
        return math.radians(step_deg)

    def _preset_target_reached(self, target: np.ndarray, tol_deg: float = 0.6) -> bool:
        dq = np.abs(np.asarray(self.q_seed, dtype=float) - np.asarray(target, dtype=float))
        return float(np.max(dq)) <= math.radians(tol_deg)

    def step_preset(self, xr: tbp.XRState) -> tuple[float, float]:
        """Left preset mode: X extend, Y home; ignore grip/trigger."""
        args = self.args
        perf_ik = 0.0
        perf_hw = 0.0
        if self._q_home_ready is None or self._q_raise_target is None or self._q_extend_target is None:
            return perf_ik, perf_hw

        if xr.button_a_pressed and not self.prev_a_pressed:
            pose_tol_deg = 2.0
            at_extend = self._preset_target_reached(self._q_extend_target, tol_deg=pose_tol_deg)
            at_raise = self._preset_target_reached(self._q_raise_target, tol_deg=pose_tol_deg)
            if at_extend:
                self._preset_active = "home"
                self._preset_chain = "extend_seq"
                self._log("X -> 先归位再抬肘前伸")
            elif at_raise:
                self._preset_active = "extend"
                self._preset_chain = None
                self._log("X -> 继续前伸")
            else:
                self._preset_active = "raise"
                self._preset_chain = None
                self._log("X -> 抬高并前伸（两阶段）")
        if xr.button_b_pressed and not self.prev_b_pressed:
            self._preset_active = "home"
            self._preset_chain = None
            self._log("Y -> 归位")
        self.prev_a_pressed = xr.button_a_pressed
        self.prev_b_pressed = xr.button_b_pressed

        step_rad = self._preset_step_rad()
        if self._preset_active == "raise":
            target = self._q_raise_target
            phase_label = "抬高"
        elif self._preset_active == "extend":
            target = self._q_extend_target
            phase_label = "前伸"
        elif self._preset_active == "home":
            target = self._q_home_ready
            phase_label = "归位"
        else:
            if (not args.dry_run) and self.arm_hw.is_servo_streaming():
                hw_t0 = time.time()
                self._send_q(self.q_seed, self.dt)
                perf_hw += time.time() - hw_t0
            return perf_ik, perf_hw

        self.q_seed = tbp.limit_joint_step(self.q_seed, target, step_rad)
        self.q_seed = self._clamp_left_j3_q(self.q_seed)
        self.robot_model.goto_given_conf(self.q_seed)
        if not args.dry_run:
            hw_t0 = time.time()
            cmd_step = step_rad if step_rad > 0.0 else None
            self._send_q(self.q_seed, self.dt, cmd_max_step_rad=cmd_step, skip_cmd_lp=True)
            perf_hw += time.time() - hw_t0

        if self._preset_target_reached(target):
            if self._preset_active == "raise":
                self._preset_active = "extend"
                self._log("[PRESET] 抬高完成，继续前伸...")
            elif self._preset_active == "extend":
                self._log("[PRESET] 前伸完成")
                self._preset_active = None
                self._preset_chain = None
            else:
                self._log("[PRESET] 归位完成")
                if self._preset_chain == "extend_seq":
                    self._preset_chain = None
                    self._preset_active = "raise"
                    self._log("[PRESET] 继续抬高...")
                else:
                    self._preset_active = None
        else:
            now_t = time.time()
            if now_t - self._last_preset_progress_log_t >= 1.0:
                remain_deg = float(
                    np.max(np.abs(np.asarray(target, dtype=float) - np.asarray(self.q_seed, dtype=float)))
                )
                self._log(
                    f"[PRESET] {phase_label}中... remain_max={math.degrees(remain_deg):.1f}deg"
                )
                self._last_preset_progress_log_t = now_t
        return perf_ik, perf_hw

    def enable(self) -> None:
        self.arm_hw.enable()
        if self.args.control_mode == "real" and not self.args.dry_run:
            self._sync_q_from_hw(resync_servo=True, tag="INIT")
            self.tcp_pos_anchor, self.tcp_rot_anchor = self.robot_model.fk(self.q_seed)
            self.tcp_target_pos = self.tcp_pos_anchor.copy()
            self.tcp_target_rot = self.tcp_rot_anchor.copy()
            self.tcp_sent_pos = self.tcp_pos_anchor.copy()
            self.tcp_sent_rot = self.tcp_rot_anchor.copy()
            if not self.home_to_zero and not self.preset_only:
                self._q_home_fixed = self.q_seed.copy()
                self.q_home = self._q_home_fixed.copy()
                self._log(
                    "[HOME] 归位目标=启动姿态（固定快照；LEFT_TRIGGER 回到上电姿态，"
                    "需全零归位请加 --left-home-to-zero）"
                )
            elif not self.home_to_zero:
                self._q_home_fixed = self.q_seed.copy()
                self.q_home = self._q_home_fixed.copy()
            elif self.home_to_zero and self.label == "RIGHT":
                arrive_deg = float(getattr(self.args, "home_arrive_deg", 3.0))
                home_target = self._resolve_home_target()
                tcp_home, _ = self.robot_model.fk(home_target)
                self._log(
                    f"[HOME] 归位目标 q (deg) = {np.round(np.rad2deg(home_target), 1)}，"
                    f"tcp≈{np.round(tcp_home, 3)} m；误差<{arrive_deg:.0f}° 即完成"
                )
                if np.allclose(home_target, 0.0, atol=1e-6):
                    self._log(
                        "[HOME][HINT] 全零归位时 J2 距软限位下限仅约 12° margin，"
                        "若跟手易跳变可加 --right-home-ready-pose"
                    )
            if self.preset_only:
                self._log("[PRESET] 上电保持折叠姿态，不自动展开（按 Y 归位 / 按 X 抬肘前伸）")
            else:
                self._prepare_left_j3_headroom()
            self._setup_preset_targets()
            self._log_limit_state(tag="INIT")

    def emergency_stop(self) -> None:
        self.arm_hw.emergency_stop()

    def disable(self) -> None:
        self.arm_hw.disable()

    def pop_servo_diag(self):
        return self.arm_hw.pop_servo_diag()

    def _update_ik_adapt_settings(self, *, reason: str) -> None:
        args = self.args
        if self.ik_base_jump_thr_rad > 0.0:
            boost_rad = math.radians(
                max(float(args.ik_adapt_jump_boost_deg), 0.0) * self.ik_adapt_level
            )
            self.ik_effective_jump_thr_rad = self.ik_base_jump_thr_rad + boost_rad
        else:
            self.ik_effective_jump_thr_rad = 0.0
        if self.ik_adapt_level <= 0:
            self.ik_effective_rot_scale = float(args.rot_scale)
        else:
            factor = float(np.clip(float(args.ik_adapt_rot_scale_factor), 0.05, 1.0))
            self.ik_effective_rot_scale = max(
                float(args.rot_scale) * (factor ** self.ik_adapt_level),
                float(args.ik_adapt_rot_scale_min),
            )
        now_adapt_t = time.time()
        if now_adapt_t - self.last_ik_adapt_log_t >= 0.5:
            if self.ik_adapt_level > 0:
                self._log(
                    f"[IK][ADAPT] {reason}: level={self.ik_adapt_level}, "
                    f"jump_thr={np.rad2deg(self.ik_effective_jump_thr_rad):.1f}deg, "
                    f"rot_scale={self.ik_effective_rot_scale:.2f}"
                )
            elif reason != "init":
                self._log(
                    f"[IK][ADAPT] {reason}: restored base "
                    f"jump_thr={np.rad2deg(self.ik_effective_jump_thr_rad):.1f}deg, "
                    f"rot_scale={self.ik_effective_rot_scale:.2f}"
                )
            self.last_ik_adapt_log_t = now_adapt_t

    def _filter_ik_candidate(self, q_try: Optional[np.ndarray]) -> Optional[np.ndarray]:
        args = self.args
        if q_try is None:
            return None
        q_try = tbp.align_joint_solution_to_seed(np.asarray(q_try, dtype=float), self.q_seed)
        if self.ik_effective_jump_thr_rad > 0.0:
            dq_abs = np.abs(q_try - self.q_seed)
            dq_max = float(np.max(dq_abs))
            if dq_max > self.ik_effective_jump_thr_rad:
                self.ik_jump_reject_count += 1
                return None
        q_try = self._clamp_left_j3_q(q_try)
        if args.control_mode == "real" and not args.dry_run:
            margin_thr = -3.0 if self.label == "LEFT" else 0.0
            if not self._within_limits(q_try, min_margin_deg=margin_thr):
                self.ik_limit_reject_count += 1
                if self.ik_limit_reject_count == 1 or self.ik_limit_reject_count % 20 == 0:
                    lim_bad = self._limit_margin(q_try)
                    if lim_bad is not None:
                        self._log(
                            f"[IK][LIMIT] reject: J{lim_bad['min_margin_joint']} "
                            f"margin={lim_bad['min_margin_deg']:.1f}deg "
                            f"(total_reject={self.ik_limit_reject_count})"
                        )
                return None
        if args.self_collision_filter and self.robot_model.cc is not None:
            self.robot_model.goto_given_conf(q_try)
            candidate_collided, _ = self.robot_model.is_collided(toggle_contacts=True)
            self.robot_model.goto_given_conf(self.q_seed)
            if candidate_collided:
                if self.collision_filter_strict:
                    self.collision_reject_count += 1
                    return None
                self.collision_bypass_count += 1
            elif not self.collision_filter_strict:
                self.collision_filter_strict = True
        return tbp.limit_joint_step(self.q_seed, q_try, self.max_joint_step_rad)

    def step(self, xr: tbp.XRState, home_pressed: bool) -> tuple[float, float]:
        """Process one XR frame for this arm. Returns (ik_elapsed_s, hw_elapsed_s)."""
        args = self.args
        perf_ik = 0.0
        perf_hw = 0.0
        xr_rot = tbp.quat_xyzw_to_rotmat(xr.controller_quat_vr)

        if home_pressed and (not self.prev_home_pressed):
            self._log("[CTRL] home hold engaged")
            self._home_stop_logged = False
            # Re-anchor seed + servo baseline to actual hardware pose before homing
            # progression begins. Otherwise q_seed may carry stale clutch values and
            # the firmware rejects frames as max_lag is exceeded (manifests as
            # SERVO_DIAG cmd_step ~50deg + cascading [HW][SERVO] frame rejected).
            if args.control_mode == "real" and not args.dry_run:
                self._sync_q_from_hw(resync_servo=True, tag="HOME_ENGAGE")
                self._align_tcp_chain_to_q_seed()
        if (not home_pressed) and self.prev_home_pressed:
            self._log("[CTRL] home hold released")
            if args.control_mode == "real" and not args.dry_run:
                self.arm_hw.resync_servo_to_command(self._model_to_hw(self.q_seed))
                self._align_tcp_chain_to_q_seed()
        self.prev_home_pressed = home_pressed

        if xr.button_a_pressed and not self.prev_a_pressed:
            if not self.arm_hw._hw_has_gripper:
                self._log(f"{self.gripper_open_label} ignored (no gripper on this arm)")
            else:
                do_open = not bool(args.invert_gripper_buttons)
                if do_open:
                    if self.jaw_open_width is not None:
                        self.robot_model.change_jaw_width(self.jaw_open_width)
                    if not args.dry_run:
                        self.arm_hw.open_gripper()
                    self._log(f"{self.gripper_open_label} -> open gripper")
                else:
                    if self.jaw_close_width is not None:
                        self.robot_model.change_jaw_width(self.jaw_close_width)
                    if not args.dry_run:
                        self.arm_hw.close_gripper()
                    self._log(f"{self.gripper_open_label} -> close gripper")
        if xr.button_b_pressed and not self.prev_b_pressed:
            if not self.arm_hw._hw_has_gripper:
                self._log(f"{self.gripper_close_label} ignored (no gripper on this arm)")
            else:
                do_open = bool(args.invert_gripper_buttons)
                if do_open:
                    if self.jaw_open_width is not None:
                        self.robot_model.change_jaw_width(self.jaw_open_width)
                    if not args.dry_run:
                        self.arm_hw.open_gripper()
                    self._log(f"{self.gripper_close_label} -> open gripper")
                else:
                    if self.jaw_close_width is not None:
                        self.robot_model.change_jaw_width(self.jaw_close_width)
                    if not args.dry_run:
                        self.arm_hw.close_gripper()
                    self._log(f"{self.gripper_close_label} -> close gripper")
        self.prev_a_pressed = xr.button_a_pressed
        self.prev_b_pressed = xr.button_b_pressed

        if home_pressed:
            self.clutch_active = False
            self._clutch_block_warned = False
            self._grip_engage_streak = 0
            self._grip_release_streak = 0
            home_step_rad = self._effective_home_step_rad()
            home_arrive_deg = float(getattr(args, "home_arrive_deg", 3.0))
            home_target = self._resolve_home_target()
            remain_deg = self._home_max_error_deg(self.q_seed, home_target)
            if remain_deg <= home_arrive_deg:
                if not self._home_stop_logged:
                    lim_now = self._limit_margin()
                    margin_text = ""
                    if lim_now is not None:
                        j_idx = int(lim_now["min_margin_joint"])
                        margin = float(lim_now["min_margin_deg"])
                        margin_text = f", J{j_idx} margin={margin:.1f}deg"
                    try:
                        q_hw = np.asarray(self.arm_hw.get_joint_values(), dtype=float)
                        q_log = np.round(np.rad2deg(self._hw_to_model(q_hw)), 1)
                    except Exception:
                        q_log = np.round(np.rad2deg(self.q_seed), 1)
                    self._log(
                        f"[HOME] 归位完成 q (deg) = {q_log}{margin_text}，可松开 trigger"
                    )
                    self._home_stop_logged = True
                if not args.dry_run and self.arm_hw.is_servo_streaming():
                    hw_t0 = time.time()
                    self._feed_idle_servo()
                    perf_hw += time.time() - hw_t0
                return perf_ik, perf_hw

            self._home_stop_logged = False
            self.q_seed = tbp.limit_joint_step(self.q_seed, home_target, home_step_rad)
            self.robot_model.goto_given_conf(self.q_seed)
            self.tcp_pos_anchor, self.tcp_rot_anchor = self.robot_model.fk(self.q_seed)
            self.tcp_target_pos = self.tcp_pos_anchor.copy()
            self.tcp_target_rot = self.tcp_rot_anchor.copy()
            self.tcp_sent_pos = self.tcp_pos_anchor.copy()
            self.tcp_sent_rot = self.tcp_rot_anchor.copy()
            if not args.dry_run:
                hw_t0 = time.time()
                home_cmd_step = home_step_rad if home_step_rad > 0.0 else None
                self._send_q(
                    self.q_seed, self.dt, cmd_max_step_rad=home_cmd_step, skip_cmd_lp=True
                )
                perf_hw += time.time() - hw_t0
            now_home_t = time.time()
            if now_home_t - self._last_home_progress_log_t >= 1.0:
                remain_deg = self._home_max_error_deg(self.q_seed, home_target)
                lim = self._limit_margin()
                if lim is not None:
                    j_idx = int(lim["min_margin_joint"])
                    margin = float(lim["min_margin_deg"])
                    self._log(
                        f"[HOME] 归位中 remain_max={remain_deg:.1f}deg, "
                        f"J{j_idx} margin={margin:.1f}deg, "
                        f"step={math.degrees(home_step_rad):.2f}deg/cycle"
                    )
                self._last_home_progress_log_t = now_home_t
            return perf_ik, perf_hw

        engage_stable = max(int(getattr(args, "grip_engage_stable_frames", 8)), 1)
        release_stable = max(int(getattr(args, "grip_release_stable_frames", 4)), 1)
        if xr.grip_value >= self.grip_engage_thr:
            self._grip_engage_streak += 1
        else:
            self._grip_engage_streak = 0
        if xr.grip_value < self.grip_release_thr:
            self._grip_release_streak += 1
        else:
            self._grip_release_streak = 0
        if self.clutch_active:
            grip_follow = self._grip_release_streak < release_stable
        else:
            grip_follow = self._grip_engage_streak >= engage_stable
        if grip_follow and not self.clutch_active:
            cooldown_s = max(float(getattr(args, "clutch_reengage_cooldown_s", 0.35)), 0.0)
            since_rel = time.time() - self._clutch_release_t
            if self._clutch_release_t > 0.0 and since_rel < cooldown_s:
                if not self._clutch_cooldown_warned:
                    self._log(
                        f"[CTRL] grip 重按过快 ({since_rel:.2f}s<{cooldown_s:.2f}s)，"
                        "稍停稳再跟以免乱窜。"
                    )
                    self._clutch_cooldown_warned = True
            else:
                self._clutch_cooldown_warned = False
                lim_now = self._limit_margin()
                min_margin_thr = float(getattr(args, "clutch_min_margin_deg", 8.0))
                if (
                    args.control_mode == "real"
                    and not args.dry_run
                    and lim_now is not None
                    and float(lim_now["min_margin_deg"]) < min_margin_thr
                ):
                    if not self._clutch_block_warned:
                        j_idx = int(lim_now["min_margin_joint"])
                        margin = float(lim_now["min_margin_deg"])
                        self._log(
                            f"[CTRL] grip 跟随被阻止: J{j_idx} margin={margin:.1f}deg "
                            f"(需要>={min_margin_thr:.0f}deg)。"
                            f"请先长按 {self.home_button} 归位。"
                        )
                        self._clutch_block_warned = True
                else:
                    self._clutch_block_warned = False
                    self.clutch_active = True
                    if args.control_mode == "real" and not args.dry_run:
                        self._sync_q_from_hw(resync_servo=True, tag="CLUTCH")
                    self.xr_anchor_pos = xr.controller_pos_vr.copy()
                    self.xr_anchor_rot = xr_rot.copy()
                    self.tcp_pos_anchor, self.tcp_rot_anchor = self.robot_model.fk(self.q_seed)
                    self.tcp_target_pos, self.tcp_target_rot = self.robot_model.fk(self.q_seed)
                    self.tcp_sent_pos = self.tcp_target_pos.copy()
                    self.tcp_sent_rot = self.tcp_target_rot.copy()
                    self.last_ik_target_pos = None
                    self.last_ik_target_rot = None
                    self.ik_adapt_level = 0
                    self.ik_adapt_recover_count = 0
                    self.ik_zero_streak = 0
                    self.ik_limit_reject_count = 0
                    self._update_ik_adapt_settings(reason="clutch engage")
                    self._log(f"[CTRL] clutch engaged (grip={xr.grip_value:.2f})")
        elif (not grip_follow) and self.clutch_active:
            self.clutch_active = False
            self._clutch_release_t = time.time()
            self._log(f"[CTRL] clutch released (grip={xr.grip_value:.2f})")
            if args.control_mode == "real" and not args.dry_run:
                # Hold last IK command; measured is often lower under gravity.
                q_hw = self._model_to_hw(self.q_seed)
                self.arm_hw.resync_servo_to_command(q_hw)
                self._align_tcp_chain_to_q_seed()

        if (
            (not args.dry_run)
            and (not self.clutch_active)
            and (not home_pressed)
            and self.arm_hw.is_servo_streaming()
        ):
            hw_t0 = time.time()
            self._feed_idle_servo()
            perf_hw += time.time() - hw_t0
            return perf_ik, perf_hw

        if not self.clutch_active:
            return perf_ik, perf_hw

        dpos_vr = (xr.controller_pos_vr - self.xr_anchor_pos) * args.pos_scale
        dpos_base = self.r_vr_fix @ (self.r_br @ dpos_vr)
        if args.max_delta_pos > 0:
            dnorm = np.linalg.norm(dpos_base)
            if dnorm > args.max_delta_pos and dnorm > 1e-12:
                dpos_base = dpos_base / dnorm * args.max_delta_pos

        dR_vr = xr_rot @ self.xr_anchor_rot.T
        dR_base = self.r_vr_fix @ (self.r_br @ dR_vr @ self.r_br.T) @ self.r_vr_fix.T
        dR_base = tbp.scale_rotation_delta_axes(dR_base, np.asarray(args.rot_axis_gain, dtype=float))
        rot_scale_eff = self.ik_effective_rot_scale
        if args.control_mode == "real" and not args.dry_run:
            lim_rot = self._limit_margin()
            j6_margin_thr = max(float(args.ik_adapt_j6_margin_deg), 0.0)
            if (
                lim_rot is not None
                and int(lim_rot["min_margin_joint"]) == 6
                and float(lim_rot["min_margin_deg"]) < j6_margin_thr
            ):
                rot_scale_eff *= max(float(lim_rot["min_margin_deg"]) / j6_margin_thr, 0.15)
            if self.label == "LEFT":
                j3_margin = self._j3_margin_deg()
                j3_headroom = max(float(getattr(args, "left_j3_headroom_deg", 35.0)), 1.0)
                if j3_margin is not None and j3_margin < j3_headroom:
                    j3_scale = max(j3_margin / j3_headroom, 0.12)
                    dpos_base = dpos_base * j3_scale
                    rot_scale_eff *= j3_scale
        dR_base = tbp.scale_rotation_delta(dR_base, rot_scale_eff)

        raw_tgt_pos = self.tcp_pos_anchor + dpos_base + self.t_br
        if args.control_mode == "real" and not args.dry_run:
            lim_now = self._limit_margin()
            if (
                lim_now is not None
                and float(lim_now["min_margin_deg"]) < 0.0
                and float(raw_tgt_pos[2]) < float(self.tcp_sent_pos[2]) - 1e-4
            ):
                self.tcp_down_stall_warn_count += 1
                raw_tgt_pos[2] = float(self.tcp_sent_pos[2])
        if args.position_only:
            raw_tgt_rot = self.tcp_rot_anchor
        else:
            raw_tgt_rot = dR_base @ self.tcp_rot_anchor

        if self.workspace_clamp_active:
            raw_tgt_pos = np.array([
                np.clip(raw_tgt_pos[0], args.ws_x_min, args.ws_x_max),
                np.clip(raw_tgt_pos[1], args.ws_y_min, args.ws_y_max),
                np.clip(raw_tgt_pos[2], args.ws_z_min, args.ws_z_max),
            ])

        self.tcp_target_pos = tbp.limit_linear_step(self.tcp_sent_pos, raw_tgt_pos, self.max_lin_step)
        self.tcp_target_rot = tbp.limit_angular_step(self.tcp_sent_rot, raw_tgt_rot, self.max_ang_step)

        ik_target_moved = True
        if self.last_ik_target_pos is not None and self.last_ik_target_rot is not None:
            pos_delta = float(np.linalg.norm(self.tcp_target_pos - self.last_ik_target_pos))
            rot_delta = tbp.rotation_angle_rad(self.last_ik_target_rot, self.tcp_target_rot)
            ik_target_moved = (
                pos_delta >= self.ik_target_pos_eps_m
                or rot_delta >= self.ik_target_rot_eps_rad
            )
        self.last_ik_target_pos = self.tcp_target_pos.copy()
        self.last_ik_target_rot = self.tcp_target_rot.copy()

        if not ik_target_moved:
            if (not args.dry_run) and self.arm_hw.is_servo_streaming():
                hw_t0 = time.time()
                self._send_q_follow(self.q_seed, self.dt, xr, xr_rot)
                perf_hw += time.time() - hw_t0
            return perf_ik, perf_hw

        ik_t0 = time.time()
        self.ik_try_count += 1
        q_sol = None
        pos_try = self.tcp_target_pos
        rot_try = self.tcp_target_rot

        q_fast = self.robot_model.ik(
            tgt_pos=pos_try,
            tgt_rotmat=rot_try,
            seed_jnt_values=self.q_seed,
            toggle_dbg=False,
        )
        q_sol = self._filter_ik_candidate(q_fast)

        j3_margin_now = self._j3_margin_deg() if self.label == "LEFT" else None
        j3_skip_retry = (
            j3_margin_now is not None and j3_margin_now < 15.0
        )
        if q_sol is None and args.ik_retry and not j3_skip_retry:
            rot_candidates = [self.tcp_target_rot]
            if not args.position_only:
                for ang in (np.deg2rad(8), np.deg2rad(-8), np.deg2rad(16), np.deg2rad(-16)):
                    rot_candidates.append(
                        self.tcp_target_rot @ rm.rotmat_from_axangle(self.tcp_target_rot[:, 2], ang)
                    )
                if args.ik_allow_flip_candidate:
                    yaw_base = float(np.arctan2(self.tcp_target_rot[1, 0], self.tcp_target_rot[0, 0]))
                    for dyaw in (0.0, np.deg2rad(15), np.deg2rad(-15), np.deg2rad(30), np.deg2rad(-30)):
                        rot_candidates.append(rm.rotmat_from_euler(np.pi, 0.0, yaw_base + dyaw))

            pos_candidates = [self.tcp_target_pos]
            dpos = self.tcp_target_pos - self.tcp_sent_pos
            for s in (0.5, 0.25):
                pos_candidates.append(self.tcp_sent_pos + dpos * s)

            seed_candidates = [self.q_seed]
            max_attempts = int(args.ik_max_attempts_per_cycle)
            if self.label == "LEFT":
                if j3_margin_now is not None and j3_margin_now < 15.0:
                    max_attempts = min(max_attempts, 1)
                elif j3_margin_now is not None and j3_margin_now < 25.0:
                    max_attempts = min(max_attempts, 2)
            attempt_count = 0
            stop_search = False
            for seed_try in seed_candidates:
                if q_sol is not None or stop_search:
                    break
                for pos_c in pos_candidates:
                    if q_sol is not None or stop_search:
                        break
                    for rot_c in rot_candidates:
                        if max_attempts > 0 and attempt_count >= max_attempts:
                            stop_search = True
                            break
                        attempt_count += 1
                        q_try = self.robot_model.ik(
                            tgt_pos=pos_c,
                            tgt_rotmat=rot_c,
                            seed_jnt_values=seed_try,
                            toggle_dbg=False,
                        )
                        q_sol = self._filter_ik_candidate(q_try)
                        if q_sol is not None:
                            pos_try = pos_c
                            rot_try = rot_c
                            break

        if q_sol is not None:
            self.tcp_target_pos = pos_try
            self.tcp_target_rot = rot_try
            self.ik_success_count += 1
            self.ik_zero_streak = 0
            if args.ik_adaptive_ease and self.ik_adapt_level > 0:
                self.ik_adapt_recover_count += 1
                if self.ik_adapt_recover_count >= max(int(args.ik_adapt_recover_streak), 1):
                    self.ik_adapt_level = 0
                    self.ik_adapt_recover_count = 0
                    self._update_ik_adapt_settings(reason="recovered")
            self.q_seed = q_sol
            self.robot_model.goto_given_conf(self.q_seed)
            self.tcp_sent_pos, self.tcp_sent_rot = self.robot_model.fk(self.q_seed)
            if not args.dry_run:
                hw_t0 = time.time()
                self._send_q_follow(self.q_seed, self.dt, xr, xr_rot)
                perf_hw += time.time() - hw_t0
        else:
            self.ik_zero_streak += 1
            self.ik_adapt_recover_count = 0
            if args.ik_adaptive_ease and self.ik_zero_streak >= max(int(args.ik_adapt_fail_streak), 1):
                fail_step = max(int(args.ik_adapt_fail_streak), 1)
                new_level = min(self.ik_zero_streak // fail_step, 3)
                if new_level != self.ik_adapt_level:
                    self.ik_adapt_level = new_level
                    self._update_ik_adapt_settings(reason="consecutive fail")
            if (not args.dry_run) and self.arm_hw.is_servo_streaming():
                hw_t0 = time.time()
                self._send_q_follow(self.q_seed, self.dt, xr, xr_rot)
                perf_hw += time.time() - hw_t0

        perf_ik += time.time() - ik_t0

        if self.workspace_clamp_active and self.ik_success_count == 0 and self.ik_try_count >= 30:
            self.workspace_clamp_active = False
            self._log("[IK] workspace clamp auto-disabled (0 success in early stage).")

        now_t = time.time()
        if now_t - self.last_ik_stat_print_t >= 1.0:
            ratio = 0.0 if self.ik_try_count == 0 else (self.ik_success_count / self.ik_try_count * 100.0)
            self._log(
                f"[IK] success={self.ik_success_count}/{self.ik_try_count} ({ratio:.1f}%), "
                f"limit_reject={self.ik_limit_reject_count}"
            )
            self.last_ik_stat_print_t = now_t

        return perf_ik, perf_hw


def _init_xr_live(args) -> None:
    try:
        import xrobotoolkit_sdk as xrt
    except Exception as exc:
        raise RuntimeError(
            "Dual-arm live mode requires xrobotoolkit_sdk. "
            "Please run in XRoboToolkit Python environment."
        ) from exc
    tbp._XRT = xrt
    tbp._XR_GRIP_THRESHOLD = float(args.xr_grip_threshold)
    grip_release_thr = float(args.xr_grip_release_threshold)
    if grip_release_thr >= tbp._XR_GRIP_THRESHOLD:
        grip_release_thr = max(tbp._XR_GRIP_THRESHOLD - 0.08, 0.0)
        print(f"[XR][WARN] grip release threshold clamped -> {grip_release_thr:.2f}")
    args.xr_grip_release_threshold = grip_release_thr
    tbp._XR_ESTOP_BUTTON = args.xr_estop_button
    tbp._LAST_XR_TIMESTAMP_NS = None
    tbp._XR_LINK_READY = False
    tbp._LAST_XR_POLL_WALL_TS = 0.0
    tbp._XRT.init()
    left_mode = (
        "left X=extend Y=home (preset)"
        if not bool(getattr(args, "left_full_teleop", False))
        else "left grip->left arm"
    )
    print(
        "[XR] dual-arm live mode: right grip->right arm, "
        f"{left_mode}, estop={tbp._XR_ESTOP_BUTTON}, "
        "right home=RIGHT_TRIGGER"
        + (", left home=LEFT_TRIGGER" if bool(getattr(args, "left_full_teleop", False)) else "")
    )


def run_dual_arm_teleop(args) -> None:
    if args.visualize:
        print("[DUAL] visualization disabled in dual-arm mode (use --no-visualize).")
    if args.control_mode != "real":
        print("[DUAL][WARN] control_mode is not 'real'; both arms use the same backend setting.")

    _init_xr_live(args)

    dt = 1.0 / args.loop_hz
    max_lin_step = args.max_lin_speed * dt
    max_ang_step = math.radians(args.max_ang_speed_deg) * dt
    max_joint_step_rad = math.radians(max(float(args.max_joint_step_deg), 0.0))
    home_joint_step_rad = math.radians(max(float(args.home_joint_step_deg), 0.0))
    servo_max_step_rad = math.radians(max(float(args.servo_max_step_deg), 0.0))
    servo_max_lag_rad = math.radians(max(float(args.servo_max_lag_deg), 0.0))
    servo_cmd_max_step_rad = math.radians(max(float(args.servo_cmd_max_step_deg), 0.0))
    timeout_s = args.timeout_ms / 1000.0

    r_br = tbp.rotmat_from_euler_xyz_deg(tbp.CALIB_RPY_DEG_BR)
    t_br = tbp.CALIB_T_BR.copy()
    r_vr_fix = rm.rotmat_from_euler(0.0, 0.0, np.deg2rad(float(args.vr_yaw_offset_deg)))

    shared_kw = dict(
        args=args,
        dt=dt,
        r_br=r_br,
        t_br=t_br,
        r_vr_fix=r_vr_fix,
        max_lin_step=max_lin_step,
        max_ang_step=max_ang_step,
        max_joint_step_rad=max_joint_step_rad,
        home_joint_step_rad=home_joint_step_rad,
        servo_max_step_rad=servo_max_step_rad,
        servo_max_lag_rad=servo_max_lag_rad,
        servo_cmd_max_step_rad=servo_cmd_max_step_rad,
    )

    right_joint_sign = _parse_joint_sign(args.right_joint_sign, "right")
    left_joint_sign = _parse_joint_sign(args.left_joint_sign, "left")
    shared_hw_cfg = args.hw_cfg.strip()
    if not shared_hw_cfg:
        shared_hw_cfg = str(
            Path(tbp.__file__).resolve().parent / "fafu_robot_python" / "robot.cfg"
        )
    left_hw_cfg = args.hw_cfg_left.strip() or shared_hw_cfg
    if left_hw_cfg != shared_hw_cfg:
        print(f"[DUAL] left arm override cfg: {left_hw_cfg}")
    print(f"[DUAL] shared robot.cfg: {shared_hw_cfg}")

    right_arm = ArmTeleopChannel(
        label="RIGHT",
        hw_port=args.hw_port_right.strip(),
        home_button="RIGHT_TRIGGER",
        gripper_open_label="A",
        gripper_close_label="B",
        has_gripper=not args.no_gripper_right,
        joint_sign=right_joint_sign,
        hw_cfg_path=shared_hw_cfg,
        **shared_kw,
    )
    left_arm = ArmTeleopChannel(
        label="LEFT",
        hw_port=args.hw_port_left.strip(),
        home_button="LEFT_TRIGGER",
        gripper_open_label="X",
        gripper_close_label="Y",
        has_gripper=not args.no_gripper_left,
        joint_sign=left_joint_sign,
        hw_cfg_path=left_hw_cfg,
        **shared_kw,
    )

    print(
        f"[DUAL] right arm port={args.hw_port_right}, left arm port={args.hw_port_left}, "
        f"loop_hz={args.loop_hz:.1f}"
    )
    right_arm.enable()
    left_arm.enable()

    loop_deadline = time.perf_counter()
    perf_window_start_t = time.perf_counter()
    perf_loop_count = 0
    perf_loop_sum = 0.0
    perf_overrun_count = 0
    perf_ik_sum = 0.0
    perf_hw_sum = 0.0

    print("Dual-arm teleop started.")
    if getattr(args, "right_home_joints_deg", None) is not None:
        right_home_desc = f"custom {list(args.right_home_joints_deg)} deg"
    elif getattr(args, "right_home_ready_pose", False):
        right_home_desc = "ready pose J2~45deg"
    else:
        right_home_desc = "joint zeros"
    print(f"RIGHT: grip follow, RIGHT_TRIGGER home ({right_home_desc}), A/B gripper.")
    print(
        f"[ANTI-JUMP] grip_stable={args.grip_engage_stable_frames}/"
        f"{args.grip_release_stable_frames}, "
        f"home_arrive={args.home_arrive_deg:.0f}deg, "
        f"servo_reject={'hold' if args.servo_reject_hold_command else 'snap'}"
    )
    if getattr(args, "right_home_joints_deg", None) is not None:
        print(f"[DUAL] right home override (deg) = {list(args.right_home_joints_deg)}")
    if left_arm.preset_only:
        print(
            "LEFT:  X=关节抬肘+前伸(两阶段), Y=归位; "
            f"step={float(args.left_preset_step_deg):.2f}deg/cycle; grip/trigger 无功能。"
        )
    elif left_arm.has_gripper:
        print("LEFT:  grip follow, LEFT_TRIGGER home, X/Y gripper.")
    else:
        print("LEFT:  grip follow, LEFT_TRIGGER home (startup pose), gripper disabled.")
        if not args.left_home_to_zero:
            print(
                "[DUAL] left LEFT_TRIGGER -> startup pose (not joint zeros); "
                "add --left-home-to-zero to lift-to-zero."
            )
    if left_arm.preset_only and not args.left_full_teleop:
        print("[DUAL] 恢复左臂 VR 全跟随: 加 --left-full-teleop")
    print(f"Global estop: {tbp._XR_ESTOP_BUTTON}")

    try:
        while True:
            loop_t_perf = time.perf_counter()
            dual = tbp.get_xr_dual_states(float(args.xr_grip_threshold))
            if dual is not None:
                xr_right, xr_left, estop_pressed = dual
                if estop_pressed:
                    right_arm.emergency_stop()
                    left_arm.emergency_stop()
                    print("[SAFE] estop pressed, stopping both arms.")
                    break

                home_right = tbp._get_button_pressed("RIGHT_TRIGGER")
                if left_arm.preset_only:
                    ik_l, hw_l = left_arm.step_preset(xr_left)
                else:
                    home_left = tbp._get_button_pressed("LEFT_TRIGGER")
                    ik_l, hw_l = left_arm.step(xr_left, home_left)
                ik_r, hw_r = right_arm.step(xr_right, home_right)
                perf_ik_sum += ik_r + ik_l
                perf_hw_sum += hw_r + hw_l

            tbp._xr_touch_heartbeat()

            if tbp._XR_LINK_READY and (time.time() - tbp._LAST_XR_POLL_WALL_TS > timeout_s):
                right_arm.emergency_stop()
                left_arm.emergency_stop()
                print(
                    f"[SAFE] XR timeout ({(time.time() - tbp._LAST_XR_POLL_WALL_TS) * 1000:.0f} ms), "
                    "emergency stop both arms."
                )
                break

            loop_deadline += dt
            sleep_t_req = loop_deadline - time.perf_counter()
            if sleep_t_req > 0:
                time.sleep(sleep_t_req)
            else:
                perf_overrun_count += 1
                loop_deadline = time.perf_counter()

            loop_elapsed = time.perf_counter() - loop_t_perf
            perf_loop_count += 1
            perf_loop_sum += loop_elapsed
            perf_window_elapsed = time.perf_counter() - perf_window_start_t
            if perf_window_elapsed >= 1.0 and perf_loop_count > 0:
                fps = perf_loop_count / perf_window_elapsed
                avg_loop_ms = perf_loop_sum / perf_loop_count * 1000.0
                avg_ik_ms = perf_ik_sum / perf_loop_count * 1000.0
                avg_hw_ms = perf_hw_sum / perf_loop_count * 1000.0
                print(
                    f"[PERF][DUAL] fps={fps:.1f}, loop_avg={avg_loop_ms:.2f}ms, "
                    f"ik_avg={avg_ik_ms:.2f}ms, hw_avg={avg_hw_ms:.2f}ms, "
                    f"overrun={perf_overrun_count}/{perf_loop_count}"
                )
                perf_window_start_t = time.perf_counter()
                perf_loop_count = 0
                perf_loop_sum = 0.0
                perf_overrun_count = 0
                perf_ik_sum = 0.0
                perf_hw_sum = 0.0
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if tbp._XRT is not None:
            tbp._xr_safe_call(tbp._XRT.close, None)
            tbp._XRT = None
        tbp._LAST_XR_TIMESTAMP_NS = None
        tbp._XR_LINK_READY = False
        tbp._LAST_XR_POLL_WALL_TS = 0.0
        for ch in (right_arm, left_arm):
            ratio = 0.0 if ch.ik_try_count == 0 else (ch.ik_success_count / ch.ik_try_count * 100.0)
            print(f"[{ch.label}][IK][FINAL] success={ch.ik_success_count}/{ch.ik_try_count} ({ratio:.1f}%)")
            ch.disable()
        print("Dual-arm teleop exited.")
