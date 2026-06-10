# -*- coding: utf-8 -*-
"""
test_fafu_motion_interactive.py
===============================

带菜单的交互式实验脚�?(基于 fafu_robot_controller 的高层封�?.
不用反复改代码就能跑�?FafuRobotController 的各种功�?
(单关�?/ 多关�?/ 夹爪 / 软限�?/ 状态监�?/ 示教复现).

⚠️ 第一次跑�?
    - 机器臂上�?
    - 桌面留出空间, 手放在键�?Ctrl+C �? 急停就在旁边
    - 推荐 --speed 不超�?15

用法:
    cd fafu_robot_sdk
    python tests/test_fafu_motion_interactive.py                      # 默认无夹�?
    python tests/test_fafu_motion_interactive.py --gripper-id 7       # 7 号是夹爪
    python tests/test_fafu_motion_interactive.py --speed 10           # 全局降�?
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# ============================================================================
#  Helpers
# ============================================================================
def deg_str(angles_rad) -> str:
    return "[" + ", ".join(f"{math.degrees(v):+7.2f}" for v in angles_rad) + "]"


def parse_floats(s: str, sep: Optional[str] = None) -> List[float]:
    """'10 20 30' / '10,20,30' / '10, 20, 30' -> [10.0, 20.0, 30.0]."""
    s = s.replace(",", " ").strip()
    out: List[float] = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            raise ValueError(f"无法解析为数�? {tok!r}")
    return out


def prompt(text: str) -> str:
    try:
        return input(text).strip()
    except EOFError:
        return ""


def yes(s: str) -> bool:
    return s.lower() in ("y", "yes", "ok", "ok!", "")


# ============================================================================
#  Menu items
# ============================================================================
class App:
    def __init__(self, arm, default_speed: int):
        from fafu_robot_controller import FafuRobotController
        assert isinstance(arm, FafuRobotController)
        self.arm = arm
        self.default_speed = default_speed
        # 示教录制的路�?(rad list 列表)
        self.taught_path: List[List[float]] = []
        # 自定义书�? name -> (joint_angles_rad, gripper_angle_rad_or_None)
        self.bookmarks: Dict[str, tuple] = {}

    # ----- 状�?信息 -----

    def show_state(self):
        q = self.arm.get_joint_values()
        print(f"\n  joint motor ids: {self.arm.joint_motor_ids}")
        print(f"  joint angles   : {deg_str(q)} (deg)")
        try:
            qd = self.arm.get_joint_velocities()
            print(f"  joint vel      : {deg_str(qd)} (deg/s)")
        except Exception as e:
            print(f"  joint vel      : <读取失败: {e}>")

        if self.arm.has_gripper:
            try:
                gs = self.arm.get_gripper_state()
                ang_deg = math.degrees(self.arm._turns_to_rad(gs.position))
                print(f"  gripper M{self.arm.gripper_motor_id:<2}    : {ang_deg:+.2f} deg "
                      f"(mode=0x{gs.mode:02X}, fault=0x{gs.fault:02X})")
            except Exception as e:
                print(f"  gripper        : <读取失败: {e}>")

        print(f"  is_enabled     : {self.arm.is_enabled}")
        print(f"  stats          : {self.arm.get_status().to_string()}")

    def show_limits(self):
        print()
        for mid in self.arm.all_motor_ids:
            lim = self.arm.get_limit(mid)
            tag = " (gripper)" if (self.arm.has_gripper
                                   and mid == self.arm.gripper_motor_id) else ""
            if lim is None:
                print(f"  M{mid}{tag}: 未设软限�?)
            else:
                lo, hi = lim
                print(f"  M{mid}{tag}: [{math.degrees(lo):+8.2f}, "
                      f"{math.degrees(hi):+8.2f}] deg")

    # ----- 单关节运�?-----

    def move_one_joint(self):
        n = self.arm.num_joints
        s = prompt(f"  动哪个关�?(0..{n - 1}, 默认 {n - 1}): ")
        try:
            idx = int(s) if s else (n - 1)
        except ValueError:
            print("  非法编号"); return
        if not (0 <= idx < n):
            print(f"  超出范围 [0, {n})"); return

        s = prompt("  目标角度 (�? �?+30): ")
        try:
            target_deg = float(s)
        except ValueError:
            print("  非法数字"); return

        s = prompt(f"  speed (默认 {self.default_speed}): ")
        speed = int(s) if s else self.default_speed

        q0 = self.arm.get_joint_values()
        target = q0.copy()
        target[idx] = math.radians(target_deg)
        print(f"\n  当前 (deg): {deg_str(q0)}")
        print(f"  目标 (deg): {deg_str(target)}")
        if not yes(prompt("  执行? [Y/n]: ")):
            print("  取消."); return
        try:
            self.arm.move_j(target, speed=speed, block=True)
        except Exception as e:
            traceback.print_exc()
            print(f"  [失败] {e}"); return
        time.sleep(0.2)
        q1 = self.arm.get_joint_values()
        err = math.degrees(q1[idx] - target[idx])
        print(f"  到位 (deg): {deg_str(q1)}  关节 {idx} 误差 {err:+.3f}°")

    # ----- 多关节运�?-----

    def move_all_joints(self):
        n = self.arm.num_joints
        print(f"  输入 {n} 个角�?(�?, 空格或逗号分隔:")
        s = prompt(f"  > ")
        try:
            vals = parse_floats(s)
        except ValueError as e:
            print(f"  {e}"); return
        if len(vals) != n:
            print(f"  需�?{n} 个�? 给了 {len(vals)} �?); return

        s = prompt(f"  speed (默认 {self.default_speed}): ")
        speed = int(s) if s else self.default_speed

        target = np.radians(vals)
        q0 = self.arm.get_joint_values()
        print(f"\n  当前 (deg): {deg_str(q0)}")
        print(f"  目标 (deg): {deg_str(target)}")
        if not yes(prompt("  执行? [Y/n]: ")):
            print("  取消."); return
        try:
            self.arm.move_j(target, speed=speed, block=True)
        except Exception as e:
            traceback.print_exc()
            print(f"  [失败] {e}"); return
        time.sleep(0.2)
        print(f"  到位 (deg): {deg_str(self.arm.get_joint_values())}")

    def go_home(self):
        s = prompt(f"  speed (默认 {self.default_speed}): ")
        speed = int(s) if s else self.default_speed
        if not yes(prompt("  全部关节回零? [Y/n]: ")):
            print("  取消."); return
        try:
            self.arm.go_home(speed=speed, block=True)
        except Exception as e:
            traceback.print_exc()
            print(f"  [失败] {e}"); return
        print(f"  到位 (deg): {deg_str(self.arm.get_joint_values())}")

    # ----- 夹爪 -----

    def gripper_menu(self):
        if not self.arm.has_gripper:
            print("\n  没有配置夹爪 (启动时未�?--gripper-id)")
            return
        while True:
            try:
                gs = self.arm.get_gripper_state()
                cur_deg = math.degrees(self.arm._turns_to_rad(gs.position))
                print(f"\n  当前夹爪角度: {cur_deg:+.2f} deg")
            except Exception as e:
                print(f"\n  读取失败: {e}")
            print("    [o] open  (软限位上�? 阻塞)")
            print("    [c] close (软限位下�? 阻塞)")
            print("    [a] 自定义角�?)
            print("    [v] 设速度 / 当前默认 vel=0.3 turns/s")
            print("    [b] 返回主菜�?)
            cmd = prompt("  > ").lower()
            if cmd in ("b", "back", "q", "exit", ""):
                return
            elif cmd == "o":
                self.arm.open_gripper()
            elif cmd == "c":
                self.arm.close_gripper()
            elif cmd == "a":
                s = prompt("  目标角度 (�?: ")
                try:
                    deg = float(s)
                except ValueError:
                    print("  非法数字"); continue
                self.arm.gripper_control(angle=math.radians(deg))
            elif cmd == "v":
                print("  (跳过, 留作 TODO)")
            else:
                print(f"  未识�? {cmd!r}")

    # ----- 软限�?-----

    def limit_menu(self):
        while True:
            print("\n  软限�?")
            self.show_limits()
            print("\n    [s] set   <id> <lo_deg> <hi_deg>")
            print("    [d] disable <id>")
            print("    [c] clear all")
            print("    [b] 返回")
            cmd = prompt("  > ").lower()
            if cmd in ("b", "back", "q", "exit", ""):
                return
            if cmd == "c":
                if yes(prompt("  确认清空全部? [Y/n]: ")):
                    self.arm.clear_limits()
                    print("  已清�?)
                continue
            tokens = cmd.split()
            if len(tokens) >= 2 and tokens[0] == "d":
                try:
                    mid = int(tokens[1])
                    self.arm.disable_limit(mid)
                    print(f"  已禁�?M{mid}")
                except Exception as e:
                    print(f"  失败: {e}")
                continue
            if len(tokens) == 4 and tokens[0] == "s":
                try:
                    mid = int(tokens[1])
                    lo = float(tokens[2])
                    hi = float(tokens[3])
                    self.arm.set_limit(mid, lo=lo, hi=hi, is_radians=False)
                    print(f"  已设 M{mid}: [{lo:+.2f}, {hi:+.2f}] deg")
                except Exception as e:
                    print(f"  失败: {e}")
                continue
            print(f"  未识�? {cmd!r}")

    # ----- 状态实时监�?(Ctrl+C 退�? -----

    def monitor(self, hz: float = 4.0):
        print("\n  [实时监视] Ctrl+C 退�?)
        period = 1.0 / max(0.5, hz)
        try:
            while True:
                q = self.arm.get_joint_values()
                line = "  " + deg_str(q)
                if self.arm.has_gripper:
                    try:
                        gs = self.arm.get_gripper_state()
                        line += f"  | grip {math.degrees(self.arm._turns_to_rad(gs.position)):+6.1f}°"
                    except Exception:
                        pass
                print(line, flush=True)
                time.sleep(period)
        except KeyboardInterrupt:
            print("\n  [退出监视]")

    # ----- 示教 / 复现 -----

    def teach_record(self):
        print("\n  示教录制:")
        print("    1) 自动 disable 所有关�? 你用手拖动到目标位姿")
        print("    2) 每按一�?Enter 就记录一个路�?)
        print("    3) 输入 'q' Enter 结束并自�?enable")
        if not yes(prompt("\n  开�? [Y/n]: ")):
            return
        try:
            self.arm.disable()
        except Exception as e:
            print(f"  disable 失败: {e}"); return
        recorded: List[List[float]] = []
        try:
            while True:
                cmd = prompt(f"  �?Enter 记录路点 #{len(recorded) + 1}, 'q' 结束: ")
                if cmd.lower() in ("q", "quit", "exit"):
                    break
                q = self.arm.get_joint_values(prefer_cache=False)
                recorded.append(q.tolist())
                print(f"     �?#{len(recorded)}: {deg_str(q)}")
        finally:
            print("\n  恢复 enable...")
            try:
                self.arm.enable()
            except Exception as e:
                print(f"  enable 失败: {e}")
        if recorded:
            self.taught_path = recorded
            print(f"  已保�?{len(recorded)} 个路�?(内存�?")
        else:
            print("  未记录任何路�?)

    def teach_replay(self):
        if not self.taught_path:
            print("\n  未录制任何路�? 先用 [t] 录制"); return
        s = prompt(f"  speed (默认 {self.default_speed}): ")
        speed = int(s) if s else self.default_speed
        n = len(self.taught_path)
        print(f"\n  复现 {n} 个路�?(speed={speed}, 阻塞 block=True)")
        if not yes(prompt("  开�? [Y/n]: ")):
            return
        try:
            for i, q in enumerate(self.taught_path, start=1):
                print(f"  [{i}/{n}] -> {deg_str(q)}")
                self.arm.move_j(q, speed=speed, block=True)
                time.sleep(0.2)
            print("  完成")
        except KeyboardInterrupt:
            print("\n  [中断] 已停止当前运�?); self.arm.emergency_stop()
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}")

    # ----- ServoJ (online streaming) -----

    def servo_menu(self):
        """servo_j sub-menu. �?move_j 实时性高, 适合 teleop / 视觉伺服 demo."""
        from fafu_robot_controller import ServoOpts

        # 当前 session 的默认参�? 可以在菜�?[s] 里改
        watchdog_ms  = 100
        max_vel      = 1.0          # rad/s
        max_step_deg = 1.5          # deg, �?C++ 例程稍宽松点便于上手
        max_lag_deg  = 10.0         # deg

        def _opts() -> ServoOpts:
            return ServoOpts(
                watchdog_ms=watchdog_ms,
                max_vel=max_vel,
                max_step_rad=math.radians(max_step_deg),
                max_lag_rad=math.radians(max_lag_deg),
                is_radians=True,
            )

        while True:
            print("\n  ServoJ (online streaming) 子菜�?")
            print(f"    当前参数: watchdog={watchdog_ms}ms, "
                  f"max_vel={max_vel:.2f}rad/s, "
                  f"step<={max_step_deg:.1f}deg, lag<={max_lag_deg:.1f}deg")
            print(f"    is_servoing = {self.arm.is_servoing}")
            print("    [1] hold-in-place 测试  (推荐先跑这个, 验证看门�?")
            print("    [2] 单关节正弦跟�?     (100Hz, 选编�?振幅+持续时间)")
            print("    [3] 多关节联动正�?     (100Hz, 相邻关节反相)")
            print("    [4] 示教路径 servo 复现 (�?[p] �?move_j 复现做对�?")
            print("    [s] 改默认参�?)
            print("    [b] 返回")
            cmd = prompt("  > ").lower()
            if cmd in ("b", "back", "q", "exit", ""):
                return
            if cmd == "1":
                self._servo_hold_test(_opts())
            elif cmd == "2":
                self._servo_single_joint_sin(_opts())
            elif cmd == "3":
                self._servo_multi_joint_sin(_opts())
            elif cmd == "4":
                self._servo_replay_path(_opts())
            elif cmd == "s":
                s = prompt(f"  watchdog_ms (当前 {watchdog_ms}, "
                           "0=禁用): ")
                if s:
                    try: watchdog_ms = int(s)
                    except ValueError: print("  非法整数")
                s = prompt(f"  max_vel rad/s (当前 {max_vel:.2f}): ")
                if s:
                    try: max_vel = float(s)
                    except ValueError: print("  非法数字")
                s = prompt(f"  max_step_deg (当前 {max_step_deg:.2f}): ")
                if s:
                    try: max_step_deg = float(s)
                    except ValueError: print("  非法数字")
                s = prompt(f"  max_lag_deg (当前 {max_lag_deg:.2f}): ")
                if s:
                    try: max_lag_deg = float(s)
                    except ValueError: print("  非法数字")
            else:
                print(f"  未识�? {cmd!r}")

    def _servo_hold_test(self, opts):
        """[1] �?servo 会话, 持续把当前位�?stream 出去 N �?

        关键观察�?
          - servo_end 后会打印实际平均频率 (应该接近 100Hz)
          - 中途按 Ctrl+C 会触�?finally -> servo_end, 看动作平�?
          - 想真正验证看门狗: 改完代码后从 task manager 强杀 python.exe,
            机械臂应该在 watchdog_ms 内自�?brake (不会保持最后一帧冲)
        """
        s = prompt("  持续时长 �?(默认 5): ")
        try: duration = float(s) if s else 5.0
        except ValueError: print("  非法数字"); return
        period_s = 0.01    # 100 Hz
        n_total = int(duration / period_s)

        q0 = self.arm.get_joint_values()
        print(f"\n  起点 (deg): {deg_str(q0)}")
        print(f"  会持续给这个目标�?{duration:.1f}s @ {1/period_s:.0f}Hz")
        if not yes(prompt("  开�? [Y/n]: ")):
            return

        try:
            self.arm.servo_start(opts)
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}"); return

        send_ok, send_fail = 0, 0
        t_start = time.monotonic()
        next_tick = t_start
        try:
            for _ in range(n_total):
                if self.arm.servo_j(q0):
                    send_ok += 1
                else:
                    send_fail += 1
                next_tick += period_s
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\n  [中断] Ctrl+C 收到, �?servo_end")
        finally:
            print(f"  sent OK={send_ok}  fail={send_fail}")
            try: self.arm.servo_end("hold")
            except Exception as e: print(f"  servo_end 失败: {e}")

    def _servo_single_joint_sin(self, opts):
        """[2] 单关�?sin 跟踪. 默认 J6 (末端), 振幅 10deg, 周期 2s."""
        n = self.arm.num_joints
        s = prompt(f"  动哪个关�?(0..{n-1}, 默认 {n-1}): ")
        try: idx = int(s) if s else (n - 1)
        except ValueError: print("  非法编号"); return
        if not (0 <= idx < n): print("  超出范围"); return

        s = prompt("  振幅 deg (默认 10): ")
        try: amp_deg = float(s) if s else 10.0
        except ValueError: print("  非法数字"); return

        s = prompt("  正弦周期 �?(默认 2.0): ")
        try: period_sin = float(s) if s else 2.0
        except ValueError: print("  非法数字"); return

        s = prompt("  持续时长 �?(默认 6): ")
        try: duration = float(s) if s else 6.0
        except ValueError: print("  非法数字"); return

        q0 = self.arm.get_joint_values()
        amp = math.radians(amp_deg)
        freq = 1.0 / max(0.1, period_sin)
        peak_vel = 2 * math.pi * freq * amp     # rad/s, 估算
        print(f"\n  起点 (deg): {deg_str(q0)}")
        print(f"  J{idx} 围绕 q0[{idx}] �?±{amp_deg:.1f}deg / "
              f"{period_sin:.1f}s sin")
        print(f"  估算峰值速度 {peak_vel:.2f} rad/s  "
              f"(opts.max_vel={opts.max_vel:.2f} rad/s, "
              f"step≈{math.degrees(2*math.pi*freq*amp*0.01):.2f}deg/tick)")
        if peak_vel > opts.max_vel:
            print(f"  �?峰值速度可能高于 max_vel, 会出�?lag, 调小振幅或加大周�?�?)
        if not yes(prompt("  开�? [Y/n]: ")):
            return

        try:
            self.arm.servo_start(opts)
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}"); return

        period_s = 0.01   # 100Hz
        n_total = int(duration / period_s)
        send_ok, send_fail = 0, 0
        t_start = time.monotonic()
        next_tick = t_start
        try:
            for k in range(n_total):
                t = k * period_s
                target = q0.copy()
                target[idx] = q0[idx] + amp * math.sin(2 * math.pi * freq * t)
                if self.arm.servo_j(target):
                    send_ok += 1
                else:
                    send_fail += 1
                next_tick += period_s
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\n  [中断]")
        finally:
            print(f"  sent OK={send_ok}  fail={send_fail}")
            try: self.arm.servo_end("hold")
            except Exception as e: print(f"  servo_end 失败: {e}")

    def _servo_multi_joint_sin(self, opts):
        """[3] 所有关节同 sin, 相邻关节反相, 视觉上更明显."""
        s = prompt("  振幅 deg (默认 8): ")
        try: amp_deg = float(s) if s else 8.0
        except ValueError: print("  非法数字"); return
        s = prompt("  正弦周期 �?(默认 2.5): ")
        try: period_sin = float(s) if s else 2.5
        except ValueError: print("  非法数字"); return
        s = prompt("  持续时长 �?(默认 6): ")
        try: duration = float(s) if s else 6.0
        except ValueError: print("  非法数字"); return

        q0 = self.arm.get_joint_values()
        amp = math.radians(amp_deg)
        freq = 1.0 / max(0.1, period_sin)
        n = self.arm.num_joints
        signs = np.array([1.0 if (j % 2 == 0) else -1.0 for j in range(n)])
        peak_vel = 2 * math.pi * freq * amp
        print(f"\n  起点 (deg): {deg_str(q0)}")
        print(f"  所有关节同 sin (相邻反相), ±{amp_deg:.1f}deg / {period_sin:.1f}s")
        print(f"  估算峰值速度 {peak_vel:.2f} rad/s "
              f"(opts.max_vel={opts.max_vel:.2f} rad/s)")
        if not yes(prompt("  开�? [Y/n]: ")):
            return

        try:
            self.arm.servo_start(opts)
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}"); return

        period_s = 0.01
        n_total = int(duration / period_s)
        send_ok, send_fail = 0, 0
        next_tick = time.monotonic()
        try:
            for k in range(n_total):
                t = k * period_s
                phase = math.sin(2 * math.pi * freq * t)
                target = q0 + signs * amp * phase
                if self.arm.servo_j(target):
                    send_ok += 1
                else:
                    send_fail += 1
                next_tick += period_s
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\n  [中断]")
        finally:
            print(f"  sent OK={send_ok}  fail={send_fail}")
            try: self.arm.servo_end("hold")
            except Exception as e: print(f"  servo_end 失败: {e}")

    def _servo_replay_path(self, opts):
        """[4] 把示教录制的路点用线性插�?streaming, �?[p] �?move_j 复现对比."""
        if not self.taught_path:
            print("\n  未录制任何路�? 先用 [t] 录制几个�?); return
        if len(self.taught_path) < 2:
            print("\n  至少需�?2 个路�?); return

        s = prompt("  段间时间 �?�?(默认 2.0, 越大越慢越平�?: ")
        try: seg_time = float(s) if s else 2.0
        except ValueError: print("  非法数字"); return

        period_s = 0.01    # 100Hz
        ticks_per_seg = max(2, int(seg_time / period_s))
        wp = [np.asarray(q, dtype=float) for q in self.taught_path]
        n_seg = len(wp) - 1
        total_ticks = n_seg * ticks_per_seg
        peak_step = 0.0
        for a, b in zip(wp[:-1], wp[1:]):
            step = np.max(np.abs(b - a)) / ticks_per_seg
            peak_step = max(peak_step, step)
        peak_step_deg = math.degrees(peak_step)
        max_step_deg  = math.degrees(opts.max_step_rad)
        print(f"\n  {len(wp)} 个路�? {n_seg} �? 每段 {seg_time:.1f}s "
              f"({total_ticks} ticks)")
        print(f"  估算最大单 tick 步长 {peak_step_deg:.3f}deg "
              f"(opts.max_step={max_step_deg:.3f}deg)")
        if peak_step > opts.max_step_rad:
            print("  �?�?tick 步长 > max_step, 会被 clamp, 段间时间太短 �?)
        if not yes(prompt("  开�? [Y/n]: ")):
            return

        # �?move_j 走到首点 (保证第一帧不大跳)
        try:
            self.arm.move_j(wp[0], speed=self.default_speed, block=True)
            time.sleep(0.2)
        except Exception as e:
            print(f"  move_j to wp[0] 失败: {e}"); return

        try:
            self.arm.servo_start(opts)
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}"); return

        send_ok, send_fail = 0, 0
        t_start = time.monotonic()
        next_tick = t_start
        try:
            for seg_idx in range(n_seg):
                a, b = wp[seg_idx], wp[seg_idx + 1]
                for k in range(ticks_per_seg):
                    alpha = (k + 1) / ticks_per_seg
                    target = a + (b - a) * alpha
                    if self.arm.servo_j(target):
                        send_ok += 1
                    else:
                        send_fail += 1
                    next_tick += period_s
                    sleep_for = next_tick - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\n  [中断]")
        finally:
            print(f"  sent OK={send_ok}  fail={send_fail}")
            try: self.arm.servo_end("hold")
            except Exception as e: print(f"  servo_end 失败: {e}")

    # ----- 安全 -----

    def estop(self):
        if yes(prompt("\n  确认急停 (mode=0x00)? [Y/n]: ")):
            self.arm.emergency_stop()
            print("  急停已发. �?'r' 恢复.")

    def resume(self):
        if yes(prompt("\n  确认恢复 (重新 enable)? [Y/n]: ")):
            try:
                self.arm.resume()
            except Exception as e:
                print(f"  失败: {e}")


# ============================================================================
#  Main loop
# ============================================================================
MENU = """
+--------------------------------------------------+
|  FafuRobotController 交互式测�?                 |
+--------------------------------------------------+
  [s]  显示当前状�?(角度 / 速度 / 夹爪 / 统计)
  [j]  单关节移�?(选编�?+ 目标角度)
  [a]  全部关节移动 (输入 N 个角�?
  [h]  回零 (go_home)
  [g]  夹爪子菜�?
  [l]  软限位子菜单
  [t]  示教录制 (手动拖动 + �?Enter 记录)
  [p]  复现已录制的路点
  [v]  ServoJ (online streaming, 100Hz 跟踪, 含看门狗)
  [m]  实时状态监�?(Ctrl+C 退�?
  [e]  急停
  [r]  从急停恢复 (enable)
  [q]  退�?
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="interactive test menu")
    parser.add_argument("--cfg", default="robot.cfg",
                        help="robot.cfg 路径 (默认: fafu_robot_sdk/robot.cfg)")
    parser.add_argument("--gripper-id", type=int, default=None,
                        help="夹爪 motor id, 不传则视为无夹爪")
    parser.add_argument("--speed", type=int, default=15,
                        help="默认 speed percent (默认 15)")
    args = parser.parse_args()

    try:
        from fafu_robot_controller import FafuRobotController
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] import fafu_robot_controller 失败: {e}")
        print("       先跑 tests/smoke_test.py 排查环境")
        return 1

    has_gripper = args.gripper_id is not None
    try:
        arm = FafuRobotController(
            cfg_path=args.cfg,
            has_gripper=has_gripper,
            gripper_motor_id=args.gripper_id,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] 创建 FafuRobotController 失败: {e}")
        return 1

    app = App(arm, default_speed=args.speed)
    rc = 0
    try:
        while True:
            print(MENU)
            cmd = prompt("  >>> ").lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "s":
                app.show_state()
            elif cmd == "j":
                app.move_one_joint()
            elif cmd in ("a", "all"):
                app.move_all_joints()
            elif cmd in ("h", "home"):
                app.go_home()
            elif cmd == "g":
                app.gripper_menu()
            elif cmd == "l":
                app.limit_menu()
            elif cmd == "t":
                app.teach_record()
            elif cmd == "p":
                app.teach_replay()
            elif cmd == "v":
                app.servo_menu()
            elif cmd == "m":
                app.monitor()
            elif cmd == "e":
                app.estop()
            elif cmd == "r":
                app.resume()
            elif cmd == "":
                continue
            else:
                print(f"  未识�? {cmd!r}")
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Ctrl+C, 紧急停�?..")
        try:
            arm.emergency_stop()
        except Exception:
            pass
        rc = 130
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] {e}")
        try:
            arm.emergency_stop()
        except Exception:
            pass
        rc = 1
    finally:
        try:
            arm.close_connection()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
