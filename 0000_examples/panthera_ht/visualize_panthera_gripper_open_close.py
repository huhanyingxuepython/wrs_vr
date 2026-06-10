#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize Panthera gripper opening/closing animation.

This script is for quickly checking whether both fingers move correctly.
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from wrs import wd, mgm
from wrs.robot_sim.robots.panthera_ht.panthera_ht import PantheraHTSglArm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=float, default=2.0, help="Open+close cycle period (seconds).")
    parser.add_argument("--fps", type=float, default=30.0, help="Visualization update rate.")
    parser.add_argument("--max-jaw-width", type=float, default=-1.0,
                        help="Max jaw width in meters (<0 uses gripper jaw_range max).")
    parser.add_argument("--debug", action="store_true",
                        help="Print gripper diagnostics periodically.")
    parser.add_argument("--debug-interval", type=float, default=1.0,
                        help="Seconds between debug prints.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    period = max(float(args.period), 0.2)
    fps = max(float(args.fps), 5.0)
    dt = 1.0 / fps

    base = wd.World(cam_pos=[1.2, 1.2, 0.8], lookat_pos=[0.0, 0.0, 0.2])
    mgm.gen_frame().attach_to(base)

    robot = PantheraHTSglArm(enable_cc=False)
    robot.goto_given_conf(np.zeros(6))
    ee = robot.end_effector

    jaw_min = float(ee.jaw_range[0])
    jaw_max = float(ee.jaw_range[1])
    # Use a conservative default cap to avoid model over-extension during quick checks.
    default_safe_max = min(jaw_max, 0.012)
    max_jaw = default_safe_max if args.max_jaw_width < 0 else float(np.clip(args.max_jaw_width, jaw_min, jaw_max))
    print(f"[GRIPPER] jaw_range=[{jaw_min:.4f}, {jaw_max:.4f}] m, animation_max={max_jaw:.4f} m")

    # Use "current initial posture" as open baseline, then only move inward.
    open_command = None
    close_command = None
    if hasattr(ee, "_set_chain_command") and hasattr(ee, "_jaw_to_command"):
        open_command = float(ee.jlc.jnts[1].motion_value)
        close_command = float(ee._jaw_to_command(jaw_min))
        print(
            f"[GRIPPER] command mapping from initial posture: "
            f"open_cmd={open_command:.4f}, close_cmd={close_command:.4f}"
        )
    else:
        print("[WARN] Gripper lacks command-level methods; fallback to jaw-width control.")

    print("[INFO] Press Ctrl+C in terminal to stop.")

    model = robot.gen_meshmodel(alpha=0.9, toggle_tcp_frame=True)
    model.attach_to(base)
    start_t = time.time()
    last_dbg_t = 0.0

    try:
        while True:
            now = time.time()
            elapsed = now - start_t
            phase = (elapsed % period) / period
            # triangle wave in [0, 1], start from initial-open, then close inward and return.
            tri = abs(2.0 * phase - 1.0)
            jaw_width = jaw_min + tri * (max_jaw - jaw_min)

            if open_command is not None and close_command is not None and hasattr(ee, "_set_chain_command"):
                command = close_command + tri * (open_command - close_command)
                ee._set_chain_command(command)
            else:
                robot.change_jaw_width(jaw_width)
            model.detach()
            model = robot.gen_meshmodel(alpha=0.9, toggle_tcp_frame=True)
            model.attach_to(base)

            if args.debug and (now - last_dbg_t >= max(float(args.debug_interval), 0.0)):
                q0 = float(ee.jlc.jnts[0].motion_value)
                q1 = float(ee.jlc.jnts[1].motion_value)
                gap = float(ee._measure_gap()) if hasattr(ee, "_measure_gap") else float("nan")
                direction = "increase->open" if getattr(ee, "_command_open_with_increasing", True) else "increase->close"
                print(
                    f"[DBG] jaw_cmd={jaw_width:.4f} m, jaw_est={ee.get_jaw_width():.4f} m, "
                    f"q0={q0:.4f}, q1={q1:.4f}, gap={gap:.4f}, dir={direction}"
                )
                last_dbg_t = now

            base.taskMgr.step()
            sleep_t = dt - (time.time() - now)
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()

