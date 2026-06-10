#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/06/10 18:25
# @Author : HuHanYing
"""
Record Pico/XR controller stream to JSONL for offline replay.

Usage examples:
    python 0000_examples/panthera_ht/record_xr_controller.py
    python 0000_examples/panthera_ht/record_xr_controller.py --hz 100 --seconds 60
    python 0000_examples/panthera_ht/record_xr_controller.py --controller right --output logs/pico_right.jsonl

Output format (one JSON object per line):
{
  "timestamp_ns": ...,
  "timestamp_s": ...,
  "controller_side": "right",
  "controller_pos_vr": [x, y, z],
  "controller_quat_vr": [qx, qy, qz, qw],
  "grip_pressed": true/false,
  "estop_pressed": true/false,
  "buttons": {...},
  "triggers": {...},
  "axes": {...},
  "left_controller_pose_raw": [...],
  "right_controller_pose_raw": [...],
  "headset_pose_raw": [...]
}

Field mapping to teleop_bridge_panthera.py::XRState
---------------------------------------------------
This recorder writes many debug fields, but teleop replay only needs:

- XRState.timestamp_s       <- record["timestamp_s"]
- XRState.controller_pos_vr <- record["controller_pos_vr"]
- XRState.controller_quat_vr<- record["controller_quat_vr"]  # [x, y, z, w]
- XRState.grip_pressed      <- record["grip_pressed"]
- XRState.estop_pressed     <- record["estop_pressed"]

Minimal replay frame example:
{
  "timestamp_s": 1715580000.123,
  "controller_pos_vr": [0.12, -0.03, 0.85],
  "controller_quat_vr": [0.01, 0.71, 0.02, 0.70],
  "grip_pressed": true,
  "estop_pressed": false
}
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import xrobotoolkit_sdk as xrt


def _safe_call(fn, default=None):
    """安全调用 SDK 接口。

    某些情况下（设备短暂断连、SDK 状态未就绪）接口可能抛异常。
    这里统一兜底，避免采集循环被一次异常打断。
    """
    try:
        return fn()
    except Exception:
        return default


def _to_list(value: Any, expected_len: int, default: Iterable[float]) -> List[float]:
    """把任意输入尽量规整成固定长度 list[float]。

    - 如果 value 无效，返回 default；
    - 如果 value 是 list/tuple 且长度够，截取前 expected_len 个元素。
    """
    if value is None:
        return list(default)

    # 兼容 numpy 数组、pybind 向量、tuple/list 等可迭代对象
    try:
        if hasattr(value, "tolist"):
            seq = value.tolist()
        else:
            seq = list(value)
        if len(seq) >= expected_len:
            return [float(seq[i]) for i in range(expected_len)]
    except Exception:
        pass

    # 兼容“带属性”的对象（不同 SDK 绑定有时返回 struct）
    attr_sets = [
        ("x", "y", "z", "qx", "qy", "qz", "qw"),
        ("px", "py", "pz", "qx", "qy", "qz", "qw"),
    ]
    for attrs in attr_sets:
        if all(hasattr(value, a) for a in attrs):
            try:
                return [float(getattr(value, a)) for a in attrs]
            except Exception:
                pass

    return list(default)


def _parse_pose7(pose: Any) -> Dict[str, List[float]]:
    """
    Expected SDK pose shape: [x, y, z, qx, qy, qz, qw].
    Fallback to zeros + identity quaternion if unavailable.
    """
    arr = _to_list(pose, 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return {
        "pos": arr[:3],
        "quat_xyzw": arr[3:7],
        "raw": arr,
    }


def build_parser() -> argparse.ArgumentParser:
    """命令行参数定义。"""
    parser = argparse.ArgumentParser(description="Record XR controller data to JSONL.")
    parser.add_argument("--hz", type=float, default=100.0, help="Sampling frequency (default: 100).")
    parser.add_argument("--seconds", type=float, default=60.0, help="Record duration seconds (default: 60).")
    parser.add_argument(
        "--controller",
        choices=["left", "right"],
        default="right",
        help="Select which controller maps to controller_pos_vr/controller_quat_vr.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output jsonl path. Default: <this_script_dir>/logs/xr_controller_YYYYMMDD_HHMMSS.jsonl",
    )
    parser.add_argument("--debug-first", action="store_true", help="Print first raw pose types/values for debugging.")
    return parser


def make_output_path(user_output: str) -> Path:
    """根据用户输入生成输出文件路径。"""
    if user_output:
        return Path(user_output)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).resolve().parent
    return script_dir / "logs" / f"xr_controller_{ts}.jsonl"


def main() -> None:
    # 采样周期（秒），例如 100Hz -> 0.01s
    args = build_parser().parse_args()
    dt = 1.0 / max(args.hz, 1.0)
    duration_s = max(args.seconds, 0.1)
    output_path = make_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] init XR SDK...")
    xrt.init()
    print(f"[INFO] recording controller='{args.controller}', hz={args.hz}, seconds={duration_s}")
    print(f"[INFO] output -> {output_path}")

    sample_count = 0
    start_wall = time.time()
    end_wall = start_wall + duration_s

    try:
        with output_path.open("w", encoding="utf-8") as f:
            while time.time() < end_wall:
                loop_t = time.time()

                # 时间戳优先使用 SDK 提供值；不可用则退化为本机时钟。
                timestamp_ns = _safe_call(xrt.get_time_stamp_ns, default=int(loop_t * 1e9))
                timestamp_s = float(timestamp_ns) / 1e9

                # 三类位姿：左右手柄 + 头显。
                left_pose_raw = _safe_call(xrt.get_left_controller_pose, default=None)
                right_pose_raw = _safe_call(xrt.get_right_controller_pose, default=None)
                headset_pose_raw = _safe_call(xrt.get_headset_pose, default=None)

                if args.debug_first and sample_count == 0:
                    print(f"[DEBUG] left_pose_raw type={type(left_pose_raw)} value={left_pose_raw}")
                    print(f"[DEBUG] right_pose_raw type={type(right_pose_raw)} value={right_pose_raw}")
                    print(f"[DEBUG] headset_pose_raw type={type(headset_pose_raw)} value={headset_pose_raw}")

                left_pose = _parse_pose7(left_pose_raw)
                right_pose = _parse_pose7(right_pose_raw)

                # 选择哪只手作为 teleop 的“主控制手”。
                # 这决定了 controller_pos_vr/controller_quat_vr 对应哪只手。
                if args.controller == "right":
                    active_pose = right_pose
                    # 注意：很多 SDK 的 grip 是 0~1 浮点，这里 bool(...) 表示“是否按下”。
                    grip_pressed = bool(_safe_call(xrt.get_right_grip, default=0))
                    # 先默认 A 键当急停逻辑输入（仅记录层，真正急停在 teleop 桥接里执行）。
                    estop_pressed = bool(_safe_call(xrt.get_A_button, default=0))
                else:
                    active_pose = left_pose
                    grip_pressed = bool(_safe_call(xrt.get_left_grip, default=0))
                    estop_pressed = bool(_safe_call(xrt.get_X_button, default=0))

                # 一条“可回放帧”：
                # - 前半部分与 teleop_bridge_panthera.py 的 XRState 对齐；
                # - 后半部分是辅助调试信息，便于后期排错和重映射。
                #
                # XRState 对照（再次强调）：
                #   timestamp_s        -> rec["timestamp_s"]
                #   controller_pos_vr  -> rec["controller_pos_vr"]
                #   controller_quat_vr -> rec["controller_quat_vr"] (xyzw)
                #   grip_pressed       -> rec["grip_pressed"]
                #   estop_pressed      -> rec["estop_pressed"]
                rec = {
                    "timestamp_ns": int(timestamp_ns),
                    "timestamp_s": timestamp_s,
                    "controller_side": args.controller,
                    # fields aligned with XRState used by teleop_bridge_panthera.py
                    "controller_pos_vr": active_pose["pos"],
                    "controller_quat_vr": active_pose["quat_xyzw"],
                    "grip_pressed": grip_pressed,
                    "estop_pressed": estop_pressed,
                    # extra debug fields
                    "buttons": {
                        "A": bool(_safe_call(xrt.get_A_button, default=0)),
                        "B": bool(_safe_call(xrt.get_B_button, default=0)),
                        "X": bool(_safe_call(xrt.get_X_button, default=0)),
                        "Y": bool(_safe_call(xrt.get_Y_button, default=0)),
                        "left_menu": bool(_safe_call(xrt.get_left_menu_button, default=0)),
                        "right_menu": bool(_safe_call(xrt.get_right_menu_button, default=0)),
                        "left_axis_click": bool(_safe_call(xrt.get_left_axis_click, default=0)),
                        "right_axis_click": bool(_safe_call(xrt.get_right_axis_click, default=0)),
                    },
                    "triggers": {
                        "left_trigger": float(_safe_call(xrt.get_left_trigger, default=0.0)),
                        "right_trigger": float(_safe_call(xrt.get_right_trigger, default=0.0)),
                        "left_grip": float(_safe_call(xrt.get_left_grip, default=0.0)),
                        "right_grip": float(_safe_call(xrt.get_right_grip, default=0.0)),
                    },
                    "axes": {
                        "left_axis": _safe_call(xrt.get_left_axis, default=[0.0, 0.0]),
                        "right_axis": _safe_call(xrt.get_right_axis, default=[0.0, 0.0]),
                    },
                    "left_controller_pose_raw": left_pose["raw"],
                    "right_controller_pose_raw": right_pose["raw"],
                    "headset_pose_raw": _to_list(headset_pose_raw, 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                }

                # JSONL：一行一帧，后处理和回放都很方便。
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                sample_count += 1

                # 每约 1 秒打印一次进度（按当前 hz 估算）。
                if sample_count % int(max(args.hz, 10)) == 0:
                    elapsed = loop_t - start_wall
                    print(f"[INFO] samples={sample_count}, elapsed={elapsed:.1f}s")

                # 固定频率采样：减去本轮处理耗时后再 sleep。
                sleep_t = dt - (time.time() - loop_t)
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("[INFO] recording stopped by user.")
    finally:
        # 无论正常结束还是 Ctrl+C，都做 SDK 关闭和统计输出。
        _safe_call(xrt.close, default=None)
        actual_time = max(time.time() - start_wall, 1e-9)
        print(f"[DONE] wrote {sample_count} frames")
        print(f"[DONE] avg rate ~ {sample_count / actual_time:.1f} Hz")
        print(f"[DONE] file: {output_path}")


if __name__ == "__main__":
    main()

