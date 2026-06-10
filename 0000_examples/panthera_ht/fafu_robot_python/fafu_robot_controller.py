# -*- coding: utf-8 -*-
"""
Fafu Robot Controller Wrapper
=============================

This module provides a high-level Python wrapper around the
``panthera_motor`` (pybind11) module for the **Fafu robot arm**
(built on the Hightorque / Panthera-HT debug board hardware).
The goal is to present an interface similar in spirit to the
``PiperArmController`` shown in :mod:`piper.py` while hiding the
low-level details of motor IDs, modes, units and the
``HightorqueSerial`` driver.

Naming note
-----------
The underlying pybind11 binding module is still called
``panthera_motor`` because it is the unchanged C++ driver for the
debug board (renaming it would require rebuilding ``bindings.cpp``).
Everything user-facing — class names, log prefixes, docstrings —
is exposed under the **Fafu** identity.

Conventions
-----------

* All joint angles exposed to the user are in **radians** by default
  (``is_radians=True``).  Internally the wrapper converts radians to
  *turns* (the protocol native unit; ``1 turn = 2*pi rad``).
* Velocities are expressed as a percentage in the ``speed`` argument
  (``0 - 100``); this is mapped to a peak average velocity in
  turns/second.
* The Fafu arm is a chain of independent motors driven over a
  USB-CAN debug board; there is no built-in inverse kinematics.
  ``move_p`` / ``move_l`` therefore raise :class:`NotImplementedError`
  unless an external IK solver is plugged in.

Example
-------

>>> from fafu_robot_controller import FafuRobotController
>>> import numpy as np
>>>
>>> # cfg_path is required; gripper is optional (motor id 7 in the
>>> # default robot.cfg).
>>> arm = FafuRobotController(
...     cfg_path="robot.cfg",
...     has_gripper=True,
...     gripper_motor_id=7,
... )
>>>
>>> # current joint angles (rad)
>>> q = arm.get_joint_values()
>>>
>>> # move to a target configuration with S-curve and wait for finish
>>> arm.move_j([0, 0.2, 0.5, 0, 0, 0], speed=20, block=True)
>>>
>>> arm.open_gripper()
>>> arm.close_gripper()
>>>
>>> # Piper-style position+effort (firmware-side torque cap)
>>> arm.gripper_control(angle=math.radians(-90), effort=600)
>>>
>>> # Force-aware grasp (Python-side torque monitoring + early stop)
>>> result = arm.grasp(force_threshold=500)
>>> if result.grasped:
...     print(f"got it, peak torque {result.peak_torque_raw} raw, "
...           f"closed {result.closed_deg:.1f} deg in {result.duration_s:.2f}s")
>>>
>>> arm.disable()
>>> arm.close_connection()
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
#  Make sure panthera_motor.pyd next to this file is importable.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import panthera_motor as pm  # noqa: E402

# Optional: TOPPRA-based time-optimal interpolation (matches piper.py).
try:
    import wrs.motion.trajectory.piecewisepoly_toppra as pwp  # type: ignore

    _TOPPRA_EXIST = True
except Exception:  # pragma: no cover - optional dependency
    _TOPPRA_EXIST = False

# Optional: wrs robot_math, only needed if move_p / move_l are wired to IK.
try:
    import wrs.basis.robot_math as rm  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    rm = None  # type: ignore


# ============================================================================
#  Constants
# ============================================================================

# Operating modes (matches motor_example_debug / arm_multi_joint_example).
MODE_POSITION = 0x0A   # release / position-control hold
MODE_BRAKE    = 0x0F   # short-circuit braking (no torque to spin)
MODE_STOP     = 0x00   # PWM off, free to move by hand

# S-curve trajectory tuning (matches arm_multi_joint_example.py).
_VEL_AVG_MAX_TPS = 0.5    # absolute cap for average velocity (turns/s)
_DT_MIN_S        = 0.3    # shortest segment time (s)
_SETTLE_MS       = 300    # extra hold time after the trajectory ends

# 1 turn = 2*pi rad
_TWO_PI = 2.0 * math.pi


# ============================================================================
#  Public dataclasses
# ============================================================================
@dataclass
class GraspResult:
    """Outcome of a :meth:`FafuRobotController.grasp` call.

    Attributes
    ----------
    grasped : bool
        ``True`` iff the wrapper concluded that an object was caught
        (torque threshold reached, or the gripper stalled after having
        clearly moved towards the target).  ``False`` for "reached the
        target with no resistance" / "did not move at all" / "timeout".
    reason : str
        One of:

        * ``'detected_object_force'`` — ``|torque| >= effort_threshold``
        * ``'detected_object_stall'`` — speed plateaued after closing
          at least ``min_close_deg``
        * ``'reached_target'``      — got to the commanded angle, nothing in the way
        * ``'no_movement'``         — stalled but barely moved (command
          may not have taken effect, or jaws were already shut)
        * ``'timeout'``             — neither condition met within ``timeout``
    angle_rad : float
        Final gripper angle (radians).
    closed_deg : float
        Absolute change in the gripper angle since the call started
        (degrees).  Always non-negative.
    peak_torque_raw : int
        Maximum ``|MotorState.torque|`` observed during the move
        (raw int16 from the motor).
    duration_s : float
        Wall-clock time the call took (seconds).
    """
    grasped: bool
    reason: str
    angle_rad: float
    closed_deg: float
    peak_torque_raw: int
    duration_s: float


@dataclass
class ServoOpts:
    """Tunables for an online ``servo_j`` streaming session.

    See the four safety lines documented on
    :meth:`FafuRobotController.servo_start`.

    Attributes
    ----------
    watchdog_ms : int, optional
        Firmware-side watchdog in milliseconds.  If the motor receives
        no new command for this long it automatically brakes.  Default
        ``100``.  Set to ``0`` to disable (★ not recommended ★).
    max_vel : float, optional
        Per-joint velocity cap in **rad/s** written into every frame
        (``set_many_pos_vel_tqe`` ``vel`` field).  Default ``1.0``
        (~57 deg/s).  When ``feedforward_vel=True`` this acts as an
        upper safety bound on the computed feedforward velocity.
    max_step_rad : float, optional
        Per-step jump limit in **rad**.  ``|target - last_target|``
        larger than this is clamped to ``±max_step_rad`` and a warning
        is logged.  Default ``0.05`` (~2.9 deg).
    max_lag_rad : float, optional
        Tracking-error guard in **rad**.  If any motor's measured
        position deviates from its last target by more than this,
        :meth:`servo_j` returns ``False``.  Default ``0.2`` (~11.5 deg).
        Set to ``0`` / negative to disable.
    is_radians : bool, optional
        Interpret arguments to :meth:`servo_j` in radians (default) or
        degrees.  Matches :meth:`move_j`.
    rate_hz : float, optional
        Nominal upper-layer call rate **only used to compute dt for
        feedforward and lookahead**.  Default ``100.0``.  The actual
        call rate is still set by however fast the caller invokes
        :meth:`servo_j`; this number does not throttle anything.
    feedforward_vel : bool, optional
        Default ``True``.  When enabled, the per-frame ``vel`` field
        written to each motor is the **true required velocity**
        ``(target[k] - target[k-1]) * rate_hz`` (clamped to
        ``±max_vel``), exactly like UR servoj's internal velocity
        feedforward.  When ``False`` every frame uses the constant
        ``max_vel`` as ``vel``, which makes the motor "always sprint"
        toward each target and is the dominant source of high-frequency
        whine in joystick / teleop scenarios.
    lookahead_time : float, optional
        Default ``0.0`` (no smoothing).  When ``> 0``, an exponential
        moving average (first-order low-pass) with time constant
        ``lookahead_time`` is applied to every target before it is
        sent.  Recommended ``0.03 - 0.10`` s for noisy upper layers
        (joystick teleop, VR, vision servoing); leave at ``0`` when the
        upper layer already produces a smooth trajectory (planned
        path).  This is the same knob UR servoj exposes as
        ``lookahead_time``.
    """

    watchdog_ms: int = 100
    max_vel: float = 1.0
    max_step_rad: float = 0.05
    max_lag_rad: float = 0.2
    is_radians: bool = True
    # ---- noise / lag tuning (UR-servoj style) ----
    rate_hz: float = 100.0
    feedforward_vel: bool = True
    lookahead_time: float = 0.0


# ============================================================================
#  Controller
# ============================================================================
class FafuRobotController:
    """High-level controller for the Fafu robotic arm.

    Parameters
    ----------
    cfg_path : str
        Path to the ``robot.cfg`` file.  Relative paths are first
        resolved against the current working directory and then
        against the directory holding this file.  The configuration
        provides ``port``, ``baudrate``, ``motor_ids``, soft limits
        and the control rate.
    port : str, optional
        Override ``cfg.port``.  ``"auto"`` (or ``None``) triggers
        USB enumeration via :func:`panthera_motor.find_likely_debug_boards`.
    baudrate : int, optional
        Override ``cfg.baudrate``.  Defaults to ``cfg.baudrate``
        (typically 4 Mbps).
    has_gripper : bool, optional
        If ``True`` the motor with id ``gripper_motor_id`` is treated
        as the gripper rather than a manipulator joint.  Joint-space
        commands (``move_j``, ``get_joint_values`` ...) then ignore
        that motor.  Defaults to ``False``.
    gripper_motor_id : int, optional
        Motor id of the gripper.  Required when ``has_gripper`` is
        ``True``; must be present in ``cfg.motor_ids``.
    auto_enable : bool, optional
        If ``True`` (default) all motors are switched into
        position-control mode (``0x0A``) immediately after the serial
        port is opened.
    auto_polling : bool, optional
        If ``True`` (default) a 50 Hz background polling thread is
        started after enabling so that :meth:`get_joint_values` reads
        from a non-blocking cache.
    async_rx : bool, optional
        Override ``cfg.use_async_rx``.  When ``None`` the value from
        the configuration file is used.

    Notes
    -----
    The Fafu firmware requires :meth:`set_motor_mode` to be
    called *before* :meth:`enable_async_rx`, so this constructor
    enforces that order regardless of how the flags are combined.
    """

    # Re-exported as class attributes so users do not have to import
    # the module-level constants.
    MODE_POSITION = MODE_POSITION
    MODE_BRAKE    = MODE_BRAKE
    MODE_STOP     = MODE_STOP

    # ------------------------------------------------------------------
    #  Construction / teardown
    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg_path: str,
        *,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
        has_gripper: bool = False,
        gripper_motor_id: Optional[int] = None,
        auto_enable: bool = True,
        auto_polling: bool = True,
        async_rx: Optional[bool] = None,
    ) -> None:
        if not cfg_path:
            raise ValueError("cfg_path must be provided")

        cfg_path = self._resolve_cfg_path(cfg_path)
        try:
            cfg = pm.RobotConfig.load(cfg_path)
        except Exception as e:
            raise RuntimeError(f"failed to load config {cfg_path!r}: {e}") from e

        self._cfg_path = cfg_path
        self._cfg: pm.RobotConfig = cfg

        if has_gripper:
            if gripper_motor_id is None:
                raise ValueError("has_gripper=True requires gripper_motor_id")
        if gripper_motor_id is not None and gripper_motor_id not in cfg.motor_ids:
            raise ValueError(
                f"gripper_motor_id {gripper_motor_id} not in cfg.motor_ids "
                f"{list(cfg.motor_ids)}"
            )
        self._has_gripper = bool(has_gripper)
        self._gripper_motor_id = gripper_motor_id

        # Joint motors == all motor_ids minus the gripper id when it is used or skipped.
        if gripper_motor_id is not None:
            self._joint_motor_ids: List[int] = [
                m for m in cfg.motor_ids if m != gripper_motor_id
            ]
        else:
            self._joint_motor_ids = list(cfg.motor_ids)

        if not self._joint_motor_ids:
            raise ValueError("no joint motors after excluding the gripper")

        port_to_use = self._pick_serial_port(port if port else cfg.port)
        baud_to_use = int(baudrate or cfg.baudrate)

        self._ht = None
        try:
            try:
                self._ht = pm.HightorqueSerial(port_to_use, baud_to_use)
            except Exception as e:
                raise RuntimeError(
                    f"failed to open serial port {port_to_use!r} @ {baud_to_use}: {e}"
                ) from e
            self._port = port_to_use
            self._baudrate = baud_to_use

            # Push soft limits configured in robot.cfg into the driver.
            try:
                cfg.apply_limits_to(self._ht)
            except Exception as e:
                print(f"[FafuRobot] warning: apply_limits_to failed: {e}")

            self._warmup_can_bus()
            # Verify that every managed motor responds before doing anything risky.
            self._precheck_communication()

            # Order is significant: set_motor_mode MUST run before
            # enable_async_rx so that the SDK can verify the mode echo.
            if auto_enable:
                self.enable()

            use_async = async_rx if async_rx is not None else bool(cfg.use_async_rx)
            if use_async:
                try:
                    self._ht.enable_async_rx()
                    time.sleep(0.1)
                    # Prime the RX cache so get_cached_state returns immediately.
                    for mid in self._managed_motor_ids():
                        self._ht.read_motor_state(mid, 0.1)
                except Exception as e:
                    print(f"[FafuRobot] warning: enable_async_rx failed: {e}")

            if auto_polling:
                try:
                    hz = float(cfg.control_rate_hz) if cfg.control_rate_hz else 50.0
                    hz = max(10.0, hz)
                    self._ht.start_state_polling(self._managed_motor_ids(), hz)
                except Exception as e:
                    print(f"[FafuRobot] warning: start_state_polling failed: {e}")
        except Exception:
            self._safe_close_ht_on_init_failure()
            raise

        # Servo (online streaming) session state. None when not servoing.
        self._servo_active: bool = False
        self._servo_opts: Optional[ServoOpts] = None
        self._servo_last_target_turns: List[float] = []
        # filtered_target_turns mirrors last_target when lookahead_time == 0;
        # when smoothing is enabled it lags the raw target by ~lookahead_time.
        self._servo_filtered_target_turns: List[float] = []
        # Pre-allocated numpy buffers for set_many_pos_vel_tqe_partial.
        # active_ids / hold_ids are set once in servo_start and never
        # re-allocated; active_pos / active_vel are overwritten in place
        # every tick to avoid per-call numpy allocation.
        self._servo_active_ids_np: Optional[np.ndarray] = None
        self._servo_hold_ids_np: Optional[np.ndarray] = None
        self._servo_active_pos_np: Optional[np.ndarray] = None
        self._servo_active_vel_np: Optional[np.ndarray] = None
        self._servo_max_motor_id: int = 0
        self._servo_max_torque: int = 0
        self._servo_started_at: float = 0.0
        self._servo_tick_count: int = 0
        # warning counters for batched logging at servo_end
        self._servo_clamp_count: int = 0
        self._servo_lag_count: int = 0

        gripper_note = ""
        if self._has_gripper and self._gripper_motor_id is not None:
            gripper_note = f" + gripper M{self._gripper_motor_id}"
        elif self._gripper_motor_id is not None:
            gripper_note = f" (skip M{self._gripper_motor_id})"
        print(
            f"[FafuRobot] connected on {self._port} @ {self._baudrate} "
            f"({len(self._joint_motor_ids)} joints{gripper_note})"
        )

    # ------------------------------------------------------------------
    #  Public properties
    # ------------------------------------------------------------------
    @property
    def cfg(self) -> pm.RobotConfig:
        """Underlying :class:`panthera_motor.RobotConfig`."""
        return self._cfg

    @property
    def port(self) -> str:
        return self._port

    @property
    def baudrate(self) -> int:
        return self._baudrate

    @property
    def joint_motor_ids(self) -> List[int]:
        """Motor ids that participate in joint-space commands (excludes gripper)."""
        return list(self._joint_motor_ids)

    def _managed_motor_ids(self) -> List[int]:
        """Motors this controller actively talks to (joints + optional gripper)."""
        ids = list(self._joint_motor_ids)
        if self._has_gripper and self._gripper_motor_id is not None:
            ids.append(self._gripper_motor_id)
        return sorted(set(ids))

    @property
    def all_motor_ids(self) -> List[int]:
        return list(self._cfg.motor_ids)

    @property
    def num_joints(self) -> int:
        return len(self._joint_motor_ids)

    @property
    def has_gripper(self) -> bool:
        return self._has_gripper

    @property
    def gripper_motor_id(self) -> Optional[int]:
        return self._gripper_motor_id
        
    @property
    def driver(self) -> pm.HightorqueSerial:
        """Escape hatch to the underlying ``HightorqueSerial`` instance."""
        return self._ht

    @property
    def is_enabled(self) -> bool:
        """``True`` iff every motor is currently in position-control mode."""
        for mid in self._cfg.motor_ids:
            s = self._ht.get_cached_state(mid)
            if s is None:
                s = self._ht.read_motor_state(mid, 0.05)
            if s is None or s.mode != self.MODE_POSITION:
                return False
        return True

    # ------------------------------------------------------------------
    #  Power management
    # ------------------------------------------------------------------
    def enable(self, *, allow_motor_reset: bool = True) -> None:
        """Switch every motor to position control (mode ``0x0A``).

        Resolution path (cheapest to most aggressive):

        0. **Fast path.** Read every motor's live state; if all are
           already in MODE_POSITION (left over from a previous
           ``servo_end("hold")`` or ``close_connection(release="hold")``)
           return without sending any mode-change frame.  This both
           saves ~30 ms and avoids a known false-failure mode of
           ``set_motor_mode(0x0A)`` whose internal "switch + read-back"
           sequence sometimes reports ``None`` on a freshly-reopened
           port even when the mode change actually succeeded.
        1. **Normal path.** Call :meth:`_switch_mode_all` with up to
           3 retries; it now does a fresh ``read_motor_state`` to
           verify failures, filtering out the false-failure case
           above.
        2. **Soft-reboot path.** If the normal path still fails, print
           a diagnostic for every motor (mode / fault / position read
           live), then issue a firmware-level ``motor_reset`` to every
           failed motor (fire-and-forget reset of the on-motor
           controller — does **not** clear zero / config), wait 1 s
           for them to come back up, and try the normal path one more
           time.

        Most cases of "refused mode 0x0A" after a previous
        ``servo_end`` / ``close_connection`` are resolved at step 2
        without a hardware power-cycle.

        Parameters
        ----------
        allow_motor_reset : bool, optional
            Default ``True``.  Set to ``False`` to keep the legacy
            behaviour (raise immediately if step 1 fails).  Useful
            when something else owns the bus and you do not want a
            stealth ``motor_reset``.

        Raises
        ------
        RuntimeError
            Either step 1 (when ``allow_motor_reset=False``) or both
            stages exhausted.  In the latter case the printed
            diagnostic above tells you whether the motors are
            unreachable (no state read), faulted (non-zero ``fault``
            register), or mechanically jammed (mode = 0x0A but the
            commanded position cannot be tracked).
        """
        # Stage-0: fast path. read once, skip the whole switch if every
        # motor is already in MODE_POSITION. This handles the common
        # case of "previous program left motors holding" cleanly and
        # cheaply, and dodges the sync-read false-failure issue.
        already_active = True
        for mid in self._managed_motor_ids():
            s = self._ht.read_motor_state(mid, 0.2)
            if s is None or int(s.mode) != self.MODE_POSITION:
                already_active = False
                break
        if already_active:
            print("[FafuRobot] all motors already in position control hold; "
                  "enable() is a no-op.")
            return

        if self._switch_mode_all(self.MODE_POSITION, label="position", max_retry=3):
            time.sleep(0.05)
            print("[FafuRobot] all motors enabled (position control hold).")
            return

        # Stage-1 failed. Print a diagnostic so the next message is
        # actionable instead of a generic "refused".
        self._print_motor_diagnostic(prefix="enable: stage-1 failed; ")

        if not allow_motor_reset:
            raise RuntimeError(
                "enable failed: at least one motor refused mode 0x0A "
                "(motor_reset recovery disabled by allow_motor_reset=False)"
            )

        # Stage-2: motor_reset is a firmware-level soft reboot of the
        # on-motor controller. It is the equivalent of pulling power on
        # *only that motor* for a few hundred ms, without losing zero
        # calibration / soft limits / etc. Use as last resort before
        # asking the user to power-cycle.
        print("[FafuRobot] enable: attempting motor_reset on every motor (soft reboot) ...")
        for mid in self._managed_motor_ids():
            try:
                self._ht.motor_reset(mid)
            except Exception as e:
                print(f"  motor {mid}: motor_reset failed: {e}")
        # Firmware needs time to come back. Empirically ~0.6 s is
        # enough on the boards we have; 1.0 s gives margin.
        time.sleep(1.0)

        if not self._switch_mode_all(self.MODE_POSITION, label="position (post-reset)", max_retry=3):
            self._print_motor_diagnostic(prefix="enable: stage-2 (post-reset) failed; ")
            raise RuntimeError(
                "enable failed even after motor_reset; check the "
                "diagnostic above. Likely causes: (a) motor controller "
                "in latched FAULT state -> hard power-cycle; "
                "(b) USB-CAN bus disconnected / wrong COM port; "
                "(c) mechanical jam holding the joint outside soft limits."
            )

        time.sleep(0.05)
        print("[FafuRobot] all motors enabled (recovered via motor_reset).")

    def _print_motor_diagnostic(self, *, prefix: str = "") -> None:
        """Read every motor's live state and print a compact summary.

        Used by :meth:`enable` to turn an opaque "refused mode 0x0A"
        into something a user can act on.  Reads are synchronous with
        a generous timeout (300 ms) so the failure mode "motor does
        not respond" is also obvious.
        """
        print(f"[FafuRobot] {prefix}motor states:")
        for mid in self._cfg.motor_ids:
            try:
                s = self._ht.read_motor_state(mid, 0.3)
            except Exception as e:
                print(f"  motor {mid}: READ EXCEPTION {e}")
                continue
            if s is None:
                print(f"  motor {mid}: NO RESPONSE (read timeout) — bus or motor power off?")
                continue
            fault_val = getattr(s, "fault", 0)
            fault_tag = "fault=OK" if not fault_val else f"fault=0x{int(fault_val):02X} ★"
            print(f"  motor {mid}: mode=0x{int(s.mode):02X}  {fault_tag}  "
                  f"pos={s.position:+.3f}t  vel={s.velocity:+.3f}t/s")

    def disable(self) -> None:
        """Switch every motor to free-spin mode (mode ``0x00``)."""
        ok = self._switch_mode_all(self.MODE_STOP, label="stop", max_retry=2)
        if ok:
            print("[FafuRobot] all motors disabled (free spin).")

    def brake(self) -> None:
        """Engage short-circuit braking on every motor (mode ``0x0F``)."""
        ok = self._switch_mode_all(self.MODE_BRAKE, label="brake", max_retry=2)
        if ok:
            print("[FafuRobot] all motors braked.")

    # ------------------------------------------------------------------
    #  Joint-space motion
    # ------------------------------------------------------------------
    def move_j(
        self,
        joint_angles: Iterable[float],
        *,
        is_radians: bool = True,
        speed: int = 50,
        block: bool = True,
        tolerance: float = 0.01,
    ) -> None:
        """Move every manipulator joint to a target configuration.

        Parameters
        ----------
        joint_angles : iterable of float
            Sequence of ``num_joints`` joint angles for the manipulator
            joints (in the order of :attr:`joint_motor_ids`).  The
            gripper, if any, is held at its current position.
        is_radians : bool, optional
            Interpret ``joint_angles`` in radians (default) or degrees.
        speed : int, optional
            Speed percentage in ``(0, 100]``.  Mapped linearly to a
            target average velocity ``(speed / 100) * 0.5`` turns/s.
            Defaults to 50 (~ 90 deg/s average).
        block : bool, optional
            * ``True`` (default): generate an S-curve trajectory and
              run it through :meth:`HightorqueSerial.run_control_loop`
              at ``cfg.control_rate_hz``; returns only after the
              trajectory finishes (plus a short settle window).
            * ``False``: send a single ``set_many_pos_vel_tqe`` frame
              and return immediately.
        tolerance : float, optional
            Joint tolerance for the *fast* one-shot blocking fallback
            used when TOPPRA / S-curve cannot run.  In radians (or
            degrees, matching ``is_radians``).  Defaults to ``0.01``.
        """
        angles = self._validate_joint_angles(joint_angles, is_radians)
        targets_turns: Dict[int, float] = {
            mid: angles[i] for i, mid in enumerate(self._joint_motor_ids)
        }
        speed = self._clamp_speed(speed)

        if block:
            self._move_scurve(targets_turns, speed_pct=speed)
            return

        # block=False: single shot, no waiting.
        v_avg = (speed / 100.0) * _VEL_AVG_MAX_TPS
        cmds = self._build_many_cmds_holding_others(targets_turns, vel_rps=v_avg)
        self._ht.set_many_pos_vel_tqe(
            cmds,
            pm.PosUnit.Turns,
            max(self._cfg.motor_ids),
            0.05,
        )

    def go_home(self, *, speed: int = 20, block: bool = True) -> None:
        """Move every manipulator joint back to 0 rad."""
        self.move_j(
            [0.0] * self.num_joints,
            is_radians=True,
            speed=speed,
            block=block,
        )

    def move_jntspace_path(
        self,
        path,
        *,
        is_radians: bool = True,
        max_jntvel: Optional[List[float]] = None,
        max_jntacc: Optional[List[float]] = None,
        start_frame_id: int = 1,
        speed: int = 50,
        control_frequency: float = 0.05,
    ) -> None:
        """Follow a joint-space waypoint path.

        Parameters
        ----------
        path : array_like, shape (N, num_joints)
            Sequence of joint configurations to traverse in order.
        is_radians : bool, optional
            Interpret ``path`` in radians (default) or degrees.
        max_jntvel, max_jntacc : list of float, optional
            Per-joint velocity and acceleration limits (passed straight
            to TOPPRA).
        start_frame_id : int, optional
            Skip the first ``start_frame_id`` interpolated frames
            (typically used to skip the robot's current configuration).
        speed : int, optional
            Speed percentage forwarded to each ``move_j`` call.
        control_frequency : float, optional
            ``ctrl_freq`` value passed to TOPPRA (in seconds).  Note
            that the original :mod:`piper` example also calls this
            value ``control_frequency`` even though TOPPRA expects a
            period (in seconds), not a rate.

        Raises
        ------
        NotImplementedError
            When the optional ``wrs`` dependency is not available.
        """
        if not _TOPPRA_EXIST:
            raise NotImplementedError(
                "TOPPRA-based interpolation requires "
                "`wrs.motion.trajectory.piecewisepoly_toppra`; "
                "install it or use a custom interpolator."
            )
        if path is None:
            raise ValueError("path must not be None")

        path_arr = np.asarray(path, dtype=float)
        if path_arr.ndim != 2 or path_arr.shape[1] != self.num_joints:
            raise ValueError(
                f"path must have shape (N, {self.num_joints}); got {path_arr.shape}"
            )

        tpply = pwp.PiecewisePolyTOPPRA()
        interpolated = tpply.interpolate_by_max_spdacc(
            path=path_arr,
            ctrl_freq=control_frequency,
            max_vels=max_jntvel,
            max_accs=max_jntacc,
            toggle_debug=False,
        )
        interpolated = interpolated[start_frame_id:]
        for jnt_values in interpolated:
            self.move_j(
                joint_angles=jnt_values,
                is_radians=is_radians,
                speed=speed,
                block=False,
            )
            time.sleep(max(0.005, control_frequency))

    # ------------------------------------------------------------------
    #  Servo (online streaming) control
    # ------------------------------------------------------------------
    #
    #  Unlike :meth:`move_j` (offline S-curve, blocking), ``servo_j`` is
    #  designed for upper-layer code that streams a fresh joint target
    #  every ~10 ms (e.g. teleop, VR, visual servoing, in-the-loop IK).
    #
    #  Lifecycle::
    #
    #      arm.servo_start(ServoOpts(...))     # once: arm watchdog, async RX
    #      while not quit:
    #          arm.servo_j(target_angles)      # ~100 Hz, non-blocking
    #          time.sleep_until(next_tick)
    #      arm.servo_end("brake")              # clear watchdog + brake
    #
    #  Four safety lines (cannot be skipped — they all live inside
    #  ``servo_j`` and ``servo_start``):
    #
    #    1) Firmware-side watchdog ( :func:`HightorqueSerial.set_timeout` )
    #       — if the motor sees no new command for ``watchdog_ms`` it
    #       brakes itself. Survives host crash, Ctrl+C, USB unplug.
    #    2) Per-step jump clamp (``max_step_rad``) — protects against
    #       upper-layer planner step bugs.
    #    3) Tracking-error monitor (``max_lag_rad``) — if measured
    #       position trails ``last_target`` by more than this, ``servo_j``
    #       returns ``False`` (stuck joint, motor too weak, planner too
    #       fast, etc.).
    #    4) Soft limits — reuses :meth:`enable_position_limit`; the
    #       underlying ``set_many_pos_vel_tqe`` clamps automatically.
    # ------------------------------------------------------------------
    def servo_start(self, opts: Optional[ServoOpts] = None) -> None:
        """Enter an online-streaming joint-servo session.

        Idempotent on the second call (warns and returns).  After this
        every joint motor has a firmware watchdog armed (1) and the
        wrapper has cached the current joint configuration as the
        initial ``last_target`` (so the first :meth:`servo_j` cannot
        trip the step-clamp warning).  The gripper is **not** touched
        and is not subject to the servo watchdog.

        Parameters
        ----------
        opts : ServoOpts, optional
            Tunables.  Defaults to :class:`ServoOpts` (watchdog 100 ms,
            max_vel 1.0 rad/s, max_step 0.05 rad, max_lag 0.2 rad).

        Raises
        ------
        RuntimeError
            If the driver fails to enable async RX (servo cannot run on
            the blocking sync-RX path) or any joint motor fails to
            report a starting position.
        """
        if self._servo_active:
            print("[FafuRobot] servo_start: already servoing; call servo_end first")
            return

        opts = ServoOpts(**vars(opts)) if opts is not None else ServoOpts()
        if opts.watchdog_ms < 0:
            opts.watchdog_ms = 0
        if opts.watchdog_ms == 0:
            print("[FafuRobot] servo_start: WARNING watchdog disabled; "
                  "host crash will not stop motors. Use only for offline tests.")
        elif opts.watchdog_ms < 30:
            print(f"[FafuRobot] servo_start: WARNING watchdog_ms={opts.watchdog_ms} "
                  "< 30 may trip spuriously at 100Hz; recommend >= 50 ms.")
        if opts.max_vel <= 0.0:
            opts.max_vel = 1.0
        if opts.max_step_rad <= 0.0:
            opts.max_step_rad = 0.05
        if opts.rate_hz <= 0.0:
            opts.rate_hz = 100.0
        if opts.lookahead_time < 0.0:
            opts.lookahead_time = 0.0

        if not self.is_enabled:
            print("[FafuRobot] servo_start: motors not enabled, calling enable() ...")
            self.enable()

        # Servo cannot tolerate the 5-15 ms sync RX wait inside
        # send_can_and_recv_, so force async RX on if it is not already.
        if not self._ht.is_async_rx():
            try:
                self._ht.enable_async_rx()
                time.sleep(0.05)
            except Exception as e:
                raise RuntimeError(f"servo_start: enable_async_rx failed: {e}") from e

        # Capture current pos as last_target so the first servo_j is a
        # zero-step move and never trips the step-clamp warning.
        last_target: List[float] = []
        for mid in self._joint_motor_ids:
            s = self._ht.get_cached_state(mid)
            if s is None:
                s = self._ht.read_motor_state(mid, 0.1)
            if s is None:
                raise RuntimeError(
                    f"servo_start: cannot read motor {mid} starting position")
            last_target.append(s.position)

        # Pre-build the numpy arrays consumed by
        # ``set_many_pos_vel_tqe_partial`` so the per-tick allocation
        # cost is zero (we only memcpy target_pos / target_vel into
        # the pre-allocated buffers below; active_ids / hold_ids never
        # change for a session).
        active_ids_np = np.asarray(self._joint_motor_ids, dtype=np.int32)
        # Non-joint motors are normally "held" at cached pos each tick.
        # The gripper motor must never be in hold_ids: servo_j would
        # overwrite M7 every tick (canceling open/close) when has_gripper,
        # or heat the gripper at max_torque when has_gripper=False but
        # gripper_motor_id is set to exclude M7 from the joint list.
        hold_motor_ids = [
            mid for mid in self._cfg.motor_ids if mid not in self._joint_motor_ids
        ]
        if self._gripper_motor_id is not None:
            hold_motor_ids = [
                mid for mid in hold_motor_ids if mid != self._gripper_motor_id
            ]
        hold_ids_np = np.asarray(hold_motor_ids, dtype=np.int32)
        # Pre-allocated output buffers reused every tick.
        N = len(self._joint_motor_ids)
        active_pos_np = np.zeros(N, dtype=np.float64)
        active_vel_np = np.zeros(N, dtype=np.float64)

        # Arm the firmware watchdog on every JOINT motor (gripper excluded
        # — gripper is not part of the servo loop, watching it would brake
        # the jaws while the loop is still in start-up).
        if opts.watchdog_ms > 0:
            for mid in self._joint_motor_ids:
                try:
                    self._ht.set_timeout(mid, int(opts.watchdog_ms))
                except Exception as e:
                    print(f"[FafuRobot] servo_start: set_timeout({mid}) failed: {e}")

        self._servo_opts = opts
        self._servo_last_target_turns = list(last_target)
        self._servo_filtered_target_turns = list(last_target)
        self._servo_active_ids_np = active_ids_np
        self._servo_hold_ids_np = hold_ids_np
        self._servo_active_pos_np = active_pos_np
        self._servo_active_vel_np = active_vel_np
        self._servo_max_motor_id = int(max(self._cfg.motor_ids))
        self._servo_max_torque = int(self._cfg.max_torque_raw)
        self._servo_active = True
        self._servo_tick_count = 0
        self._servo_clamp_count = 0
        self._servo_lag_count = 0
        self._servo_started_at = time.monotonic()

        ff_tag = "on" if opts.feedforward_vel else "off"
        la_tag = (f"{opts.lookahead_time * 1000:.0f}ms"
                  if opts.lookahead_time > 0 else "off")
        print(
            f"[FafuRobot] servo_start: watchdog={opts.watchdog_ms}ms, "
            f"max_vel={opts.max_vel}rad/s, max_step={opts.max_step_rad}rad, "
            f"max_lag={opts.max_lag_rad}rad, "
            f"feedforward={ff_tag}, lookahead={la_tag}, rate={opts.rate_hz:.0f}Hz"
        )

    def servo_j(self, target_angles: Iterable[float]) -> bool:
        """Stream one joint-space target (non-blocking).

        Call repeatedly at a fixed cadence (recommended 100-200 Hz).
        Caller is responsible for the sleep between calls; we do not
        block to enforce a rate.

        Parameters
        ----------
        target_angles : iterable of float
            ``num_joints`` joint angles in the unit declared by
            :attr:`ServoOpts.is_radians` (radians by default).

        Returns
        -------
        bool
            * ``True``  — frame was sent.  ``last_target`` advanced.
              (Step- or limit-clamping does not count as failure; it
              prints a warning but still returns ``True``.)
            * ``False`` — payload was rejected and **no frame was
              sent**.  Reasons: wrong length, NaN/Inf, tracking error
              over ``max_lag_rad``, send failed, or
              :meth:`servo_start` was never called.  ``last_target``
              stays untouched so the next ``servo_j`` will still see
              the same step delta.
        """
        if not self._servo_active or self._servo_opts is None:
            print("[FafuRobot] servo_j: not in a servo session; call servo_start first")
            return False

        opts = self._servo_opts
        arr = np.asarray(list(target_angles), dtype=float)
        if arr.size != self.num_joints:
            print(f"[FafuRobot] servo_j: expected {self.num_joints} angles, "
                  f"got {arr.size}")
            return False
        if not np.all(np.isfinite(arr)):
            print("[FafuRobot] servo_j: target contains NaN/Inf, refused")
            return False

        N = self.num_joints
        dt = 1.0 / opts.rate_hz
        max_vel_tps = opts.max_vel / _TWO_PI    # rad/s -> turns/s

        # (a) Convert user-units -> turns (protocol native)
        if opts.is_radians:
            target_turns = [float(v) / _TWO_PI for v in arr]
        else:
            target_turns = [float(v) / 360.0 for v in arr]

        # (b) Defense 2 — per-step clamp on the *raw* upper-layer target,
        # measured against last successfully-sent filtered target. Without
        # this an upper-layer bug could push the EMA filter far in one
        # step (lookahead does NOT bound a single step, only smoothes it).
        max_step_turns = abs(opts.max_step_rad) / _TWO_PI
        was_clamped = False
        for i in range(N):
            delta = target_turns[i] - self._servo_last_target_turns[i]
            if abs(delta) > max_step_turns:
                target_turns[i] = (self._servo_last_target_turns[i]
                                   + math.copysign(max_step_turns, delta))
                was_clamped = True
        if was_clamped:
            self._servo_clamp_count += 1
            # No per-tick print — would dominate jitter on 100Hz loops.
            # The aggregated count is reported in servo_end.

        # (c) Lookahead smoothing — exponential moving average with time
        # constant ``lookahead_time``. Matches UR servoj's lookahead.
        # alpha = dt / (lookahead + dt); alpha=1 when lookahead=0 (no smoothing).
        if opts.lookahead_time > 0.0:
            alpha = dt / (opts.lookahead_time + dt)
            target_to_send = [
                self._servo_filtered_target_turns[i]
                + alpha * (target_turns[i] - self._servo_filtered_target_turns[i])
                for i in range(N)
            ]
        else:
            target_to_send = target_turns

        # (d) Per-joint velocity, written directly into the pre-allocated
        # buffer (no per-tick allocation):
        #   - feedforward_vel=True (default, UR-style):
        #         vel[i] = (target_to_send[i] - last_sent[i]) / dt, clamped to ±max_vel
        #     Eliminates the dominant "motor sprints to every micro-step"
        #     noise source. Tiny target changes get tiny vel commands.
        #   - feedforward_vel=False (legacy):
        #         vel[i] = max_vel_tps for every joint
        active_pos_buf = self._servo_active_pos_np
        active_vel_buf = self._servo_active_vel_np
        if opts.feedforward_vel:
            for i in range(N):
                active_pos_buf[i] = target_to_send[i]
                v = (target_to_send[i] - self._servo_filtered_target_turns[i]) / dt
                if v > max_vel_tps:    v = max_vel_tps
                elif v < -max_vel_tps: v = -max_vel_tps
                active_vel_buf[i] = v
        else:
            for i in range(N):
                active_pos_buf[i] = target_to_send[i]
            active_vel_buf.fill(max_vel_tps)

        # (e) Defense 3 — tracking error against measured, batched.
        # Bad lag = refuse send.
        if opts.max_lag_rad > 0.0:
            max_lag_turns = opts.max_lag_rad / _TWO_PI
            for i, mid in enumerate(self._joint_motor_ids):
                s = self._ht.get_cached_state(mid)
                if s is None:
                    continue
                lag = abs(s.position - target_to_send[i])
                if lag > max_lag_turns:
                    self._servo_lag_count += 1
                    # Aggregated print at servo_end; here we just bail.
                    return False

        # (f) Defense 4 — soft limits handled inside set_many_pos_vel_tqe_partial.

        # (g) One C++ call: active joints get (target, ff_vel); optional
        # hold motors (never the gripper) get hold-at-cached-pos. Hot path is
        # entirely C++ now: marshalling is 4x numpy memcpy (no Python
        # ManyMotorCmd object construction), ~5us per tick.
        try:
            self._ht.set_many_pos_vel_tqe_partial(
                self._servo_active_ids_np,
                active_pos_buf,
                active_vel_buf,
                self._servo_max_torque,
                self._servo_hold_ids_np,
                pm.PosUnit.Turns,
                self._servo_max_motor_id,
                0.0,                    # async_rx is on, don't wait for replies
            )
        except Exception as e:
            print(f"[FafuRobot] servo_j: send failed: {e}")
            return False

        # Advance state only after a successful send so the next tick's
        # clamp / feedforward see a consistent "last sent" reference.
        self._servo_last_target_turns = list(target_turns)
        self._servo_filtered_target_turns = list(target_to_send)
        self._servo_tick_count += 1
        return True

    def servo_end(self, finish_mode: str = "hold") -> None:
        """End the servo session and place every joint motor in
        ``finish_mode``.

        Parameters
        ----------
        finish_mode : {"hold", "brake", "stop"}, optional
            * ``"hold"`` (default): keep the motor in MODE_POSITION
              (``0x0A``) holding its last commanded frame.  Identical
              behaviour to the state every motor is in after a
              successful :meth:`move_j`, so you can immediately chain
              another ``move_j`` / ``servo_start`` without calling
              :meth:`enable` again.  Motors stay energised.
            * ``"brake"``: switch to short-circuit braking
              (``0x0F``).  No torque output, no current draw, but the
              joint resists motion (it may still drift slowly under
              load).  Use when you want to release the loop but keep
              the arm roughly in place without burning power.
            * ``"stop"``: PWM off (``0x00``).  Joints free to be moved
              by hand.  Use only when a human will reposition the arm
              right after, otherwise the arm will sag.

        Always clears the firmware watchdog first so that the mode
        change is not interrupted by the watchdog firing in the
        middle.  The gripper is left alone (we never touched it).
        """
        if not self._servo_active:
            print("[FafuRobot] servo_end: not in a servo session, ignored")
            return

        valid = {"stop", "brake", "hold"}
        if finish_mode not in valid:
            raise ValueError(
                f"finish_mode must be one of {sorted(valid)}, got {finish_mode!r}")

        opts = self._servo_opts
        # 1) clear watchdog FIRST so the mode switch is not pre-empted
        if opts is not None and opts.watchdog_ms > 0:
            for mid in self._joint_motor_ids:
                try:
                    self._ht.set_timeout(mid, 0)
                except Exception:
                    pass

        # 2) place motors in finish_mode
        for mid in self._joint_motor_ids:
            try:
                if finish_mode == "stop":
                    self._ht.stop(mid)
                elif finish_mode == "brake":
                    self._ht.set_motor_mode(mid, self.MODE_BRAKE)
                # "hold": leave in MODE_POSITION with last frame intact
            except Exception:
                pass

        elapsed = max(1e-3, time.monotonic() - self._servo_started_at)
        rate = self._servo_tick_count / elapsed
        warn_tag = ""
        if self._servo_clamp_count or self._servo_lag_count:
            warn_tag = (f"   [warn: step-clamp={self._servo_clamp_count}, "
                        f"lag-trip={self._servo_lag_count}]")
        print(f"[FafuRobot] servo_end ({finish_mode}): "
              f"{self._servo_tick_count} ticks in {elapsed:.2f}s "
              f"(~{rate:.1f} Hz){warn_tag}")

        self._servo_active = False
        self._servo_tick_count = 0
        self._servo_clamp_count = 0
        self._servo_lag_count = 0
        self._servo_last_target_turns = []
        self._servo_filtered_target_turns = []
        self._servo_active_ids_np = None
        self._servo_hold_ids_np = None
        self._servo_active_pos_np = None
        self._servo_active_vel_np = None
        self._servo_max_motor_id = 0
        self._servo_max_torque = 0

    @property
    def is_servoing(self) -> bool:
        """``True`` while a :meth:`servo_start` / :meth:`servo_end`
        window is open."""
        return self._servo_active

    # ------------------------------------------------------------------
    #  Cartesian motion (placeholders)
    # ------------------------------------------------------------------
    def move_p(
        self,
        pos: Iterable[float],
        rot: np.ndarray,
        *,
        is_euler: bool = False,
        speed: int = 50,
    ) -> None:
        """Move the end effector to a Cartesian pose (placeholder).

        Raises
        ------
        NotImplementedError
            The Fafu stack ships motor-level controls only;
            inverse kinematics must be supplied by the caller.  Once
            an IK solver returns ``q`` (joint angles), feed it to
            :meth:`move_j`.
        """
        # TODO: plug in an external IK solver (e.g. wrs / pinocchio)
        # to convert (pos, rot) -> joint targets, then dispatch via
        # self.move_j(targets, ...).
        raise NotImplementedError(
            "FafuRobotController has no built-in IK; "
            "compute joint angles externally and call move_j()."
        )

    def move_l(
        self,
        pos: Iterable[float],
        rot: np.ndarray,
        *,
        is_euler: bool = False,
        speed: int = 50,
    ) -> None:
        """Linear Cartesian motion (placeholder).

        Raises
        ------
        NotImplementedError
            See :meth:`move_p`.
        """
        # TODO: implement by sampling the Cartesian line, IK-ing each
        # waypoint and feeding the resulting joint path to
        # move_jntspace_path().
        raise NotImplementedError(
            "FafuRobotController has no built-in IK; "
            "build a Cartesian-line waypoint list, IK each pose and "
            "call move_jntspace_path()."
        )

    # ------------------------------------------------------------------
    #  Feedback
    # ------------------------------------------------------------------
    def get_joint_values(self, *, prefer_cache: bool = True) -> np.ndarray:
        """Return current manipulator joint angles, in radians."""
        states = self._read_states(self._joint_motor_ids, prefer_cache=prefer_cache)
        out = np.zeros(self.num_joints, dtype=float)
        for i, mid in enumerate(self._joint_motor_ids):
            s = states.get(mid)
            if s is None:
                raise RuntimeError(f"no feedback from motor {mid}")
            out[i] = self._turns_to_rad(s.position)
        return out

    def get_joint_values_raw(self, *, prefer_cache: bool = True) -> List[float]:
        """Return current manipulator joint positions in *turns*."""
        states = self._read_states(self._joint_motor_ids, prefer_cache=prefer_cache)
        return [
            (states[mid].position if states.get(mid) is not None else float("nan"))
            for mid in self._joint_motor_ids
        ]

    def get_joint_velocities(self, *, prefer_cache: bool = True) -> np.ndarray:
        """Return current manipulator joint velocities, in rad/s."""
        states = self._read_states(self._joint_motor_ids, prefer_cache=prefer_cache)
        out = np.zeros(self.num_joints, dtype=float)
        for i, mid in enumerate(self._joint_motor_ids):
            s = states.get(mid)
            if s is None:
                raise RuntimeError(f"no feedback from motor {mid}")
            # velocity is reported in turns/s -> rad/s
            out[i] = self._turns_to_rad(s.velocity)
        return out

    def get_motor_states(self, *, prefer_cache: bool = True) -> Dict[int, "pm.MotorState"]:
        """Return raw :class:`MotorState` objects keyed by motor id."""
        return self._read_states(self._cfg.motor_ids, prefer_cache=prefer_cache)

    def get_pose(self):
        """Return end-effector pose (placeholder).

        Raises
        ------
        NotImplementedError
            Forward kinematics are not built in; compute them from
            :meth:`get_joint_values` using your URDF / wrs model.
        """
        raise NotImplementedError(
            "FafuRobotController has no built-in FK; "
            "use get_joint_values() and an external kinematics model."
        )

    # ------------------------------------------------------------------
    #  Gripper
    # ------------------------------------------------------------------
    # Default gripper speed: 0.3 turns/s = 108 deg/s.  Full Fafu
    # gripper range (~113 deg) finishes in roughly 1 second.
    _GRIPPER_VEL_DEFAULT = 0.3
    _GRIPPER_ACC_DEFAULT = 0.5
    _GRIPPER_TOLERANCE_TURNS = 0.005          # ~ 1.8 deg
    _GRIPPER_STALL_VEL_TPS = 0.005            # < 1.8 deg/s ⇒ "not moving"
    _GRIPPER_STALL_PATIENCE_S = 0.3           # treat as done if stalled this long

    def gripper_control(
        self,
        angle: float,
        effort: Optional[int] = None,
        *,
        is_radians: bool = True,
        vel: float = _GRIPPER_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        block: bool = True,
        timeout: float = 8.0,
        tolerance_deg: float = 1.5,
        effort_threshold: Optional[int] = None,
    ) -> Optional[GraspResult]:
        """Drive the gripper joint to an explicit ``angle`` (Piper-style).

        Unlike the Piper gripper (linear width in metres), the
        Fafu gripper is just another rotational motor, so
        ``angle`` is interpreted as a **joint angle** (radians by
        default).  No "open / close" semantics are applied here -
        the value is sent as-is and clamped by the soft limit.

        On the stock Fafu cfg the convention is::

            limits.7 = -114.98 deg (lower)  ─ closer to fully closed
                        -1.83  deg (upper)  ─ open

        ...so a more *open* gripper is a *less negative* angle.  Use
        :meth:`open_gripper` / :meth:`close_gripper` if you do not
        want to think about which direction is which.

        This signature intentionally mirrors :meth:`piper.PiperArmController.gripper_control`
        (``angle``, ``effort``) so client code can be ported across
        the two arms with minimal changes.  The semantics are slightly
        different though: Piper's firmware does the force-limited
        closure internally, whereas on Fafu the wrapper either
        passes ``effort`` to ``set_pos_vel_tqe`` as a torque cap, or
        polls ``MotorState.torque`` in Python to implement the early
        exit (``effort_threshold``).

        Parameters
        ----------
        angle : float
            Target joint angle.  Required.
        effort : int, optional
            Maximum torque (**raw int16**, as reported by
            :attr:`MotorState.torque`) the firmware is allowed to use
            while following ``angle``.  When ``None`` (default) the
            command goes through ``set_pos_vel_acc`` and inherits the
            motor's internal current limit; when given, the command
            goes through ``set_pos_vel_tqe``.  Typical safe values
            are ``cfg.max_torque_raw`` (full effort) down to a few
            hundred for soft contact.
        is_radians : bool, optional
            Interpret ``angle`` as radians (default) or degrees.
        vel : float, optional
            Velocity limit in turns/s.  Defaults to ``0.3`` turns/s
            (~ 108 deg/s).
        acc : float, optional
            Acceleration limit in turns/s^2.  Ignored when ``effort``
            is provided (``set_pos_vel_tqe`` has no acceleration arg).
        block : bool, optional
            * ``True`` (default): poll the gripper position until it
              reaches the target (within ``tolerance_deg``), stalls,
              ``effort_threshold`` is exceeded, or ``timeout`` elapses.
            * ``False``: just send the command and return immediately.
        timeout : float, optional
            Maximum seconds to wait when ``block`` is True.
        tolerance_deg : float, optional
            "Reached" tolerance in degrees.  Defaults to 1.5 deg.
        effort_threshold : int, optional
            Raw int16 ``|torque|`` value that, when exceeded while
            blocking, causes an early return.  This is the lightweight
            "did I grab something?" check.  When ``None`` (default)
            only position / stall / timeout are used.  When given,
            this method returns a :class:`GraspResult` describing the
            outcome; otherwise it returns ``None`` for backwards
            compatibility with the Piper-style void signature.

        Returns
        -------
        GraspResult or None
            A :class:`GraspResult` iff ``block=True`` AND a stop
            reason was recorded (i.e. ``effort_threshold`` was given,
            or the stall / target conditions fired). Returns ``None``
            otherwise.
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        pos_turns = self._rad_to_turns(angle) if is_radians else angle / 360.0

        if effort is None:
            self._ht.set_pos_vel_acc(
                self._gripper_motor_id,
                pos_turns,
                float(vel),
                float(acc),
                pm.PosUnit.Turns,
            )
        else:
            self._ht.set_pos_vel_tqe(
                self._gripper_motor_id,
                pos_turns,
                float(vel),
                int(effort),
                pm.PosUnit.Turns,
            )

        if not block:
            return None

        # When the caller did not ask for force-aware blocking we keep
        # the legacy return-None behaviour so existing call sites
        # don't suddenly start receiving objects.
        result = self._wait_until_gripper_done(
            pos_turns,
            timeout=float(timeout),
            tolerance_turns=tolerance_deg / 360.0,
            effort_threshold=effort_threshold,
        )
        if effort_threshold is None:
            return None
        return result

    def _gripper_limit_turns(self) -> Tuple[Optional[float], Optional[float]]:
        """Helper: return the gripper's (lo, hi) soft limit in turns."""
        try:
            lim = self._ht.get_position_limit_turns(self._gripper_motor_id)
        except Exception:
            lim = None
        if lim is None:
            return (None, None)
        return (float(lim[0]), float(lim[1]))

    def open_gripper(
        self,
        angle: Optional[float] = None,
        effort: Optional[int] = None,
        *,
        is_radians: bool = True,
        vel: float = _GRIPPER_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        block: bool = True,
        timeout: float = 8.0,
    ) -> None:
        """Open the gripper.

        On Fafu a more *open* gripper corresponds to the
        **upper** soft limit (a less negative angle).  When ``angle``
        is ``None`` (default) the gripper is driven to that upper
        limit, which gives the largest opening allowed by the
        configuration.

        Parameters
        ----------
        angle : float, optional
            Explicit target angle.  When omitted, the upper soft
            limit is used (or +0.25 turns ~ 90 deg if no limit is
            configured).
        effort : int, optional
            Max torque (raw int16) the firmware may use during the
            move; forwarded to :meth:`gripper_control`.  ``None`` keeps
            the legacy ``set_pos_vel_acc`` behaviour.
        is_radians : bool, optional
            Interpret ``angle`` as radians (default) or degrees.
        vel : float, optional
            Velocity limit in turns/s.
        acc : float, optional
            Acceleration limit in turns/s^2.  Ignored when ``effort``
            is provided.
        block : bool, optional
            Block until the gripper reaches the target / stalls /
            ``timeout`` elapses.  Defaults to ``True``.
        timeout : float, optional
            Max seconds to wait when ``block`` is True.
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        if angle is None:
            _, hi_t = self._gripper_limit_turns()
            target_turns = hi_t if hi_t is not None else 0.25
            # Convert turns -> radians so we can reuse gripper_control's
            # full plumbing (effort, blocking, GraspResult-free void path).
            self.gripper_control(
                self._turns_to_rad(target_turns),
                effort,
                is_radians=True,
                vel=vel, acc=acc,
                block=block, timeout=timeout,
            )
        else:
            self.gripper_control(
                angle, effort,
                is_radians=is_radians, vel=vel, acc=acc,
                block=block, timeout=timeout,
            )

    def close_gripper(
        self,
        angle: Optional[float] = None,
        effort: Optional[int] = None,
        *,
        is_radians: bool = True,
        vel: float = _GRIPPER_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        block: bool = True,
        timeout: float = 8.0,
    ) -> None:
        """Close the gripper.

        On Fafu a more *closed* gripper corresponds to the
        **lower** soft limit (a more negative angle).  When ``angle``
        is ``None`` (default) the gripper is driven to that lower
        limit, which gives the tightest grip allowed by the
        configuration.

        Parameters
        ----------
        angle : float, optional
            Explicit target angle.  When omitted, the lower soft
            limit is used (or -0.25 turns ~ -90 deg if no limit is
            configured).
        effort : int, optional
            Max torque (raw int16) the firmware may use during the
            move; forwarded to :meth:`gripper_control`.  ``None`` keeps
            the legacy ``set_pos_vel_acc`` behaviour.  For **force-aware
            grasping with an early stop on contact**, use
            :meth:`grasp` instead.
        is_radians : bool, optional
            Interpret ``angle`` as radians (default) or degrees.
        vel : float, optional
            Velocity limit in turns/s.
        acc : float, optional
            Acceleration limit in turns/s^2.  Ignored when ``effort``
            is provided.
        block : bool, optional
            Block until the gripper reaches the target / stalls /
            ``timeout`` elapses.  Defaults to ``True`` so that a
            grasp action does not get cut short by a subsequent
            ``close_connection()``.
        timeout : float, optional
            Max seconds to wait when ``block`` is True.
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        if angle is None:
            lo_t, _ = self._gripper_limit_turns()
            target_turns = lo_t if lo_t is not None else -0.25
            self.gripper_control(
                self._turns_to_rad(target_turns),
                effort,
                is_radians=True,
                vel=vel, acc=acc,
                block=block, timeout=timeout,
            )
        else:
            self.gripper_control(
                angle, effort,
                is_radians=is_radians, vel=vel, acc=acc,
                block=block, timeout=timeout,
            )

    # Default grasp tuning.  Slower than open/close because we want
    # to feel contact, not slam through it.
    _GRASP_VEL_DEFAULT = 0.15            # turns/s (~ 54 deg/s)
    _GRASP_FORCE_THRESHOLD_DEFAULT = 500 # raw int16, conservative

    def grasp(
        self,
        *,
        target_angle: Optional[float] = None,
        is_radians: bool = True,
        force_threshold: int = _GRASP_FORCE_THRESHOLD_DEFAULT,
        effort: Optional[int] = None,
        vel: float = _GRASP_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        timeout: float = 5.0,
        min_close_deg: float = 3.0,
    ) -> GraspResult:
        """Close the gripper until contact is detected (Piper-style force grasp).

        This is the Fafu-side equivalent of Piper's
        ``close_gripper(effort=...)``: it commands the gripper toward
        the closing direction and stops as soon as the **measured**
        torque exceeds ``force_threshold`` (or the gripper plateaus
        at finite stiffness against an object).

        Because Fafu's firmware does not implement the position
        + effort loop itself, the detection logic runs in Python by
        polling ``MotorState.torque`` from the background poller
        (``start_state_polling``).  This is good enough for catching
        objects but is **not** a substitute for a real wrist F/T
        sensor.

        Parameters
        ----------
        target_angle : float, optional
            Maximum closure angle.  When ``None`` (default) the lower
            soft limit from ``cfg.limits[gripper_motor_id]`` is used,
            which is the tightest grip the configuration allows.
        is_radians : bool, optional
            Interpret ``target_angle`` as radians (default) or degrees.
        force_threshold : int, optional
            Raw int16 ``|torque|`` value that, when exceeded, ends the
            grasp early and marks it as successful
            (``reason='detected_object_force'``).  Calibrate by
            running an empty close and noting the steady-state torque.
        effort : int, optional
            Torque cap passed to the **firmware** via
            ``set_pos_vel_tqe`` (raw int16).  When ``None`` the
            firmware uses its default current limit.  Pass this if
            you also want a hard hardware-level torque ceiling, not
            just the soft Python-side trip wire.
        vel : float, optional
            Closing velocity in turns/s.  Defaults to 0.15 turns/s
            (deliberately slower than ``open_gripper`` so that
            contact is gentle).
        acc : float, optional
            Acceleration limit in turns/s^2.  Ignored when ``effort``
            is provided (``set_pos_vel_tqe`` has no acceleration arg).
        timeout : float, optional
            Maximum wall-clock seconds to wait.
        min_close_deg : float, optional
            Minimum closure (in degrees) before a stall counts as
            "object grasped".  Anything below this is reported as
            ``'no_movement'`` instead, to catch cases where the
            command never reached the motor or the jaws were already
            closed.  Defaults to 3 deg.

        Returns
        -------
        GraspResult
            Always returned; inspect ``.grasped`` and ``.reason``
            to decide what happened.

        Raises
        ------
        RuntimeError
            If this controller was built without a gripper.

        Examples
        --------
        Empty close (will report no object)::

            r = arm.grasp(force_threshold=500)
            print(r.grasped, r.reason, r.peak_torque_raw)

        Grasp with a hard hardware torque cap as well::

            r = arm.grasp(force_threshold=600, effort=800, vel=0.1)
            if not r.grasped:
                arm.open_gripper()
                raise RuntimeError(f"grasp failed: {r.reason}")
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )

        if target_angle is None:
            lo_t, _ = self._gripper_limit_turns()
            target_turns = lo_t if lo_t is not None else -0.25
        else:
            target_turns = (
                self._rad_to_turns(target_angle) if is_radians
                else target_angle / 360.0
            )

        if effort is None:
            self._ht.set_pos_vel_acc(
                self._gripper_motor_id,
                target_turns,
                float(vel),
                float(acc),
                pm.PosUnit.Turns,
            )
        else:
            self._ht.set_pos_vel_tqe(
                self._gripper_motor_id,
                target_turns,
                float(vel),
                int(effort),
                pm.PosUnit.Turns,
            )

        return self._wait_until_gripper_done(
            target_turns,
            timeout=float(timeout),
            effort_threshold=int(force_threshold),
            min_progress_turns=max(0.0, float(min_close_deg)) / 360.0,
        )

    def release(
        self,
        *,
        target_angle: Optional[float] = None,
        is_radians: bool = True,
        vel: float = _GRIPPER_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        timeout: float = 5.0,
        block: bool = True,
    ) -> None:
        """Counterpart to :meth:`grasp`: drive the gripper open to drop the object.

        Thin alias of :meth:`open_gripper` — provided so that grasp /
        release form a symmetric pair in user code.
        """
        self.open_gripper(
            angle=target_angle,
            is_radians=is_radians,
            vel=vel,
            acc=acc,
            block=block,
            timeout=timeout,
        )

    def get_gripper_state(self) -> "pm.MotorState":
        """Return the latest :class:`MotorState` of the gripper motor."""
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        s = self._ht.get_cached_state(self._gripper_motor_id)
        if s is None:
            s = self._ht.read_motor_state(self._gripper_motor_id, 0.1)
        if s is None:
            raise RuntimeError("no feedback from gripper motor")
        return s

    # ------------------------------------------------------------------
    #  Soft limits
    # ------------------------------------------------------------------
    def set_limit(
        self,
        motor_id: int,
        lo: float,
        hi: float,
        *,
        is_radians: bool = True,
    ) -> None:
        """Enable a soft position limit for ``motor_id``.

        ``lo`` / ``hi`` are interpreted as radians (default) or degrees.
        """
        if motor_id not in self._cfg.motor_ids:
            raise ValueError(f"motor {motor_id} is not in cfg.motor_ids")
        if lo > hi:
            raise ValueError(f"lo ({lo}) > hi ({hi})")
        unit = pm.PosUnit.Radians if is_radians else pm.PosUnit.Degrees
        self._ht.enable_position_limit(motor_id, float(lo), float(hi), unit)
        # Mirror into cfg.limits (kept in turns) so subsequent saves work.
        lo_t = pm.to_turns(float(lo), unit)
        hi_t = pm.to_turns(float(hi), unit)
        try:
            self._cfg.limits[motor_id] = (lo_t, hi_t)
        except Exception:
            pass

    def get_limit(
        self,
        motor_id: int,
        *,
        is_radians: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Return ``(lo, hi)`` for the given motor or ``None`` if unset."""
        r = self._ht.get_position_limit_turns(motor_id)
        if r is None:
            return None
        lo_t, hi_t = r
        if is_radians:
            return (self._turns_to_rad(lo_t), self._turns_to_rad(hi_t))
        return (lo_t * 360.0, hi_t * 360.0)

    def disable_limit(self, motor_id: int) -> None:
        """Disable the soft limit for ``motor_id``."""
        self._ht.disable_position_limit(motor_id)
        try:
            if motor_id in self._cfg.limits:
                del self._cfg.limits[motor_id]
        except Exception:
            pass

    def clear_limits(self) -> None:
        """Disable every soft position limit."""
        self._ht.clear_all_position_limits()
        try:
            self._cfg.limits.clear()
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Safety
    # ------------------------------------------------------------------
    def emergency_stop(self) -> None:
        """Immediately stop every motor (free-spin mode)."""
        for mid in self._cfg.motor_ids:
            try:
                self._ht.stop(mid)
            except Exception:
                pass
        print("[FafuRobot] EMERGENCY STOP issued (all motors → mode 0x00).")

    def resume(self) -> None:
        """Re-enable position control after an emergency stop."""
        self.enable()

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    def get_status(self):
        """Return the latest :class:`Stats` object from the driver."""
        return self._ht.get_stats()

    def get_can_status(self):
        """Return the latest :class:`CanStatus` (live read)."""
        return self._ht.read_can_status()

    def reset_zero(self, motor_id: int, *, confirm: bool = False) -> None:
        """Set the current position of ``motor_id`` as the new zero.

        ``confirm=True`` is required because this is destructive.
        """
        if not confirm:
            raise RuntimeError("reset_zero is destructive; pass confirm=True")
        if motor_id not in self._cfg.motor_ids:
            raise ValueError(f"motor {motor_id} not in cfg.motor_ids")
        self._ht.reset_zero(motor_id)

    # ------------------------------------------------------------------
    #  Connection lifecycle
    # ------------------------------------------------------------------
    def close_connection(
        self,
        *,
        joint_release: str = "stop",
        gripper_release: str = "brake",
    ) -> None:
        """Stop background tasks and close the serial port.

        Parameters
        ----------
        joint_release : {"stop", "brake", "hold"}, optional
            What to do with the manipulator joints before closing.

            * ``"stop"`` (default): mode 0x00, PWM off, joints free
              to be moved by hand. Safest if a human will reposition
              the arm. Heavy joints may sag slightly under gravity.
            * ``"brake"``: mode 0x0F, short-circuit damping. No
              current required, but resists motion. Use this if you
              want the pose to roughly hold without keeping the
              motors energised.
            * ``"hold"``: keep mode 0x0A, motors actively hold their
              last commanded position. Uses current; do not leave
              like this for long.
        gripper_release : {"stop", "brake", "hold"}, optional
            Same options for the gripper.  Defaults to ``"brake"`` so
            that the gripper does **not** drift open the moment the
            program disconnects (the previous default of ``"stop"``
            caused exactly that issue).
        """
        # If a servo session is still open, end it gracefully BEFORE we
        # stop polling / disable async RX. servo_end clears the firmware
        # watchdog (which would otherwise fire as soon as we stop
        # sending frames) and parks the joints in ``joint_release`` mode.
        if self._servo_active:
            try:
                self.servo_end(finish_mode=joint_release)
            except Exception as e:
                print(f"[FafuRobot] close_connection: servo_end fallback failed: {e}")

        try:
            if self._ht.is_polling():
                self._ht.stop_state_polling()
        except Exception:
            pass
        try:
            if self._ht.is_async_rx():
                self._ht.disable_async_rx()
        except Exception:
            pass

        valid = {"stop", "brake", "hold"}
        if joint_release not in valid:
            raise ValueError(f"joint_release must be one of {valid}")
        if gripper_release not in valid:
            raise ValueError(f"gripper_release must be one of {valid}")

        mode_map = {
            "stop":  self.MODE_STOP,
            "brake": self.MODE_BRAKE,
            "hold":  self.MODE_POSITION,
        }
        for mid in self._managed_motor_ids():
            policy = (gripper_release if (self._has_gripper
                                          and mid == self._gripper_motor_id)
                      else joint_release)
            try:
                if policy == "stop":
                    self._ht.stop(mid)
                else:
                    # brake (0x0F) or hold (0x0A) — both are mode changes,
                    # not the dedicated stop()/brake() RPCs.
                    self._ht.set_motor_mode(mid, mode_map[policy])
            except Exception:
                pass
        try:
            self._ht.close()
        except Exception:
            pass
        print(f"[FafuRobot] connection closed "
              f"(joints={joint_release}, gripper={gripper_release}).")

    def __enter__(self) -> "FafuRobotController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_connection()

    # ==================================================================
    #  Internals
    # ==================================================================
    @staticmethod
    def _rad_to_turns(rad: float) -> float:
        return float(rad) / _TWO_PI

    @staticmethod
    def _turns_to_rad(turns: float) -> float:
        return float(turns) * _TWO_PI

    @staticmethod
    def _clamp_speed(speed: int) -> int:
        s = int(speed)
        if s < 1:
            s = 1
        if s > 100:
            s = 100
        return s

    @staticmethod
    def _resolve_cfg_path(cfg_path: str) -> str:
        if os.path.isabs(cfg_path) or os.path.exists(cfg_path):
            return cfg_path
        candidate = os.path.join(_HERE, cfg_path)
        if os.path.exists(candidate):
            return candidate
        return cfg_path

    def _pick_serial_port(self, preferred: Optional[str]) -> str:
        pref = (preferred or "").strip()
        is_auto = (not pref) or pref.lower() == "auto"
        try:
            usb_ports = pm.find_likely_debug_boards()
        except Exception as e:
            print(f"[FafuRobot] failed to enumerate ports: {e}")
            return preferred or ""
        if not is_auto:
            for p in usb_ports:
                if p.port == pref:
                    return pref
            print(
                f"[FafuRobot] preferred port {pref!r} not found; "
                f"falling back to auto"
            )
        if not usb_ports:
            raise RuntimeError(
                "no USB debug board detected; check the cable and try again"
            )
        return usb_ports[0].port

    def _warmup_can_bus(self) -> None:
        """Prime CAN RX with lightweight reads before strict precheck."""
        ids = list(self._cfg.motor_ids)
        if not ids:
            return
        try:
            cs = self._ht.read_can_status()
            if cs is not None:
                print(
                    f"[FafuRobot] CAN status: fault={getattr(cs, 'fault', '?')}, "
                    f"lec={getattr(cs, 'lec', '?')}, "
                    f"tx_err={getattr(cs, 'tx_err_count', '?')}, "
                    f"rx_err={getattr(cs, 'rx_err_count', '?')}"
                )
        except Exception as exc:
            print(f"[FafuRobot] warning: read_can_status failed: {exc}")

        for pass_idx in range(2):
            for mid in sorted(ids, reverse=True):
                try:
                    self._ht.read_motor_state(int(mid), 0.15)
                except Exception:
                    pass
            if pass_idx == 0:
                time.sleep(0.08)

    def _precheck_communication(self) -> None:
        """Verify every managed motor responds; retry for cold CAN / port reopen."""
        round_timeouts = (0.6, 1.0, 1.5)
        max_rounds = len(round_timeouts)
        pending = list(self._managed_motor_ids())
        bad: List[int] = []

        for attempt in range(max_rounds):
            per_read_timeout = round_timeouts[attempt]
            if attempt > 0:
                delay_s = 0.3 * attempt
                print(
                    f"[FafuRobot] precheck: {len(bad)} motor(s) missing, "
                    f"retry {attempt + 1}/{max_rounds} after {delay_s:.1f}s "
                    f"(timeout={per_read_timeout:.1f}s) ..."
                )
                time.sleep(delay_s)
                self._warmup_can_bus()
            bad = []
            for mid in pending:
                s = self._ht.read_motor_state(mid, per_read_timeout)
                if s is None:
                    bad.append(mid)
            if not bad:
                if attempt > 0:
                    print("[FafuRobot] precheck: all motors responding after retry.")
                return
            pending = bad

        ok_ids = [m for m in self._managed_motor_ids() if m not in bad]
        print(
            f"[FafuRobot] precheck failed on {self._port!r}: "
            f"responding={ok_ids}, missing={bad}"
        )
        self._print_motor_diagnostic(prefix="precheck failed; ")
        raise RuntimeError(
            f"motors {bad} did not respond on {self._port!r} "
            f"(after {max_rounds} attempts); only M1 alive usually means "
            f"CAN downstream power/wiring fault — power-cycle the arm hub, "
            f"then run diag_motors.py --port {self._port} --timeout 1.0"
        )

    def _safe_close_ht_on_init_failure(self) -> None:
        """Release the serial port if __init__ fails partway through."""
        if self._ht is None:
            return
        try:
            if hasattr(self._ht, "is_polling") and self._ht.is_polling():
                self._ht.stop_state_polling()
        except Exception:
            pass
        try:
            if hasattr(self._ht, "is_async_rx") and self._ht.is_async_rx():
                self._ht.disable_async_rx()
        except Exception:
            pass
        try:
            self._ht.close()
        except Exception:
            pass
        self._ht = None

    def _switch_mode_all(
        self,
        mode: int,
        *,
        label: str,
        max_retry: int = 3,
    ) -> bool:
        """Drive every motor to ``mode``, verifying the actual state.

        Returns True as soon as every motor's *measured* state is at
        ``mode``.  This intentionally trusts the post-switch read
        rather than ``set_motor_mode``'s return value, because that
        return value can legitimately be ``None`` on a busy /
        freshly-reopened port even when the mode change succeeded
        (the binding's internal "switch -> sleep -> read confirm"
        sequence is timing-sensitive).  We back-stop by issuing a
        fresh read on every motor that the binding flagged as
        failing, and only treat it as a real failure if the read
        confirms a wrong mode.
        """
        for attempt in range(1, max_retry + 1):
            failed: List[int] = []
            for mid in self._managed_motor_ids():
                s = self._ht.set_motor_mode(mid, mode)
                if s is None or int(s.mode) != mode:
                    failed.append(mid)

            # ★ Post-failure verify: do a fresh read on every "failed"
            # motor to filter out binding false negatives. If the read
            # confirms the target mode is already there, the motor is
            # actually fine — common case after a sync-read timeout
            # inside set_motor_mode(0x0A).
            if failed:
                time.sleep(0.03)
                really_failed: List[int] = []
                for mid in failed:
                    s = self._ht.read_motor_state(mid, 0.3)
                    if s is None or int(s.mode) != mode:
                        really_failed.append(mid)
                failed = really_failed

            if not failed:
                return True
            if attempt < max_retry:
                print(
                    f"[FafuRobot] switch to {label}: retry {attempt + 1}, "
                    f"failed motors = {failed}"
                )
                time.sleep(0.1)
        return False

    def _validate_joint_angles(
        self,
        joint_angles: Iterable[float],
        is_radians: bool,
    ) -> List[float]:
        """Validate length and convert input angles to **turns**."""
        arr = np.asarray(list(joint_angles), dtype=float)
        if arr.size != self.num_joints:
            raise ValueError(
                f"expected {self.num_joints} joint values, got {arr.size}"
            )
        if not is_radians:
            return [v / 360.0 for v in arr]  # degrees -> turns
        return [self._rad_to_turns(v) for v in arr]

    def _read_states(
        self,
        motor_ids: Iterable[int],
        *,
        prefer_cache: bool,
    ) -> Dict[int, "pm.MotorState"]:
        out: Dict[int, "pm.MotorState"] = {}
        ids = list(motor_ids)
        if prefer_cache:
            for mid in ids:
                s = self._ht.get_cached_state(mid)
                if s is not None:
                    out[mid] = s
            missing = [m for m in ids if m not in out]
            if not missing:
                return out
        else:
            missing = ids
        # Fall back to a synchronous read for whatever is missing.
        for mid in missing:
            s = self._ht.read_motor_state(mid, 0.1)
            if s is not None:
                out[mid] = s
        return out

    # Minimum closure (in turns) before a stall counts as "grasped object"
    # rather than "no movement / command never took effect".
    _GRIPPER_MIN_PROGRESS_TURNS = 0.008   # ~ 2.9 deg

    def _wait_until_gripper_done(
        self,
        target_turns: float,
        *,
        timeout: float = 8.0,
        tolerance_turns: Optional[float] = None,
        effort_threshold: Optional[int] = None,
        min_progress_turns: Optional[float] = None,
    ) -> GraspResult:
        """Block until the gripper reaches ``target_turns``, stalls,
        ``|torque| >= effort_threshold``, or ``timeout`` elapses.

        Returns a :class:`GraspResult` regardless of why we stopped;
        callers that don't care can simply ignore the return value.

        We treat the move as "done" if any of:

        * ``|position - target| <= tolerance_turns`` (reached target).
        * ``|torque| >= effort_threshold`` (force trip, if given).
        * ``|velocity| < stall threshold`` for
          ``_GRIPPER_STALL_PATIENCE_S`` and the gripper has moved at
          least ``min_progress_turns`` (grasped something).
        * Stall without enough movement → ``'no_movement'``.
        * Wall-clock ``timeout`` exceeded → ``'timeout'``.
        """
        if tolerance_turns is None:
            tolerance_turns = self._GRIPPER_TOLERANCE_TURNS
        if min_progress_turns is None:
            min_progress_turns = self._GRIPPER_MIN_PROGRESS_TURNS

        t0 = time.monotonic()
        deadline = t0 + max(0.05, float(timeout))

        # Capture the starting position so we can report closed_deg
        # and classify "no movement" vs "real grasp".
        start_state = self._ht.get_cached_state(self._gripper_motor_id)
        if start_state is None:
            start_state = self._ht.read_motor_state(self._gripper_motor_id, 0.05)
        start_pos = start_state.position if start_state is not None else float("nan")
        last_pos = start_pos

        stall_since: Optional[float] = None
        peak_torque = 0

        while True:
            now = time.monotonic()
            if now >= deadline:
                return self._make_grasp_result(
                    reason="timeout", grasped=False,
                    last_pos=last_pos, start_pos=start_pos,
                    peak_torque=peak_torque, duration=now - t0,
                )

            s = self._ht.get_cached_state(self._gripper_motor_id)
            if s is None:
                s = self._ht.read_motor_state(self._gripper_motor_id, 0.05)

            if s is not None:
                last_pos = s.position
                t_raw = int(abs(s.torque))
                if t_raw > peak_torque:
                    peak_torque = t_raw

                if effort_threshold is not None and t_raw >= int(effort_threshold):
                    return self._make_grasp_result(
                        reason="detected_object_force", grasped=True,
                        last_pos=last_pos, start_pos=start_pos,
                        peak_torque=peak_torque, duration=now - t0,
                    )

                if abs(s.position - target_turns) <= tolerance_turns:
                    return self._make_grasp_result(
                        reason="reached_target", grasped=False,
                        last_pos=last_pos, start_pos=start_pos,
                        peak_torque=peak_torque, duration=now - t0,
                    )

                if abs(s.velocity) < self._GRIPPER_STALL_VEL_TPS:
                    if stall_since is None:
                        stall_since = now
                    elif now - stall_since >= self._GRIPPER_STALL_PATIENCE_S:
                        progress = abs(s.position - start_pos)
                        if progress >= min_progress_turns:
                            return self._make_grasp_result(
                                reason="detected_object_stall", grasped=True,
                                last_pos=last_pos, start_pos=start_pos,
                                peak_torque=peak_torque, duration=now - t0,
                            )
                        return self._make_grasp_result(
                            reason="no_movement", grasped=False,
                            last_pos=last_pos, start_pos=start_pos,
                            peak_torque=peak_torque, duration=now - t0,
                        )
                else:
                    stall_since = None

            time.sleep(0.02)

    def _make_grasp_result(
        self,
        *,
        reason: str,
        grasped: bool,
        last_pos: float,
        start_pos: float,
        peak_torque: int,
        duration: float,
    ) -> GraspResult:
        if math.isnan(last_pos) or math.isnan(start_pos):
            closed_deg = 0.0
            angle_rad = float("nan") if math.isnan(last_pos) else self._turns_to_rad(last_pos)
        else:
            closed_deg = abs(last_pos - start_pos) * 360.0
            angle_rad = self._turns_to_rad(last_pos)
        return GraspResult(
            grasped=grasped,
            reason=reason,
            angle_rad=angle_rad,
            closed_deg=closed_deg,
            peak_torque_raw=int(peak_torque),
            duration_s=float(duration),
        )

    def _build_many_cmds_holding_others(
        self,
        targets_turns: Dict[int, float],
        *,
        vel_rps: float,
    ) -> List["pm.ManyMotorCmd"]:
        """Build a ``set_many_pos_vel_tqe`` payload for all motors.

        Motors *not* listed in ``targets_turns`` (e.g. the gripper
        when issuing a ``move_j``) are commanded to hold their
        current position with zero velocity.
        """
        cmds: List[pm.ManyMotorCmd] = []
        max_torque = int(self._cfg.max_torque_raw)
        for mid in self._cfg.motor_ids:
            if mid in targets_turns:
                cmds.append(
                    pm.ManyMotorCmd(mid, float(targets_turns[mid]),
                                    float(vel_rps), max_torque)
                )
            else:
                s = self._ht.get_cached_state(mid)
                if s is None:
                    s = self._ht.read_motor_state(mid, 0.1)
                hold_pos = s.position if s is not None else 0.0
                cmds.append(pm.ManyMotorCmd(mid, hold_pos, 0.0, max_torque))
        return cmds

    def _move_scurve(
        self,
        targets_turns: Dict[int, float],
        *,
        speed_pct: int,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """S-curve trajectory + ``run_control_loop``, mirrors the
        reference :func:`arm_multi_joint_example.move_scurve`."""
        rate_hz = max(10.0, float(self._cfg.control_rate_hz) or 100.0)
        v_avg_target = (speed_pct / 100.0) * _VEL_AVG_MAX_TPS

        # 1) capture starting positions for *every* motor (so we can
        #    hold non-target ones, e.g. the gripper).
        start_pos: Dict[int, float] = {}
        for mid in self._cfg.motor_ids:
            s = self._ht.get_cached_state(mid)
            if s is None:
                s = self._ht.read_motor_state(mid, 0.1)
            if s is None:
                raise RuntimeError(
                    f"could not read motor {mid} starting position; "
                    f"aborting move_j for safety"
                )
            start_pos[mid] = s.position

        # 2) adaptive segment time based on the largest delta.
        max_abs_dpos = 0.0
        for mid, tgt in targets_turns.items():
            max_abs_dpos = max(max_abs_dpos, abs(tgt - start_pos[mid]))

        dt_s = max(_DT_MIN_S, float(self._cfg.trajectory_dt_s) or 1.0)
        if max_abs_dpos > 1e-5:
            dt_target = max_abs_dpos / max(v_avg_target, 1e-3)
            dt_s = max(_DT_MIN_S, dt_target)

        # Plan: per-motor (delta, signed peak velocity).
        plans: Dict[int, Tuple[float, float]] = {}
        for mid, tgt in targets_turns.items():
            dpos = tgt - start_pos[mid]
            if abs(dpos) < 1e-5:
                plans[mid] = (0.0, 0.0)
                continue
            v_avg = abs(dpos) / dt_s
            v_peak = min(_VEL_AVG_MAX_TPS, v_avg) * (math.pi / 2.0)
            plans[mid] = (dpos, math.copysign(v_peak, dpos))

        total_ticks  = max(1, int(dt_s * rate_hz))
        settle_ticks = max(1, int(_SETTLE_MS * rate_hz / 1000.0))
        last_tick    = total_ticks + settle_ticks
        max_mid      = max(self._cfg.motor_ids)
        max_torque   = int(self._cfg.max_torque_raw)

        # Cache once so we do not allocate Python lists every tick.
        all_ids = list(self._cfg.motor_ids)

        def on_tick(tick: int, _dt_ms: float) -> bool:
            if tick >= last_tick:
                return False

            alpha = min(1.0, tick / total_ticks)
            smooth = 0.5 * (1.0 - math.cos(math.pi * alpha))
            vel_factor = math.sin(math.pi * alpha)

            cmds: List[pm.ManyMotorCmd] = []
            for mid in all_ids:
                if mid in plans and plans[mid][1] != 0.0:
                    dpos, v_peak_signed = plans[mid]
                    desired = start_pos[mid] + smooth * dpos
                    v_inst = v_peak_signed * vel_factor
                    cmds.append(pm.ManyMotorCmd(mid, desired, v_inst, max_torque))
                else:
                    # Motors with no target (or zero delta) hold their start.
                    cmds.append(
                        pm.ManyMotorCmd(mid, start_pos[mid], 0.0, max_torque)
                    )

            self._ht.set_many_pos_vel_tqe(
                cmds, pm.PosUnit.Turns, max_mid, 0.002,
            )
            return True

        # The control loop runs on the C++ side with GIL released.
        rc = self._ht.run_control_loop(
            rate_hz,
            list(self._cfg.motor_ids),
            on_tick,
            abort_check=abort_check,
            on_exception=lambda msg: print(f"[FafuRobot] ctrl-loop exception: {msg}"),
            stop_on_finish=False,
            stop_on_abort=True,
        )
        if rc == 1:
            raise RuntimeError("move_j aborted (abort_check returned True)")
        if rc == 2:
            print("[FafuRobot] warning: control loop exited abnormally")


# ============================================================================
#  Demo
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FafuRobotController demo (Piper-style high-level API)"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(_HERE, "robot.cfg"),
        help="path to robot.cfg",
    )
    parser.add_argument(
        "--gripper-id",
        type=int,
        default=None,
        help="motor id of the gripper (omit if no gripper)",
    )
    parser.add_argument("--speed", type=int, default=15,
                        help="speed percent for the demo (default 15)")
    parser.add_argument("--no-move", action="store_true",
                        help="only read state, do not issue any motion")
    args = parser.parse_args()

    has_gripper = args.gripper_id is not None
    arm = FafuRobotController(
        cfg_path=args.config,
        has_gripper=has_gripper,
        gripper_motor_id=args.gripper_id,
    )

    def _print(*a, **kw):
        # The 50Hz polling thread can interleave its log output with
        # ours. Sleep briefly so its line is fully flushed first,
        # then print our own line with explicit flush.
        time.sleep(0.02)
        kw.setdefault("flush", True)
        print(*a, **kw)

    try:
        _print("\n--- Initial state ---")
        q = arm.get_joint_values()
        _print(f"  joint angles (deg): {np.degrees(q).round(2).tolist()}")
        _print(f"  is_enabled        : {arm.is_enabled}")
        _print(f"  stats             : {arm.get_status().to_string()}")

        if args.no_move:
            _print("\n[--no-move] skipping motion demo.")
        else:
            _print("\n--- Going home (joint zero) ---")
            arm.go_home(speed=args.speed, block=True)

            _print("\n--- Tiny sinusoidal demo on the last joint (block=False) ---")
            base = arm.get_joint_values()
            # When has_gripper=True the gripper motor is *not* in
            # joint_motor_ids, so we drive J6 (the last manipulator
            # joint).  When there is no gripper, the loop drives the
            # 7th motor as before.
            for i in range(40):
                base[-1] = math.radians(10.0) * math.sin(i * math.pi / 20.0)
                arm.move_j(base, speed=args.speed, block=False)
                time.sleep(0.05)

            if has_gripper:
                _print("\n--- Gripper demo (open -> close, using soft limits) ---")
                arm.open_gripper()        # blocking, default vel 0.3 turns/s
                gs = arm.get_gripper_state()
                _print(f"  after open : gripper "
                       f"{math.degrees(arm._turns_to_rad(gs.position)):+.2f} deg")
                time.sleep(0.3)
                arm.close_gripper()
                gs = arm.get_gripper_state()
                _print(f"  after close: gripper "
                       f"{math.degrees(arm._turns_to_rad(gs.position)):+.2f} deg")

        _print("\n--- Final state ---")
        q = arm.get_joint_values()
        _print(f"  joint angles (deg): {np.degrees(q).round(2).tolist()}")
        if has_gripper:
            gs = arm.get_gripper_state()
            _print(f"  gripper           : "
                   f"{math.degrees(arm._turns_to_rad(gs.position)):+.2f} deg")
    finally:
        arm.close_connection()
