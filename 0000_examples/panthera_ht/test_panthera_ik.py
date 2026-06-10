#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from wrs0 import wd, mgm, rm
from wrs0.robot_sim.robots.panthera_ht.panthera_ht import PantheraHTSglArm

base = wd.World(
    cam_pos=[1.2, 1.2, 0.8],
    lookat_pos=[0.1, 0.0, 0.2]
)

mgm.gen_frame().attach_to(base)

robot = PantheraHTSglArm(enable_cc=False)

# 初始姿态
start_conf = np.array([0.0, 0.3, 0.3, 0.0, 0.0, 0.0])
robot.goto_given_conf(start_conf)

# 显示初始姿态，半透明
robot.gen_meshmodel(
    alpha=0.3,
    toggle_jnt_frames=True,
    toggle_tcp_frame=True
).attach_to(base)

# 目标末端位置
tgt_pos = np.array([0.25, 0.0, 0.18])

# 目标末端姿态
# 先用单位矩阵测试，表示 TCP 坐标系和世界坐标系方向一致
tgt_rotmat = np.eye(3)

# 显示目标坐标系
mgm.gen_frame(
    pos=tgt_pos,
    rotmat=tgt_rotmat,
    ax_length=0.08
).attach_to(base)

print("Start conf:")
print(start_conf)

print("Target pos:")
print(tgt_pos)

# 求 IK
goal_conf = robot.ik(
    tgt_pos=tgt_pos,
    tgt_rotmat=tgt_rotmat,
    seed_jnt_values=start_conf,
    toggle_dbg=True
)

print("IK result:")
print(goal_conf)

if goal_conf is not None:
    robot.goto_given_conf(goal_conf)

    # 显示 IK 后姿态，不透明
    robot.gen_meshmodel(
        alpha=1.0,
        toggle_jnt_frames=True,
        toggle_tcp_frame=True
    ).attach_to(base)

    tcp_pos, tcp_rotmat = robot.fk(goal_conf)
    print("FK TCP pos after IK:")
    print(tcp_pos)

    print("Position error:")
    print(np.linalg.norm(tcp_pos - tgt_pos))

else:
    print("IK failed. Try changing target position or target rotation.")

base.run()