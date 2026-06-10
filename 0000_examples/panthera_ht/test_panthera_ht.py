#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from wrs0 import wd, mgm, rm
from wrs0.robot_sim.robots.panthera_ht.panthera_ht import PantheraHTSglArm

base = wd.World(cam_pos=[1.2, 1.2, 0.8], lookat_pos=[0, 0, 0.2])
mgm.gen_frame().attach_to(base)

robot = PantheraHTSglArm(enable_cc=False)

# 测试 1：零位
conf0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
robot.goto_given_conf(conf0)
robot.gen_meshmodel(alpha=0.3, toggle_jnt_frames=True).attach_to(base)

# 测试 2：小角度姿态
conf1 = np.array([0.3, 0.4, 0.4, 0.3, 0.2, 0.2])
robot.goto_given_conf(conf1)
robot.gen_meshmodel(alpha=1.0, toggle_tcp_frame=True).attach_to(base)

print("Current joint values:")
print(robot.get_jnt_values())

base.run()