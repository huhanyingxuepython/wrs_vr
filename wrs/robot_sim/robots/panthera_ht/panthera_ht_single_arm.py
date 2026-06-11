#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/06/10
# @Author : HuHanYing

import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
from wrs.robot_sim.manipulators.panthera_ht.panthera_ht import PantheraHT
from wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper import PantheraGripper


class PantheraHTSglArm(sari.SglArmRobotInterface):
    """
    Panthera-HT single arm wrapper for WRS.

    Panthera-HT 机械臂整合类：基于 PantheraHT 本体与 PantheraGripper 夹爪。
    结构参照 PiperSglArm：本体层负责运动学，夹爪层负责开合与 TCP acting center。
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 name="panthera_ht_arm",
                 enable_cc=True):

        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(6)

        self.manipulator = PantheraHT(
            pos=self.pos,
            rotmat=self.rotmat,
            name=name + "_arm",
            enable_cc=enable_cc
        )
        self.manipulator.home_conf = home_conf

        # 与 PiperSglArm 一致：在单臂封装层处理末端坐标补偿。
        compensation_rotmat = rm.rotmat_from_euler(0.0, 0.0, math.pi / 2.0)
        corrected_rotmat = self.manipulator.gl_flange_rotmat @ compensation_rotmat

        self.end_effector = PantheraGripper(
            pos=self.manipulator.gl_flange_pos,
            rotmat=corrected_rotmat,
            name=name + "_panthera_gripper",
            jaw_range=np.array([0.0, 0.012]),
            close_bias=0.006,
            use_palm_mesh=True
        )
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """设置自碰撞检测，风格参照 PiperSglArm。"""
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = None
        if self.manipulator.jlc.jnts[5].lnk.cmodel is not None:
            ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        el0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)

        # 使用“中后段连杆 + 末端手指”对“基座/近基座连杆”的组合。
        from_list = [ml3, ml4, el0, el1]
        if ml5 is not None:
            from_list.append(ml5)
        if self.end_effector.jlc.anchor.lnk_list[0].cmodel is not None:
            mlee = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])
            from_list.append(mlee)
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2]
        self.cc.dynamic_ext_list = []

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        results = self.manipulator.fk(
            jnt_values=jnt_values,
            toggle_jacobian=toggle_jacobian,
            update=update
        )
        if update:
            self.update_end_effector()
        return results

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.basis.robot_math as rm

    base = wd.World(cam_pos=[1.2, 1.2, 0.8], lookat_pos=[0, 0, 0.15])
    mgm.gen_frame().attach_to(base)

    robot = PantheraHTSglArm(enable_cc=False)

    test_conf = np.array([0.0, 0.3, 0.3, 0.0, 0.0, 0.0])
    robot.goto_given_conf(test_conf)

    robot.gen_meshmodel(
        toggle_jnt_frames=True,
        toggle_tcp_frame=True
    ).attach_to(base)

    base.run()