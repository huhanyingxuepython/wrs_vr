#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/06/10
# @Author : HuHanYing

import numpy as np
import wrs.modeling.geometric_model as gm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
from wrs.robot_sim.manipulators.panthera_ht.panthera_ht import PantheraHT
from wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper import (
    PantheraGripper,
)


class PantheraHTSglArm(sari.SglArmRobotInterface):
    """
    Panthera-HT 6-DoF + Panthera 双指夹爪整合类。

    与 ``PiperSglArm`` / ``OpenSglArm`` 一致：在 manipulator 法兰挂上
    ``PantheraGripper`` 作为 end_effector，TCP 由夹爪 acting center 接管，
    高层接口统一走 ``SglArmRobotInterface``。
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 name="robot_panthera_ht", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = PantheraHT(pos=self.pos,
                                      rotmat=self.rotmat,
                                      name=name + "_arm",
                                      enable_cc=False)
        self.end_effector = PantheraGripper(
            pos=self.manipulator.gl_flange_pos,
            rotmat=self.manipulator.gl_flange_rotmat,
            name=name + "_gripper")

        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """Conservative CC setup for PantheraHT + gripper.

        The Panthera meshes around the wrist/flange/gripper overlap slightly
        in the default pose, so do not register internal self-collision pairs
        here. Keep external masks enabled so table/part/obstacle collision
        checks still work during planning.
        """
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        # link6 mesh 已在 PantheraHT 内部禁用以避免与夹爪 palm 重叠，故不入 CC

        mlee = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])
        el0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)

        self.cc.enable_extcd_by_id_list(
            id_list=[ml1, ml2, ml3, ml4, mlee, el0, el1], type="from")
        self.cc.enable_innercd_by_id_list(
            id_list=[mlb, ml0, ml1], type="into")
        self.cc.dynamic_into_list = [mlb, ml0, ml1]
        self.cc.dynamic_ext_list = []

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        results = self.manipulator.fk(jnt_values=jnt_values,
                                      toggle_jacobian=toggle_jacobian,
                                      update=update)
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

    base = wd.World(cam_pos=[1.5, 1.5, 1.0], lookat_pos=[0, 0, 0.3])
    mgm.gen_frame().attach_to(base)

    robot = PantheraHTSglArm(enable_cc=True)
    joint = robot.rand_conf()
    print("Random Joint:", joint)

    pos, rot = robot.fk(joint)
    print("TCP Pos:", pos)
    print("TCP Rot:\n", rot)
    gm.gen_sphere(pos=pos, radius=0.02, rgb=np.array([1, 0, 0])).attach_to(base)
    joint_ik = robot.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=joint)
    print("IK Solved Joint:", joint_ik)

    robot.goto_given_conf(joint_ik)
    robot.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    base.run()
