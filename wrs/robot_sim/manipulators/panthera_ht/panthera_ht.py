#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/06/10
# @Author : HuHanYing
import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.collision_model as mcm

try:
    from trac_ik.trac_ik import TracIK
    _HAS_TRAC_IK = True
except Exception:
    _HAS_TRAC_IK = False


class PantheraHT(mi.ManipulatorInterface):
    """
    Panthera-HT 6-DoF arm for WRS.

    This file is manually converted from:
    Panthera-HT_description_with_finger.urdf

    Only the 6 revolute arm joints are modeled here.
    The two finger prismatic joints are ignored here.
    Gripper should be attached in higher-level robot wrapper.
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 ik_solver='d',
                 name='PantheraHT',
                 enable_cc=False):

        super().__init__(
            pos=pos,
            rotmat=rotmat,
            home_conf=np.zeros(6),
            name=name,
            enable_cc=enable_cc
        )

        current_file_dir = os.path.dirname(__file__)
        mesh_dir = os.path.join(current_file_dir, "meshes1")

        rgba_base = np.array([0.55, 0.55, 0.55, 1.0])
        rgba_link = np.array([0.35, 0.35, 0.35, 1.0])

        # ---------------- Anchor: base_link ----------------
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(mesh_dir, "base_link.stl")
        )
        self.jlc.anchor.lnk_list[0].loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rgba_base

        # ---------------- Joint 1: base_link -> link1 ----------------
        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.0584])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[0].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[0].motion_range = np.array([-2.4, 2.4])

        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(mesh_dir, "link1.STL")
        )
        self.jlc.jnts[0].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[0].lnk.cmodel.rgba = rgba_link

        # ---------------- Joint 2: link1 -> link2 ----------------
        self.jlc.jnts[1].loc_pos = np.array([0.018199, 0.0, 0.053])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0.0, 1.0, 0.0])
        # Sync with fafu robot.cfg limits.2 = [-12.0, 128.0] deg.
        self.jlc.jnts[1].motion_range = np.array([
            np.deg2rad(-12.0),
            np.deg2rad(128.0),
        ])

        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(mesh_dir, "link2.STL")
        )
        self.jlc.jnts[1].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[1].lnk.cmodel.rgba = rgba_link

        # ---------------- Joint 3: link2 -> link3 ----------------
        self.jlc.jnts[2].loc_pos = np.array([-0.26, 0.0, 0.0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[2].loc_motion_ax = np.array([0.0, -1.0, 0.0])
        # Sync with fafu robot.cfg limits.3 = [-150.0, 176.976] deg.
        self.jlc.jnts[2].motion_range = np.array([
            np.deg2rad(-150.0),
            np.deg2rad(176.976),
        ])

        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(mesh_dir, "link3.STL")
        )
        self.jlc.jnts[2].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[2].lnk.cmodel.rgba = rgba_link

        # ---------------- Joint 4: link3 -> link4 ----------------
        self.jlc.jnts[3].loc_pos = np.array([0.23, 0.0, 0.06])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0.0, -1.0, 0.0])
        # Sync with fafu robot.cfg limits.4 = [-128.0, 95.0] deg
        self.jlc.jnts[3].motion_range = np.array([
            np.deg2rad(-128.0),
            np.deg2rad(95.0),
        ])

        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(mesh_dir, "link4.STL")
        )
        self.jlc.jnts[3].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[3].lnk.cmodel.rgba = rgba_link

        # ---------------- Joint 5: link4 -> link5 ----------------
        self.jlc.jnts[4].loc_pos = np.array([0.07, 0.0, 0.036319])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0.0, 0.0, -1.0])
        self.jlc.jnts[4].motion_range = np.array([
            np.deg2rad(-92.0),
            np.deg2rad(78.0),
        ])

        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            os.path.join(mesh_dir, "link5.STL")
        )
        self.jlc.jnts[4].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[4].lnk.cmodel.rgba = rgba_link

        # ---------------- Joint 6: link5 -> link6 ----------------
        # URDF 中 link6 的 frame 是 x 轴朝前；这里把 jnts[5] 的 loc_rotmat
        # 绕 y 轴 +π/2，使 wrs0 中 jnts[5] / link6 的新 frame 满足：
        #   新 z 轴 = 原 +x 轴（朝"前"）
        #   新 x 轴 = 原 -z 轴
        #   新 y 轴 不变
        # 物理转动轴仍是 link6 长方向（即原 +x），在新 frame 下表达为 +z。
        # IK 端通过 R_y(-π/2) 补偿与 URDF 的 frame 差异（见 self.ik）。
        self.jlc.jnts[5].loc_pos = np.array([0.02345, 0.0, -0.039])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(0.0, np.pi / 2.0, 0.0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        # Sync with fafu robot.cfg limits.6 = [-110.0, 112.0] deg.
        self.jlc.jnts[5].motion_range = np.array([
            np.deg2rad(-110.0),
            np.deg2rad(112.0),
        ])
        # wrs0 link6 frame = URDF link6 frame · R_y(π/2)
        # → IK 输入需要把 wrs0 期望姿态右乘 R_y(-π/2) 还原为 URDF 朝向。
        self._urdf_to_wrs_flange_rotmat = rm.rotmat_from_euler(0.0, np.pi / 2.0, 0.0)

        # Keep link6 in kinematics chain, but do not render link6 mesh.
        # This avoids visual overlap with the attached PantheraGripper.
        # self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
        #     os.path.join(mesh_dir, "link6.STL")
        # )
        self.jlc.jnts[5].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        if self.jlc.jnts[5].lnk.cmodel is not None:
            self.jlc.jnts[5].lnk.cmodel.rgba = rgba_link

        # Finalize JLC
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)

        # TCP (tool center point):
        # 与 Piper 设计一致：本体层默认 TCP 设在法兰原点，
        # 若挂接 end-effector，由上层 wrapper 覆盖为夹爪 acting center。
        self.loc_tcp_pos = np.array([0.0, 0.0, 0.0])
        self.loc_tcp_rotmat = np.eye(3)
        if _HAS_TRAC_IK:
            urdf = os.path.join(current_file_dir, "Panthera-HT_description_with_finger.urdf")
            self._ik_solver = TracIK(
                "base_link",
                "link6",
                urdf,
                timeout=0.01,
                solver_type="Distance"
            )
        else:
            self._ik_solver = None

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """
        Conservative self-collision setup.

        Adjacent/nearby Panthera meshes slightly overlap at the default
        configuration, so this manipulator-level setup does not register
        internal self-collision pairs. External collision masks are still
        enabled for all moving links.
        """
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)

        self.cc.enable_extcd_by_id_list(id_list=[l1, l2, l3, l4], type="from")
        self.cc.enable_innercd_by_id_list(id_list=[lb, l0, l1], type="into")
        self.cc.dynamic_into_list = [lb, l0, l1]
        self.cc.dynamic_ext_list = []

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rotmat: np.ndarray,
           seed_jnt_values=None,
           option: str = "empty",
           toggle_dbg: bool = False):
        """
        Panthera-HT IK.

        优先使用 TracIK（若环境已安装 trac_ik），否则回退到 WRS 的 JLC IK。
        """
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos
        if _HAS_TRAC_IK and self._ik_solver is not None:
            # TracIK使用base_link坐标系，这里把目标位姿从世界系转到anchor系。
            anchor_inv_homomat = np.linalg.inv(
                rm.homomat_from_posrot(self.jlc.anchor.pos, self.jlc.anchor.rotmat)
            )
            tgt_homomat = anchor_inv_homomat.dot(rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            tgt_pos_base = tgt_homomat[:3, 3]
            tgt_rotmat_base = tgt_homomat[:3, :3]
            # wrs0 link6 frame 比 URDF link6 frame 多一个 R_y(π/2)（让 z 朝前）。
            # trac_ik 用的是原始 URDF，因此把 wrs0 期望姿态右乘 R_y(-π/2) 还原为
            # URDF 朝向；位置不变，因为 jnts[5].lnk 的 origin 不受 frame 旋转影响。
            tgt_rotmat_base = tgt_rotmat_base @ self._urdf_to_wrs_flange_rotmat.T
            seed = self.home_conf if seed_jnt_values is None else seed_jnt_values.copy()
            return self._ik_solver.ik(tgt_pos_base, tgt_rotmat_base, seed_jnt_values=seed)
        return self.jlc.ik(
            tgt_pos=tgt_pos,
            tgt_rotmat=tgt_rotmat,
            seed_jnt_values=seed_jnt_values,
            toggle_dbg=toggle_dbg
        )


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[1.2, 1.2, 0.8], lookat_pos=[0, 0, 0.15])
    mgm.gen_frame().attach_to(base)

    arm = PantheraHT(enable_cc=True)

    test_conf = np.array([0.0, 0, 0, 0.0, 0.0, 0.0])
    arm.goto_given_conf(jnt_values=test_conf)

    arm.gen_meshmodel(toggle_cdprim=True,toggle_jnt_frames=True, toggle_tcp_frame=True).attach_to(base)
    print(arm.is_collided())
    base.run()