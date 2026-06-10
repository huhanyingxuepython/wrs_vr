import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc


class PantheraGripper(gpi.GripperInterface):
    """
    Panthera 双指夹爪（可选掌部 + 左指 + 右指）。

    目录约定：
    - 当前文件同级 `meshes/` 目录中放置 3 个网格
      - palm/base mesh
      - left finger mesh
      - right finger mesh
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name="panthera_gripper",
                 jaw_range=np.array([0.0, 0.08]),
                 close_bias=0.0,
                 use_palm_mesh=True,
                 palm_mesh_name=None,
                 left_finger_mesh_name=None,
                 right_finger_mesh_name=None):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        self.jaw_range = np.asarray(jaw_range, dtype=float)
        if self.jaw_range.shape != (2,) or self.jaw_range[0] < 0 or self.jaw_range[1] <= self.jaw_range[0]:
            raise ValueError(f"Invalid jaw_range: {self.jaw_range}")
        self.close_bias = float(close_bias)
        if self.close_bias < 0:
            raise ValueError(f"close_bias must be >= 0, got {self.close_bias}")
        # Use max-open as the initial state, then close inward from there.
        self.jaw_width = float(self.jaw_range[1])

        current_dir = os.path.dirname(__file__)
        mesh_dir = os.path.join(current_dir, "meshes")
        palm_mesh_path = None
        if use_palm_mesh:
            palm_mesh_path = self._resolve_mesh_path(
                mesh_dir=mesh_dir,
                explicit_name=palm_mesh_name,
                role_name="palm",
                candidate_names=["palm.stl", "base.stl", "gripper_base.stl", "hand_base.stl"]
            )
        left_mesh_path = self._resolve_mesh_path(
            mesh_dir=mesh_dir,
            explicit_name=left_finger_mesh_name,
            role_name="left finger",
            candidate_names=["left_finger.stl", "finger_left.stl", "lfinger.stl"]
        )
        right_mesh_path = self._resolve_mesh_path(
            mesh_dir=mesh_dir,
            explicit_name=right_finger_mesh_name,
            role_name="right finger",
            candidate_names=["right_finger.stl", "finger_right.stl", "rfinger.stl"]
        )

        self.jlc = rkjlc.JLChain(
            pos=self.coupling.gl_flange_pose_list[0][0],
            rotmat=self.coupling.gl_flange_pose_list[0][1],
            n_dof=2,
            name=name
        )

        # 掌部：若 link6 已包含底座，可关闭以避免重叠
        if palm_mesh_path is not None:
            self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
                initor=palm_mesh_path,
                name=f"{name}_palm",
                cdmesh_type=self.cdmesh_type
            )
            self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([0.35, 0.35, 0.35, 1.0])

        command_max = self.jaw_range[1] + self.close_bias
        self._command_max = float(command_max)
        side_range = np.array([0.0, command_max / 2.0])
        jaw_range_full = np.array([0.0, command_max])

        # 左指：沿 +Y 张开
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=side_range)
        self.jlc.jnts[0].loc_pos = np.zeros(3)
        self.jlc.jnts[0].loc_rotmat = np.eye(3)
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].motion_range = side_range
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=left_mesh_path,
            name=f"{name}_left_finger",
            cdmesh_type=self.cdmesh_type
        )
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # 右指：沿 -Y 张开
        # 注意：JLChain 是串联结构。为得到“左右对称开合”效果，
        # 第二个关节使用总开口量 jaw_width（而非 side_jaw_width）作为驱动值。
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=jaw_range_full)
        self.jlc.jnts[1].loc_pos = np.zeros(3)
        self.jlc.jnts[1].loc_rotmat = np.eye(3)
        self.jlc.jnts[1].loc_motion_ax = -rm.const.y_ax
        self.jlc.jnts[1].motion_range = jaw_range_full
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=right_mesh_path,
            name=f"{name}_right_finger",
            cdmesh_type=self.cdmesh_type
        )
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        self.jlc.finalize()
        self._command_open_with_increasing = True
        self._calibrate_opening_direction()

        # 与此前 Panthera TCP 习惯保持一致（沿夹爪前向 x 方向）
        self.loc_acting_center_pos = np.array([0, 0.0, 0.13])
        self.loc_acting_center_rotmat = np.eye(3)

        self.cdelements = [self.jlc.jnts[0].lnk, self.jlc.jnts[1].lnk]
        if self.jlc.anchor.lnk_list[0].cmodel is not None:
            self.cdelements.insert(0, self.jlc.anchor.lnk_list[0])
        self.change_jaw_width(self.jaw_width)

    @staticmethod
    def _resolve_mesh_path(mesh_dir, explicit_name, role_name, candidate_names):
        if not os.path.isdir(mesh_dir):
            raise FileNotFoundError(f"Panthera gripper mesh directory not found: {mesh_dir}")

        if explicit_name:
            mesh_path = os.path.join(mesh_dir, explicit_name)
            if os.path.isfile(mesh_path):
                return mesh_path
            raise FileNotFoundError(f"{role_name} mesh not found: {mesh_path}")

        lower_to_real = {}
        for filename in os.listdir(mesh_dir):
            if os.path.isfile(os.path.join(mesh_dir, filename)):
                lower_to_real[filename.lower()] = filename

        for name in candidate_names:
            if name.lower() in lower_to_real:
                return os.path.join(mesh_dir, lower_to_real[name.lower()])

        stl_files = sorted([f for f in lower_to_real.values() if f.lower().endswith(".stl")])
        raise FileNotFoundError(
            f"Cannot resolve {role_name} mesh under {mesh_dir}. "
            f"Expected one of {candidate_names}, existing STL files: {stl_files}"
        )

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def _set_chain_command(self, command):
        command = float(np.clip(command, 0.0, self._command_max))
        side_command = command / 2.0
        self.jlc.goto_given_conf(jnt_values=np.array([side_command, command]))

    def _measure_gap(self):
        left_pos = self.jlc.jnts[0].lnk.gl_pos
        right_pos = self.jlc.jnts[1].lnk.gl_pos
        return float(np.linalg.norm(left_pos - right_pos))

    def _calibrate_opening_direction(self):
        # Probe two command levels and infer whether larger command opens or closes.
        sample_a = 0.0
        sample_b = max(min(self._command_max * 0.5, self._command_max), 1e-6)
        self._set_chain_command(sample_a)
        gap_a = self._measure_gap()
        self._set_chain_command(sample_b)
        gap_b = self._measure_gap()
        self._command_open_with_increasing = (gap_b >= gap_a)

    def _jaw_to_command(self, jaw_width):
        jaw_min = float(self.jaw_range[0])
        jaw_max = float(self.jaw_range[1])
        if self._command_open_with_increasing:
            # Larger command => larger gap
            command = jaw_width + self.close_bias
        else:
            # Larger command => smaller gap
            command = jaw_max - jaw_width + jaw_min + self.close_bias
        return float(np.clip(command, 0.0, self._command_max))

    def _command_to_jaw(self, command):
        jaw_min = float(self.jaw_range[0])
        jaw_max = float(self.jaw_range[1])
        if self._command_open_with_increasing:
            jaw_width = command - self.close_bias
        else:
            jaw_width = jaw_max + jaw_min + self.close_bias - command
        return float(np.clip(jaw_width, jaw_min, jaw_max))

    def get_jaw_width(self):
        command = float(self.jlc.jnts[1].motion_value)
        return self._command_to_jaw(command)

    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        jaw_width = float(jaw_width)
        if self.jaw_range[0] <= jaw_width <= self.jaw_range[1]:
            command = self._jaw_to_command(jaw_width)
            self._set_chain_command(command)
            self.jaw_width = jaw_width
        else:
            raise ValueError("The jaw_width parameter is out of range!")

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False):
        m_col = mmc.ModelCollection(name=self.name + "_stickmodel")
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_root_frame=False,
                                    toggle_flange_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.jlc.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        self._gen_oiee_meshmodel(m_col,
                                 rgb=rgb,
                                 alpha=alpha,
                                 toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh,
                                 toggle_frame=toggle_jnt_frames)
        return m_col


if __name__ == "__main__":
    from wrs import wd

    base = wd.World(cam_pos=[0.5, 0.5, 0.5], lookat_pos=[0.0, 0.0, 0.0])

    gripper = PantheraGripper(cdmesh_type=mcm.const.CDMeshType.OBB)
    gripper.change_jaw_width(0.04)
    print(f"Current jaw width: {gripper.get_jaw_width():.3f} m")
    gripper.gen_stickmodel().attach_to(base)
    gripper.gen_meshmodel(toggle_cdprim=False, toggle_tcp_frame=True, alpha=1.0).attach_to(base)

    base.run()
