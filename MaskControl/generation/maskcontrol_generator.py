import torch
import numpy as np
import os
from os.path import join as pjoin
from utils.paramUtil import t2m_kinematic_chain
from utils.plot_script import plot_3d_motion
from visualization.joints2bvh import Joint2BVHConvertor
from generation.load_model import get_models
from types import SimpleNamespace
import json as json_module

class MaskControlGenerator:
    def __init__(self, opt_path="./generation/args.json", device="cuda:0"):
        import json
        with open(opt_path, "r") as f:
            opt = json.load(f)
        self.opt = SimpleNamespace(**opt)
        self.device = torch.device(device)

        self.ct2m_transformer, self.vq_model, self.res_model, self.moment = get_models(self.opt)
        self.ct2m_transformer.eval()
        self.vq_model.eval()

        self.converter = Joint2BVHConvertor()
    
    def export_joints_json(self, motion_joints, out_path):
        """
        Export raw joint positions as JSON for custom retargeting.
        
        Args:
            motion_joints: numpy array of shape (num_frames, 22, 3) - joint positions
            out_path: path to save JSON file
        
        Returns:
            out_path: path to saved JSON file
        """
        joint_names = [
            "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
            "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
            "left_foot", "right_foot", "neck", "left_collar", "right_collar",
            "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist"
        ]
        
        motion_data = {
            "skeleton_type": "HumanML3D",
            "num_frames": len(motion_joints),
            "num_joints": 22,
            "fps": 20,
            "joint_names": joint_names,
            "frames": motion_joints.tolist()
        }
        
        with open(out_path, 'w') as f:
            json_module.dump(motion_data, f, indent=2)
        
        return out_path

    def generate_with_avoidance(self, text_prompt, motion_length=196, control_frame=195, control_joint=0,
                                 control_pos=(0.5, 1.0229, 6.5),
                                 avoid_points=None,
                                 out_dir="generation/maskcontrol/animations/0",
                                 out_name="avoidance.bvh",
                                 iter_each=0, iter_last=100,
                                 export_json=False):

        m_length = torch.tensor([motion_length]).to(self.device)

        global_joint = torch.zeros((1, motion_length, 22, 3), device=self.device)
        global_joint[0, control_frame, control_joint] = torch.tensor(control_pos, device=self.device)
        global_joint_mask = (global_joint.sum(-1) != 0)

        if avoid_points is None:
            avoid_points = torch.tensor([
                [0.5, 0.9508, 1.5, 1],
                [-1.5, 0.9898, 4, 2],
                [1.5, 0.9508, 5.5, 1],
                [3, 0.9898, 3, 2],
            ], device=self.device)

        def abitary_func(pred):
            cond = avoid_points
            if len(cond.shape) == 2:
                from einops import repeat
                cond = repeat(cond, 'o four -> b f o four', b=pred.shape[0], f=pred.shape[1])
                pred = repeat(pred, 'b f j d -> b f j o d', o=cond.shape[2])
            dist = torch.norm(pred[:, :, 0] - cond[..., :3], dim=-1)
            dist = torch.clamp(cond[..., 3] - dist, min=0.0)
            loss_colli = dist[cond[..., 3]>0].mean()
            return loss_colli

        # Removed torch.no_grad() context - gradient computation is needed for optimization
        pred_motions_denorm, pred_motions = self.ct2m_transformer.generate_with_control(
            [text_prompt], m_length,
            time_steps=10, cond_scale=4,
            temperature=1, topkr=.9,
            force_mask=self.opt.force_mask,
            vq_model=self.vq_model,
            global_joint=global_joint,
            global_joint_mask=global_joint_mask,
            _mean=torch.tensor(self.moment[0]).to(self.device),
            _std=torch.tensor(self.moment[1]).to(self.device),
            res_cond_scale=5,
            res_model=None,
            control_opt={
                'each_lr': 6e-2,
                'each_iter': iter_each,
                'lr': 6e-2,
                'iter': iter_last,
            },
            abitary_func=abitary_func
        )

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)

        # pred_motions_denorm is already in joint format, no need for recover_from_ric
        motion_joints = pred_motions_denorm[0, :m_length[0]].detach().cpu().numpy()
        self.converter.convert(motion_joints, filename=out_path, iterations=100, foot_ik=False)
        
        # Optionally export JSON for frontend retargeting
        if export_json:
            json_name = out_name.replace('.bvh', '.json')
            json_path = os.path.join(out_dir, json_name)
            self.export_joints_json(motion_joints, json_path)
            print(f'✅ Exported JSON: {json_path}')
        
        return out_path

    def generate_with_control_joint(self, text_prompt, motion_length=196,
                                     traj1=None, traj2=None,
                                     out_dir="generation/maskcontrol/animations/0",
                                     out_name="control_joint.bvh",
                                     iter_each=0, iter_last=100,
                                     export_json=False,
                                     export_html=False):
        """
        Generate motion with controlled joint trajectories.
        
        Args:
            text_prompt: Text description of the motion
            motion_length: Length of motion in frames
            traj1: Trajectory for joint 0 (root), shape (motion_length, 3). If None, uses default circle.
            traj2: Trajectory for joint 20 (right wrist), shape (motion_length, 3). If None, uses default circle.
            out_dir: Output directory
            out_name: Output filename
            iter_each: Number of optimization iterations at each unmask step
            iter_last: Number of optimization iterations at the last unmask step
            export_json: If True, also export raw joint positions as JSON (default: False)
            export_html: If True, also export HTML skeleton visualization (default: False)
        """
        from utils.trajectory_plot import draw_circle_with_waves, draw_circle_with_waves2
        
        m_length = torch.tensor([motion_length]).to(self.device)
        
        # Use provided trajectories or defaults
        if traj1 is None:
            traj1 = draw_circle_with_waves()
            if self.device != traj1.device:
                traj1 = traj1.to(self.device)
        
        # Note: traj2 is not currently being used (commented out below)
        # Only set default traj2 if we actually plan to use it
        using_traj2 = False  # Set to True if you uncomment the traj2 line below
        if using_traj2 and traj2 is None:
            traj2 = draw_circle_with_waves2()
            if self.device != traj2.device:
                traj2 = traj2.to(self.device)
        
        # Set up global joint control
        global_joint = torch.zeros((1, motion_length, 22, 3), device=self.device)
        global_joint[0, :, 0] = traj1  # Control root joint (pelvis)
        #global_joint[0, :, 20] = traj2  # Control right wrist joint
        global_joint_mask = (global_joint.abs().sum(-1) > 0)
        
        # Temporarily enable training mode for gradient-based optimization
        self.ct2m_transformer.eval()
        
        print(' Optimizing with control_joint...')
        pred_motions_denorm, pred_motions = self.ct2m_transformer.generate_with_control(
            [text_prompt], m_length,
            time_steps=10, cond_scale=4,
            temperature=1, topkr=.9,
            force_mask=self.opt.force_mask,
            vq_model=self.vq_model,
            global_joint=global_joint,
            global_joint_mask=global_joint_mask,
            _mean=torch.tensor(self.moment[0]).to(self.device),
            _std=torch.tensor(self.moment[1]).to(self.device),
            res_cond_scale=5,
            res_model=self.res_model,
            control_opt={
                'each_lr': 6e-2,
                'each_iter': iter_each,
                'lr': 6e-2,
                'iter': iter_last,
            }
        )
        print('Done.')
        
        # Restore eval mode
        self.ct2m_transformer.eval()
        
        # Save BVH output
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
        
        # pred_motions_denorm is already in joint format, no need for recover_from_ric
        motion_joints = pred_motions_denorm[0, :m_length[0]].detach().cpu().numpy()
        print(f'💾 Saving BVH to: {out_path}')
        self.converter.convert(motion_joints, filename=out_path, iterations=10, foot_ik=False)
        print(f'✅ BVH saved successfully')
        
        # Optionally export JSON for frontend retargeting
        print(f'🔍 export_json parameter: {export_json}')
        if export_json:
            json_name = out_name.replace('.bvh', '.json')
            json_path = os.path.join(out_dir, json_name)
            print(f'💾 Exporting JSON to: {json_path}')
            self.export_joints_json(motion_joints, json_path)
            print(f'✅ JSON exported successfully: {json_path}')
        else:
            print(f'⚠️  JSON export skipped (export_json=False)')
        
        # Also save numpy arrays for visualization
        np.save(os.path.join(out_dir, "generation.npy"), pred_motions[0, :m_length[0]].detach().cpu().numpy())
        np.save(os.path.join(out_dir, "trj_cond.npy"), global_joint[0, :m_length[0]].detach().cpu().numpy())
        
        # Optionally export HTML visualization
        if export_html:
            from exit.utils import visualize_2motions
            
            html_name = out_name.replace('.bvh', '.html')
            html_path = os.path.join(out_dir, html_name)
            print(f'💾 Exporting HTML visualization to: {html_path}')
            
            # Only visualize traj2 if it was actually used in generation
            visualize_2motions(
                pred_motions[0].detach().cpu().numpy(),
                self.moment[1],  # std
                self.moment[0],  # mean
                't2m',
                m_length[0],
                root_path=traj1.detach().cpu().numpy(),
                root_path2=None,  # traj2 is not being used (line 173 is commented out)
                save_path=html_path,
                show=False  # Don't auto-open in browser
            )
            print(f'✅ HTML visualization exported successfully: {html_path}')
        
        return out_path
