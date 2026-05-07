"""
Generate a single animal motion from a text prompt and save:
  1. joints.npy - Joint positions trimmed to actual length (seqlen, njoints, 3)
  2. joint_vecs.npy - Motion vectors trimmed to actual length (seqlen, nfeats)
  3. motion.mp4 - 3D skeleton video visualization
  4. feature.npy - Image features (if image conditioning is used)
  5. image.png - Input image (if --image_path is provided)

Usage:
    # Text-only generation
    python generate_animal_single.py --model_path <path> --text "a dog walking"
    
    # Image-conditioned generation (from image file)
    python generate_animal_single.py --model_path <path> --text "a dog walking" \\
        --image_path /path/to/image.png
    
    # Image-conditioned generation (from pre-extracted features)
    python generate_animal_single.py --model_path <path> --text "a dog walking" \\
        --image_feature_path /path/to/features.npy

Arguments:
    --model_path: Path to pretrained model checkpoint (.pt)
    --text: Text prompt describing the desired motion
    --save_dir: Root directory for results (default: /viscam/projects/animal_motion/briannlz/gen_results/)
    --image_path: (Optional) Path to image file - features will be extracted automatically
    --image_feature_path: (Optional) Path to pre-extracted .npy feature file
    --motion_length: Duration in seconds (default: 6.0)
    --guidance_param: Classifier-free guidance scale (default: 2.5)
    --num_repetitions: Number of independent samples (default: 1)
    --seed: Random seed (default: 10)
    --device: CUDA device index (default: 0)

The script creates <save_dir>/<sanitised_text>/ and writes all output files there.
"""

import argparse
import json
import os
import re
import shutil

import numpy as np
import torch
from transformers import pipeline

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.tensors import collate
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion


# ---------------------------------------------------------------------------
# Constants (must match training / generate_animal.py)
# ---------------------------------------------------------------------------
MAX_FRAMES = 196
FPS = 20
NJOINTS = 34


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate a single animal motion from a text prompt.'
    )
    parser.add_argument('--model_path',          type=str,
                        help='Path to the pretrained model checkpoint (.pt).')
    parser.add_argument('--text',                type=str,
                        help='Text prompt describing the desired motion.')
    parser.add_argument('--save_dir',            type=str, default="/viscam/projects/animal_motion/briannlz/gen_results/",
                        help='Root directory where results will be saved.')
    parser.add_argument('--image_path', default=None, type=str,
                        help='(Optional) Path to an image file for image-conditioned generation. Features will be extracted automatically.')
    parser.add_argument('--image_feature_path', default=None, type=str,
                        help='(Optional) Path to a .npy image-feature file for image-conditioned generation.')
    parser.add_argument('--motion_length',     default=3.0, type=float,
                        help='Duration of the generated motion in seconds (max 9.8). Default: 5.0.')
    parser.add_argument('--guidance_param',    default=2.5, type=float,
                        help='Classifier-free guidance scale. Default: 2.5.')
    parser.add_argument('--num_repetitions',   default=1,   type=int,
                        help='Number of independent samples to generate. Default: 1.')
    parser.add_argument('--seed',              default=111,  type=int,
                        help='Random seed.')
    parser.add_argument('--device',            default=0,   type=int,
                        help='CUDA device index. Default: 0.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitise(text: str, max_len: int = 100) -> str:
    """Turn arbitrary text into a safe directory / file name component."""
    s = re.sub(r'[^A-Za-z0-9_-]+', '_', text)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:max_len] if s else 'motion'


def extract_image_features(image_path: str, device: int = 0) -> np.ndarray:
    """Extract DINOv3 features from an image.
    
    Args:
        image_path: Path to the image file
        device: CUDA device index (0 for GPU, -1 for CPU)
    
    Returns:
        Feature vector of shape (1, 768)
    """
    print(f"Extracting features from image: {image_path}")
    feature_extractor = pipeline(
        model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        task="image-feature-extraction",
        device=device,
    )
    features = feature_extractor([image_path])
    # Extract the [CLS] token feature (first token)
    # The pipeline may return different shapes, so we extract and reshape
    feature_array = np.array(features)
    # Get the CLS token (typically at position 0 in the sequence dimension)
    # and squeeze to ensure shape is (1, 768)
    mean_feature = feature_array[..., 0, :].reshape(1, 768)
    print(f"Extracted feature shape: {mean_feature.shape}")
    return mean_feature


def load_model_args(model_path: str) -> dict:
    """Load the training args that were saved alongside the checkpoint."""
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f"args.json not found next to checkpoint: {args_path}"
    with open(args_path, 'r') as f:
        return json.load(f)


def build_args_namespace(user_args, model_args: dict):
    """
    Merge user-supplied CLI args with the model's saved args.
    User args take priority for fields they expose; everything else
    comes from the saved model args.
    """
    ns = argparse.Namespace(**model_args)

    # Override / set fields that the user controls
    ns.model_path       = user_args.model_path
    ns.seed             = user_args.seed
    ns.device           = user_args.device
    ns.motion_length    = user_args.motion_length
    ns.guidance_param   = user_args.guidance_param
    ns.num_repetitions  = user_args.num_repetitions
    ns.num_samples      = 1                # single generation
    ns.batch_size       = 1
    ns.text_prompt      = user_args.text
    ns.unconstrained    = False
    ns.autoregressive   = False
    ns.context_len      = getattr(ns, 'context_len', 0)
    ns.dynamic_text_path = ''

    # Image conditioning
    # Keep image_condition / cond_mode from the saved model args so that
    # create_model_and_diffusion builds the correct architecture (e.g. embed_image
    # layers).  We will override model.cond_mode AFTER loading the checkpoint,
    # exactly like generate_animal.py does.
    ns.image_feature_path = user_args.image_feature_path
    ns.image_path         = None
    # ns.image_condition and ns.cond_mode intentionally left as loaded from args.json

    # Classifier-free guidance: if cond_mask_prob == 0 then guidance == 1
    if getattr(ns, 'cond_mask_prob', 1.0) == 0:
        ns.guidance_param = 1.0

    # Ensure pred_len rule from apply_rules
    if getattr(ns, 'pred_len', 0) == 0:
        ns.pred_len = ns.context_len

    # Fields used elsewhere that may not exist in older checkpoints
    ns.use_ema = getattr(ns, 'use_ema', False)

    return ns


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_single(args, model, diffusion, mean, std, image_feature=None):
    """Run the diffusion sampler and return joint positions.

    Args:
        image_feature: Optional pre-loaded image feature array (1, 768)

    Returns
    -------
    joints : np.ndarray  shape (seqlen, njoints, 3)
    motion_vec : np.ndarray  raw HML vector (before inv-transform), shape (seqlen, nfeats)
    length : int  actual motion length in frames
    """
    n_frames = min(MAX_FRAMES, int(args.motion_length * FPS))

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}]
    collate_args = [dict(collate_args[0], text=args.text_prompt)]
    _, model_kwargs = collate(collate_args)

    model_kwargs['y'] = {
        k: v.to(dist_util.dev()) if torch.is_tensor(v) else v
        for k, v in model_kwargs['y'].items()
    }

    # CFG scale
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(1, device=dist_util.dev()) * args.guidance_param

    # Encode text once
    if 'text' in model_kwargs['y']:
        model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])

    # Image feature
    if args.image_condition:
        if image_feature is None:
            # Load from file if not provided directly
            image_feature = np.load(args.image_feature_path)
        # Convert to tensor and ensure 3D shape: [batch=1, seq=1, features=768]
        # Using .repeat(1, 1, 1) to match generate_animal.py behavior
        model_kwargs['y']['feature'] = torch.as_tensor(
            image_feature, device=dist_util.dev()
        ).repeat(1, 1, 1)  # (1, 768) -> (1, 1, 768)

    motion_shape = (1, model.njoints, model.nfeats, n_frames)

    sample_fn = diffusion.p_sample_loop

    all_joints = []
    all_vecs   = []

    for rep_i in range(args.num_repetitions):
        print(f'  Sampling repetition {rep_i + 1}/{args.num_repetitions} ...')
        sample = sample_fn(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Inverse-transform to HML vectors
        sample_vec = (sample.cpu().permute(0, 2, 3, 1) * std + mean).float()
        all_vecs.append(sample_vec.squeeze().cpu().numpy())   # (seqlen, nfeats)

        # Recover XYZ joint positions
        joints = recover_from_ric(sample_vec, NJOINTS)        # (1, seqlen, njoints, 3)
        joints = joints.view(-1, *joints.shape[2:]).permute(0, 2, 3, 1)  # (1, njoints, 3, seqlen)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = (
            None if rot2xyz_pose_rep == 'xyz'
            else model_kwargs['y']['mask'].reshape(1, n_frames).bool()
        )
        joints = model.rot2xyz(
            x=joints, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep,
            glob=True, translation=True, jointstype='smpl',
            vertstrans=True, betas=None, beta=0, glob_rot=None,
            get_rotations_back=False,
        )
        # joints: (1, njoints, 3, seqlen) -> (seqlen, njoints, 3)
        joints_np = joints.squeeze(0).permute(2, 0, 1).cpu().numpy()
        all_joints.append(joints_np)

    length = int(model_kwargs['y']['lengths'].cpu().numpy()[0])
    # Return first repetition's joints (shape: seqlen, njoints, 3) and its vector
    return all_joints[0], all_vecs[0], length


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def compute_orientation_from_legs(joints):
    """Compute orientation vector from legs.
    
    Uses the average direction of left and right leg pairs:
    - Left side: front left foot to rear left foot
    - Right side: front right foot to rear right foot
    
    Args:
        joints: Joint positions with shape (seqlen, njoints, 3)
    
    Returns:
        Orientation vectors with shape (seqlen, 3) - normalized direction vectors
    """
    # Joint indices from kinematic chain:
    # Rear left leg: [0, 16, 17, 18, 19] -> endpoint is 19
    # Rear right leg: [0, 20, 21, 22, 23] -> endpoint is 23
    # Front left leg: [5, 6, 7, 8, 9] -> endpoint is 9
    # Front right leg: [5, 10, 11, 12, 13] -> endpoint is 13
    
    rear_left_foot = 19
    rear_right_foot = 23
    front_left_foot = 9
    front_right_foot = 13
    
    orientations = []
    
    for frame_idx in range(joints.shape[0]):
        frame_joints = joints[frame_idx]  # (njoints, 3)
        
        # Get foot positions
        fl_pos = frame_joints[front_left_foot]
        rl_pos = frame_joints[rear_left_foot]
        fr_pos = frame_joints[front_right_foot]
        rr_pos = frame_joints[rear_right_foot]
        
        # Compute forward vectors for each side
        left_forward = fl_pos - rl_pos  # Front left - rear left
        right_forward = fr_pos - rr_pos  # Front right - rear right
        
        # Average the two sides to get overall forward direction
        forward_vec = (left_forward + right_forward) / 2.0
        
        # Normalize
        norm = np.linalg.norm(forward_vec)
        if norm > 1e-6:
            forward_vec = forward_vec / norm
        else:
            forward_vec = np.array([0, 0, 1])  # Default to +Z if degenerate
        
        orientations.append(forward_vec)
    
    return np.array(orientations)


def apply_8shape_trajectory_regularization(joints, motion_vec):
    """Smooth root trajectory, then align animal orientation to trajectory direction.

    Pipeline:
    1) Smooth root XZ trajectory with moving average.
    2) Translate all joints so root follows the smoothed trajectory.
    3) Compute smoothed trajectory heading and align leg-based body heading to it
       with per-frame yaw rotation around the root.

    Args:
        joints: Joint positions with shape (seqlen, njoints, 3)
        motion_vec: Motion vectors with shape (seqlen, nfeats)

    Returns:
        Adjusted (joints, motion_vec) tuples
    """
    def moving_average_1d(values, window):
        if values.shape[0] <= 1 or window <= 1:
            return values.copy()
        window = min(window, values.shape[0])
        if window % 2 == 0:
            window -= 1
        if window <= 1:
            return values.copy()
        pad = window // 2
        kernel = np.ones(window, dtype=np.float32) / float(window)
        padded = np.pad(values, (pad, pad), mode='edge')
        return np.convolve(padded, kernel, mode='valid')

    def wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    adjusted_joints = joints.copy()
    adjusted_motion_vec = motion_vec.copy()
    n_frames = adjusted_joints.shape[0]

    if n_frames <= 1:
        return adjusted_joints, adjusted_motion_vec

    # 1) Smooth root trajectory in XZ.
    root_xz = adjusted_joints[:, 0, [0, 2]]
    smooth_window = 9
    smooth_root_x = moving_average_1d(root_xz[:, 0], smooth_window)
    smooth_root_z = moving_average_1d(root_xz[:, 1], smooth_window)
    smooth_root_xz = np.stack([smooth_root_x, smooth_root_z], axis=-1)

    # 2) Move the full body with the smoothed root trajectory.
    delta_root_xz = smooth_root_xz - root_xz
    adjusted_joints[:, :, 0] += delta_root_xz[:, 0:1]
    adjusted_joints[:, :, 2] += delta_root_xz[:, 1:2]

    # 3) Compute trajectory heading from smoothed root trajectory.
    traj_vel = np.zeros_like(smooth_root_xz)
    traj_vel[1:-1] = 0.5 * (smooth_root_xz[2:] - smooth_root_xz[:-2])
    traj_vel[0] = smooth_root_xz[1] - smooth_root_xz[0]
    traj_vel[-1] = smooth_root_xz[-1] - smooth_root_xz[-2]

    target_dir_xz = np.zeros_like(traj_vel)
    last_dir = np.array([0.0, 1.0], dtype=np.float32)
    # Treat very small speeds as standing to avoid noisy heading flips.
    speed_eps = 1e-3
    for i in range(n_frames):
        speed = np.linalg.norm(traj_vel[i])
        if speed > speed_eps:
            target_dir_xz[i] = traj_vel[i] / speed
            last_dir = target_dir_xz[i]
        else:
            target_dir_xz[i] = last_dir

    # Current body heading from legs, projected to XZ.
    body_orient = compute_orientation_from_legs(adjusted_joints)
    body_dir_xz = body_orient[:, [0, 2]].copy()
    last_body_dir = np.array([0.0, 1.0], dtype=np.float32)
    for i in range(n_frames):
        norm = np.linalg.norm(body_dir_xz[i])
        if norm > 1e-6:
            body_dir_xz[i] /= norm
            last_body_dir = body_dir_xz[i]
        else:
            body_dir_xz[i] = last_body_dir
    body_yaw = np.arctan2(body_dir_xz[:, 0], body_dir_xz[:, 1])
    body_yaw = np.unwrap(body_yaw)
    body_yaw = moving_average_1d(body_yaw, window=5)

    # Build target yaw with continuity and 180-degree disambiguation.
    # For each frame we choose either yaw or yaw+pi, whichever is closer to
    # the previous target yaw to avoid sudden branch switches.
    raw_target_yaw = np.arctan2(target_dir_xz[:, 0], target_dir_xz[:, 1])
    target_yaw = np.zeros_like(raw_target_yaw)
    for i in range(n_frames):
        cand_a = raw_target_yaw[i]
        cand_b = cand_a + np.pi
        if i == 0:
            # Initialize using the body-facing direction for smallest first-frame jump.
            err_a = abs(wrap_to_pi(cand_a - body_yaw[i]))
            err_b = abs(wrap_to_pi(cand_b - body_yaw[i]))
            chosen = cand_a if err_a <= err_b else cand_b
        else:
            err_a = abs(wrap_to_pi(cand_a - target_yaw[i - 1]))
            err_b = abs(wrap_to_pi(cand_b - target_yaw[i - 1]))
            chosen = cand_a if err_a <= err_b else cand_b
            # Keep temporal continuity with the previous chosen target.
            chosen = target_yaw[i - 1] + wrap_to_pi(chosen - target_yaw[i - 1])
        target_yaw[i] = chosen
    target_yaw = moving_average_1d(target_yaw, window=9)

    # Smooth relative yaw correction and limit frame-to-frame yaw change
    # to prevent sudden visible turns.
    delta_yaw = wrap_to_pi(target_yaw - body_yaw)
    delta_yaw = moving_average_1d(delta_yaw, window=7)
    max_total_turn = np.deg2rad(75.0)
    delta_yaw = np.clip(delta_yaw, -max_total_turn, max_total_turn)

    max_step_turn = np.deg2rad(8.0)
    stable_delta_yaw = delta_yaw.copy()
    for i in range(1, n_frames):
        step = wrap_to_pi(stable_delta_yaw[i] - stable_delta_yaw[i - 1])
        step = np.clip(step, -max_step_turn, max_step_turn)
        stable_delta_yaw[i] = stable_delta_yaw[i - 1] + step
    delta_yaw = stable_delta_yaw

    # Rotate each frame around the root so orientation aligns with trajectory.
    for i in range(n_frames):
        theta = float(delta_yaw[i])
        c = np.cos(theta)
        s = np.sin(theta)
        root = adjusted_joints[i, 0, :].copy()
        rel = adjusted_joints[i] - root[None, :]
        x = rel[:, 0].copy()
        z = rel[:, 2].copy()
        rel[:, 0] = c * x + s * z
        rel[:, 2] = -s * x + c * z
        adjusted_joints[i] = rel + root[None, :]

    return adjusted_joints, adjusted_motion_vec


def remove_foot_sliding(joints, fps=20, height_threshold=0.05, velocity_threshold=0.03):
    """Remove foot sliding by fixing feet in contact with the ground.
    
    Detects foot contacts based on height and velocity, then locks foot XZ positions
    during contact phases to prevent sliding.
    
    Args:
        joints: Joint positions with shape (seqlen, njoints, 3)
        fps: Frame rate (default: 20)
        height_threshold: Max height to consider foot in contact (default: 0.05)
        velocity_threshold: Max velocity to consider foot in contact (default: 0.03)
    
    Returns:
        Adjusted joints with shape (seqlen, njoints, 3)
    """
    # Foot joint indices from kinematic chain
    foot_joints = [9, 13, 19, 23]  # front_left, front_right, rear_left, rear_right
    
    adjusted_joints = joints.copy()
    n_frames = adjusted_joints.shape[0]
    
    if n_frames <= 2:
        return adjusted_joints
    
    for foot_idx in foot_joints:
        # Compute foot height and velocity
        foot_pos = adjusted_joints[:, foot_idx, :]  # (n_frames, 3)
        foot_height = foot_pos[:, 1]  # Y coordinate
        
        # Compute velocity in XZ plane
        foot_vel_xz = np.zeros((n_frames, 2))
        foot_vel_xz[1:] = foot_pos[1:, [0, 2]] - foot_pos[:-1, [0, 2]]
        foot_speed = np.linalg.norm(foot_vel_xz, axis=1)
        
        # Detect contact: low height AND low velocity
        is_contact = (foot_height < height_threshold) & (foot_speed < velocity_threshold)
        
        # Find contact segments (consecutive contact frames)
        contact_starts = []
        contact_ends = []
        in_contact = False
        
        for i in range(n_frames):
            if is_contact[i] and not in_contact:
                contact_starts.append(i)
                in_contact = True
            elif not is_contact[i] and in_contact:
                contact_ends.append(i - 1)
                in_contact = False
        
        if in_contact:
            contact_ends.append(n_frames - 1)
        
        # For each contact segment, fix the foot XZ position
        for start, end in zip(contact_starts, contact_ends):
            if end - start >= 2:  # Only fix contacts lasting at least 3 frames
                # Use the position at contact start as the fixed position
                fixed_xz = adjusted_joints[start, foot_idx, [0, 2]].copy()
                
                # Lock XZ position for all frames in this contact segment
                for i in range(start, end + 1):
                    adjusted_joints[i, foot_idx, [0, 2]] = fixed_xz
    
    return adjusted_joints


def adjust_root_to_floor(joints):
    """Adjust root translation so that the lowest joint is always on the floor.
    
    Args:
        joints: Joint positions with shape (seqlen, njoints, 3) where last dim is [X, Y, Z]
    
    Returns:
        Adjusted joints with shape (seqlen, njoints, 3)
    """
    # Make a copy to avoid modifying the original
    adjusted_joints = joints.copy()
    
    # Iterate through each frame
    for frame_idx in range(adjusted_joints.shape[0]):
        frame_joints = adjusted_joints[frame_idx]  # (njoints, 3)
        
        # Find the minimum Y value across all joints in this frame
        min_y = frame_joints[:, 1].min()  # Y is at index 1
        
        # Adjust all joints in this frame by raising them so min_y = 0
        if min_y < 0:
            adjusted_joints[frame_idx, :, 1] -= min_y
    
    return adjusted_joints


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(joints, motion_vec, length, text, out_dir, model_name, args, image_path=None, image_feature=None):
    """Save joint positions as .npy and render a video.
    
    Creates:
        joints.npy - Joint positions trimmed to actual length (seqlen, njoints, 3)
        joint_vecs.npy - Motion vectors trimmed to actual length (seqlen, nfeats)
        motion.mp4 - 3D skeleton visualization
        feature.npy - Image features (if image was provided)
        image.png - Input image (if image path was provided)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save joints npy
    npy_path = os.path.join(out_dir, 'joints.npy')
    np.save(npy_path, joints[:length])
    print(f'Saved joints  -> {npy_path}  shape={joints[:length].shape}')

    # Save motion vectors npy
    vecs_path = os.path.join(out_dir, 'joint_vecs.npy')
    np.save(vecs_path, motion_vec[:length])
    print(f'Saved vectors -> {vecs_path}  shape={motion_vec[:length].shape}')

    # Save image feature if provided
    if image_feature is not None:
        feature_path = os.path.join(out_dir, 'feature.npy')
        np.save(feature_path, image_feature)
        print(f'Saved feature -> {feature_path}  shape={image_feature.shape}')
    
    # Save image if path was provided
    if image_path is not None:
        image_save_path = os.path.join(out_dir, 'image.png')
        shutil.copy(image_path, image_save_path)
        print(f'Saved image   -> {image_save_path}')

    # Render video
    skeleton   = paramUtil.t2m_animal_kinematic_chain
    video_path = os.path.join(out_dir, 'motion.mp4')
    motion_vis = joints.copy()
    if motion_vis.shape[0] > length:
        motion_vis[length:] = motion_vis[length - 1]

    print(f'Rendering video -> {video_path}')
    clip = plot_3d_motion(
        video_path, skeleton, motion_vis,
        dataset='animal', title=text, fps=FPS, gt_frames=[],
    )
    if clip is not None:
        try:
            clip.duration = length / FPS
            clip.write_videofile(video_path, fps=FPS, threads=4, logger=None)
            clip.close()
            print(f'Saved video   -> {video_path}')
        except Exception as e:
            print(f'Warning: Failed to save video: {e}')
    else:
        print('Warning: plot_3d_motion returned None, no video saved')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    user_args = parse_args()

    fixseed(user_args.seed)
    dist_util.setup_dist(user_args.device)

    # Build output directory: <save_dir>/<sanitised_text>/
    out_dir = os.path.join(user_args.save_dir, sanitise(user_args.text))
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')

    # Handle image input - extract features if image_path is provided
    image_feature = None
    image_path_for_save = None
    if user_args.image_path is not None and user_args.image_feature_path is not None:
        raise ValueError("Cannot specify both --image_path and --image_feature_path. Choose one.")
    
    if user_args.image_path is not None:
        # Extract features from image
        image_feature = extract_image_features(user_args.image_path, device=user_args.device)
        image_path_for_save = user_args.image_path
    elif user_args.image_feature_path is not None:
        # Load pre-computed features
        image_feature = np.load(user_args.image_feature_path)
        print(f'Loaded image feature from {user_args.image_feature_path}, shape: {image_feature.shape}')

    # Load mean / std for inverse-transform
    mean = np.load('dataset/AnimalMotion/Mean.npy')
    std  = np.load('dataset/AnimalMotion/Std.npy')

    # Load model config from saved args.json
    print(f'Loading model config from checkpoint directory ...')
    model_args_dict = load_model_args(user_args.model_path)
    args = build_args_namespace(user_args, model_args_dict)

    # Create model + diffusion
    print('Creating model and diffusion ...')
    model, diffusion = create_model_and_diffusion(args, data=None)

    print(f'Loading checkpoint from [{user_args.model_path}] ...')
    load_saved_model(model, user_args.model_path, use_avg=args.use_ema)

    # Mirror generate_animal.py: override image_condition / cond_mode based on
    # what the user actually passed, not what the model was trained with.
    args.image_condition = user_args.image_path is not None or user_args.image_feature_path is not None
    model.cond_mode = 'text' if not args.image_condition else model.cond_mode

    # Wrap with CFG sampler AFTER loading weights (same order as generate_animal.py)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)

    model.to(dist_util.dev())
    model.eval()

    # Generate
    print(f'\nGenerating motion for: "{user_args.text}"')
    joints, motion_vec, length = generate_single(args, model, diffusion, mean, std, image_feature=image_feature)
    print(f'Generated motion: {length} frames  ({length / FPS:.1f}s)')

    # Postprocess: remove sideways sliding
    print('Postprocessing: removing sideways sliding...')
    joints, motion_vec = apply_8shape_trajectory_regularization(joints, motion_vec)

    # Postprocess: remove foot sliding
    print('Postprocessing: removing foot sliding...')
    joints = remove_foot_sliding(joints, fps=FPS)

    # Postprocess: adjust root translation so lowest joint is on floor
    print('Postprocessing: adjusting root motion to ensure lowest joint on floor...')
    joints = adjust_root_to_floor(joints)

    # Save
    model_name = sanitise(
        os.path.basename(os.path.dirname(user_args.model_path)) + '_' +
        os.path.basename(user_args.model_path).replace('model', '').replace('.pt', '')
    )
    save_results(joints, motion_vec, length, user_args.text, out_dir, model_name, args,
                 image_path=image_path_for_save, image_feature=image_feature)

    print(f'\n[Done] Results saved to: {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()
