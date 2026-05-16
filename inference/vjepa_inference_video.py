"""rolling inference over a recon trajectory rendered to mp4.

at each timestep t:
  - obs context = frames [t-(Tc-1), ..., t]
  - goal image  = frame at t + goal_offset
  - goal_xy     = goal world position in robot's local frame at t (normalized)
  - run model -> N predicted trajectories
  - render: current obs | goal image | trajectory plot (GT vs predictions)
"""
import argparse
import os
import pickle
import time
from typing import Tuple

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from model_vjepa_diffusion import VJepa2NavModel, IMAGENET_MEAN, IMAGENET_STD
from vjepa_dataset import load_image, to_local_coords


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    b, c, h, w = images.shape
    n = c // 3
    if n > 1:
        images = images.view(b * n, 3, h, w)
        images = (images - mean) / std
        return images.view(b, c, h, w)
    return (images - mean) / std


def load_model(ckpt_path: str, device: torch.device, cfg: dict, num_samples: int) -> VJepa2NavModel:
    model = VJepa2NavModel(
        context_size=cfg.get("context_size", 5),
        len_traj_pred=cfg.get("len_traj_pred", 16),
        action_dim=3 if cfg.get("learn_angle", False) else 2,
        hidden_dim=cfg.get("vision_hidden_dim", 1024),
        inference_num_samples=num_samples,
        num_diffusion_steps=cfg.get("num_diffusion_steps", 40),
        num_fusion_layers=cfg.get("num_fusion_layers") or 2,
        unet_down_dims=cfg.get("unet_down_dims"),
        use_residual=cfg.get("use_residual", False),
        diffusion_clip_sample=cfg.get("diffusion_clip_sample", True),
        diffusion_clip_sample_range=cfg.get("diffusion_clip_sample_range", 1.0),
        diffusion_beta_schedule=cfg.get("diffusion_beta_schedule", "squaredcos_cap_v2"),
        diffusion_prediction_type=cfg.get("diffusion_prediction_type", "epsilon"),
        cfg_dropout_prob=cfg.get("cfg_dropout_prob", 0.0),
        cfg_guidance_scale=cfg.get("cfg_guidance_scale", 1.0),
    )
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"loaded checkpoint: step={ck.get('global_step')} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    return model


def build_context(folder: str, t: int, Tc: int, ws: int, image_size: Tuple[int, int]) -> torch.Tensor:
    times = [max(0, t - (Tc - 1 - i) * ws) for i in range(Tc)]
    imgs = [load_image(os.path.join(folder, f"{ti}.jpg"), image_size) for ti in times]
    return torch.cat(imgs, dim=0)  # [3*Tc, H, W]


def tensor_to_rgb(img_t: torch.Tensor) -> np.ndarray:
    """[3,H,W] in [0,1] -> [H,W,3] uint8."""
    arr = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def sample_mixed_modes(model, obs_n, goal_n, goal_xy, n_goal=1, n_explore=8):
    """sample one goal-directed trajectory and n_explore exploration trajectories.

    goal sample    : goal-conditioned (cond_goal), single denoiser call per step.
    explore samples: null-conditioned (cond_null, goal token masked out like NoMaD
                     explore.py), single denoiser call per step. diversity comes
                     purely from independent noise seeds — no goal correlation.

    returns (goal_samples [B, n_goal, L, A], explore_samples [B, n_explore, L, A]).
    """
    b = obs_n.shape[0]
    device = obs_n.device
    with torch.no_grad():
        if not model._cfg_enabled:
            raise RuntimeError("model was trained without cfg dropout; cannot run exploration mode")

        cond_goal = model.get_condition(obs_n, goal_n, goal_xy)
        drop_all = torch.ones(b, dtype=torch.bool, device=device)
        cond_null = model.get_condition(obs_n, goal_n, goal_xy, drop_goal_mask=drop_all)

        nsteps = getattr(model, "_num_inference_steps", model.num_diffusion_steps)
        model.scheduler.set_timesteps(nsteps, device=device)
        timesteps = model.scheduler.timesteps
        base_traj = model._base_traj(goal_xy) if model.use_residual else None

        def sample_once(cond):
            x = torch.randn(b, model.len_traj_pred, model.action_dim, device=device)
            for t in timesteps:
                t_batch = t.unsqueeze(0).repeat(b).to(device)
                noise_pred = model.predict_noise(x, cond, t_batch)
                x = model.scheduler.step(model_output=noise_pred, timestep=t, sample=x).prev_sample
            return (x + base_traj) if model.use_residual else x

        goal_list = [sample_once(cond_goal) for _ in range(n_goal)]
        explore_list = [sample_once(cond_null) for _ in range(n_explore)]
    return torch.stack(goal_list, dim=1), torch.stack(explore_list, dim=1)


def render_frame(
    fig,
    axes,
    obs_rgb: np.ndarray,
    goal_rgb: np.ndarray,
    goal_pred_m: np.ndarray,    # [Ng, L, 2] in meters - goal-conditioned
    explore_pred_m: np.ndarray, # [Ne, L, 2] in meters - exploration (null cond)
    gt_m: np.ndarray,           # [L, 2]   in meters
    goal_xy_m: np.ndarray,      # [2]      in meters
    t: int,
    goal_t: int,
    plot_lim: float,
) -> np.ndarray:
    for ax in axes:
        ax.cla()

    ax_obs, ax_goal, ax_traj = axes
    ax_obs.imshow(obs_rgb)
    ax_obs.set_title(f"observation t={t}")
    ax_obs.axis("off")

    ax_goal.imshow(goal_rgb)
    ax_goal.set_title(f"goal frame t={goal_t}")
    ax_goal.axis("off")

    # exploration trajectories first (faded orange) so goal-directed sits on top
    for s in range(explore_pred_m.shape[0]):
        label = "exploration (null goal)" if s == 0 else None
        ax_traj.plot(explore_pred_m[s, :, 0], explore_pred_m[s, :, 1],
                     color="orange", linestyle="-", marker=".",
                     alpha=0.55, linewidth=1, markersize=3, label=label)
    for s in range(goal_pred_m.shape[0]):
        label = "goal-directed" if s == 0 else None
        ax_traj.plot(goal_pred_m[s, :, 0], goal_pred_m[s, :, 1],
                     "r.-", alpha=0.95, linewidth=1.8, markersize=4, label=label)
    ax_traj.plot(gt_m[:, 0], gt_m[:, 1], "g.-", linewidth=1.5, markersize=4, label="ground truth")
    ax_traj.plot(goal_xy_m[0], goal_xy_m[1], "b*", markersize=14, label="goal xy")
    ax_traj.plot(0, 0, "ko", markersize=8, label="robot")
    ax_traj.set_xlim(-plot_lim, plot_lim)
    ax_traj.set_ylim(-plot_lim, plot_lim)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel("x forward (m)")
    ax_traj.set_ylabel("y left (m)")
    ax_traj.set_title("trajectory (robot frame)")
    ax_traj.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--folder", "-f", required=True, help="recon trajectory folder containing *.jpg + traj_data.pkl")
    p.add_argument("--output", "-o", required=True, help="output mp4 path")
    p.add_argument("--start_frame", type=int, default=None, help="default: context_size-1")
    p.add_argument("--end_frame", type=int, default=None, help="default: traj_len - goal_offset - 1")
    p.add_argument("--goal_offset", type=int, default=18, help="goal frame is current+offset")
    p.add_argument("--num_samples", type=int, default=8, help="(legacy) total samples used to size internal model setup")
    p.add_argument("--n_goal", type=int, default=1, help="goal-directed samples (red)")
    p.add_argument("--n_explore", type=int, default=7, help="exploration samples (orange, goal-dropped)")
    p.add_argument("--inference_steps", type=int, default=None, help="override num diffusion steps")
    p.add_argument("--scheduler", type=str, default=None, choices=[None, "ddpm", "ddim", "dpmsolver"])
    p.add_argument("--cfg_guidance_scale", type=float, default=None)
    p.add_argument("--metric_waypoint_spacing", type=float, default=0.25, help="recon=0.25")
    p.add_argument("--waypoint_spacing", type=int, default=1)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_frames", type=int, default=None, help="cap number of rendered frames")
    p.add_argument("--plot_lim", type=float, default=8.0, help="trajectory plot half-extent in meters")
    args = p.parse_args()

    device = torch.device(args.device)
    pkl_path = os.path.join(args.folder, "traj_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    Tc = cfg.get("context_size", 5)
    Lt = cfg.get("len_traj_pred", 16)
    image_size = tuple(cfg.get("image_size", [256, 256]))
    ws = args.waypoint_spacing
    norm_factor = args.metric_waypoint_spacing * ws if cfg.get("normalize", True) else 1.0

    model = load_model(args.checkpoint, device, cfg, num_samples=args.num_samples)
    if args.cfg_guidance_scale is not None:
        model.cfg_guidance_scale = args.cfg_guidance_scale
    if args.scheduler is not None or args.inference_steps is not None:
        sched = args.scheduler or "ddpm"
        nsteps = args.inference_steps or cfg.get("num_diffusion_steps", 40)
        model.set_inference_scheduler(sched, nsteps)

    with open(pkl_path, "rb") as f:
        traj = pickle.load(f)
    positions = np.asarray(traj["position"])  # [N, 2]
    yaws = np.asarray(traj["yaw"]).squeeze()  # [N]
    N = positions.shape[0]

    available = sorted(int(x.split(".")[0]) for x in os.listdir(args.folder) if x.endswith(".jpg") and x.split(".")[0].isdigit())
    last_avail = available[-1]
    N = min(N, last_avail + 1)
    print(f"trajectory length: {N} frames, last image idx: {last_avail}")

    start = args.start_frame if args.start_frame is not None else (Tc - 1) * ws
    end = args.end_frame if args.end_frame is not None else (N - args.goal_offset - 1)
    end = min(end, N - 1 - args.goal_offset)
    start = max(start, (Tc - 1) * ws)
    if end <= start:
        raise ValueError(f"empty range: start={start} end={end}")
    frames = list(range(start, end + 1))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    print(f"rendering {len(frames)} frames: t in [{frames[0]}, {frames[-1]}], goal_offset={args.goal_offset}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    writer = None

    t0 = time.time()
    for i, t in enumerate(frames):
        goal_t = min(t + args.goal_offset, N - 1)

        obs = build_context(args.folder, t, Tc, ws, image_size).unsqueeze(0)
        goal_img = load_image(os.path.join(args.folder, f"{goal_t}.jpg"), image_size).unsqueeze(0)

        curr_pos = positions[t]
        curr_yaw = float(yaws[t])
        goal_pos = positions[goal_t]
        goal_xy_local_m = to_local_coords(goal_pos[None], curr_pos, curr_yaw)[0]  # meters
        goal_xy_norm = goal_xy_local_m / norm_factor
        goal_xy_in = torch.as_tensor(goal_xy_norm, dtype=torch.float32, device=device).unsqueeze(0)

        gt_end = min(t + Lt * ws, N - 1)
        gt_world = positions[t:gt_end + 1:ws]
        if gt_world.shape[0] < Lt + 1:
            pad = np.tile(gt_world[-1:], (Lt + 1 - gt_world.shape[0], 1))
            gt_world = np.concatenate([gt_world, pad], axis=0)
        gt_local_m = to_local_coords(gt_world, curr_pos, curr_yaw)[1:]  # [Lt, 2] meters

        obs_n = normalize_batch(obs, device)
        goal_n = normalize_batch(goal_img, device)

        goal_samples, explore_samples = sample_mixed_modes(
            model, obs_n, goal_n, goal_xy_in,
            n_goal=args.n_goal, n_explore=args.n_explore,
        )
        goal_pred_m = goal_samples[0].cpu().numpy() * norm_factor
        explore_pred_m = explore_samples[0].cpu().numpy() * norm_factor

        # for the panels: take the latest obs frame (last 3 channels) and the goal image
        obs_last = obs[0, -3:]
        goal_disp = goal_img[0]
        obs_rgb = tensor_to_rgb(obs_last)
        goal_rgb = tensor_to_rgb(goal_disp)

        frame_bgr = render_frame(
            fig, axes, obs_rgb, goal_rgb,
            goal_pred_m[..., :2], explore_pred_m[..., :2],
            gt_local_m, goal_xy_local_m,
            t=t, goal_t=goal_t, plot_lim=args.plot_lim,
        )

        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"could not open VideoWriter for {args.output}")
            print(f"writing video {args.output}  size={w}x{h}  fps={args.fps}")
        writer.write(frame_bgr)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  frame {i+1}/{len(frames)}  t={t}  elapsed={elapsed:.1f}s  ({(i+1)/max(elapsed,1e-6):.2f} fps)")

    if writer is not None:
        writer.release()
    plt.close(fig)
    print(f"done. wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
