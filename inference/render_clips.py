"""render multiple inference clips in one process (single model load)."""
import os
import time
import argparse
import pickle
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vjepa_inference_video import (
    load_model, build_context, render_frame, normalize_batch, tensor_to_rgb,
    sample_mixed_modes,
)
from vjepa_dataset import load_image, to_local_coords


CLIPS = [
    # (folder, start_frame, n_frames)
    ("random_mdps_B_Jackal_GDC_Library_Sat_Nov_13_87_0", 420, 80),
    ("random_mdps_B_Jackal_Library_Sanjac_Sat_Nov_13_88_0", 590, 80),
    ("random_mdps_A_Jackal_Library_Fountain_Fri_Oct_29_10_0", 320, 80),
    ("random_mdps_A_Jackal_Library_AHG_Fri_Oct_29_8_0", 550, 80),
]
DATA_ROOT = "<path-to-datasets>/scand"
OUT_DIR = "<path-to-inference-videos>"
CKPT = "<path-to-checkpoint>.pth"
GOAL_OFFSET = 18
NUM_SAMPLES = 8  # only used to size model.inference_num_samples (unused at sample time)
N_GOAL = 1
N_EXPLORE = 8
FPS = 6
PLOT_LIM = 12.0  # scand metric_waypoint_spacing=0.38 so trajectories span more meters
WAYPOINT_SPACING = 1
METRIC_WS = 0.38  # scand


def render_clip(model, cfg, folder, start, n_frames, out_path, device):
    Tc = cfg.get("context_size", 5)
    Lt = cfg.get("len_traj_pred", 16)
    image_size = tuple(cfg.get("image_size", [256, 256]))
    norm_factor = METRIC_WS * WAYPOINT_SPACING if cfg.get("normalize", True) else 1.0

    pkl_path = os.path.join(folder, "traj_data.pkl")
    with open(pkl_path, "rb") as f:
        traj = pickle.load(f)
    positions = np.asarray(traj["position"])
    yaws = np.asarray(traj["yaw"]).squeeze()

    available = sorted(int(x.split(".")[0]) for x in os.listdir(folder)
                       if x.endswith(".jpg") and x.split(".")[0].isdigit())
    last_avail = available[-1]
    N = min(positions.shape[0], last_avail + 1)

    start = max(start, (Tc - 1) * WAYPOINT_SPACING)
    end = min(start + n_frames - 1, N - 1 - GOAL_OFFSET)
    if end <= start:
        print(f"  skipping {folder} - empty range")
        return
    frames = list(range(start, end + 1))
    print(f"  {os.path.basename(folder)}: rendering {len(frames)} frames t=[{frames[0]},{frames[-1]}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    writer = None
    t0 = time.time()
    for i, t in enumerate(frames):
        goal_t = min(t + GOAL_OFFSET, N - 1)
        obs = build_context(folder, t, Tc, WAYPOINT_SPACING, image_size).unsqueeze(0)
        goal_img = load_image(os.path.join(folder, f"{goal_t}.jpg"), image_size).unsqueeze(0)

        curr_pos = positions[t]
        curr_yaw = float(yaws[t])
        goal_pos = positions[goal_t]
        goal_xy_local_m = to_local_coords(goal_pos[None], curr_pos, curr_yaw)[0]
        goal_xy_norm = goal_xy_local_m / norm_factor
        goal_xy_in = torch.as_tensor(goal_xy_norm, dtype=torch.float32, device=device).unsqueeze(0)

        gt_end = min(t + Lt * WAYPOINT_SPACING, N - 1)
        gt_world = positions[t:gt_end + 1:WAYPOINT_SPACING]
        if gt_world.shape[0] < Lt + 1:
            pad = np.tile(gt_world[-1:], (Lt + 1 - gt_world.shape[0], 1))
            gt_world = np.concatenate([gt_world, pad], axis=0)
        gt_local_m = to_local_coords(gt_world, curr_pos, curr_yaw)[1:]

        obs_n = normalize_batch(obs, device)
        goal_n = normalize_batch(goal_img, device)
        goal_samples, explore_samples = sample_mixed_modes(
            model, obs_n, goal_n, goal_xy_in,
            n_goal=N_GOAL, n_explore=N_EXPLORE,
        )
        goal_pred_m = goal_samples[0].cpu().numpy() * norm_factor
        explore_pred_m = explore_samples[0].cpu().numpy() * norm_factor

        obs_rgb = tensor_to_rgb(obs[0, -3:])
        goal_rgb = tensor_to_rgb(goal_img[0])

        frame_bgr = render_frame(
            fig, axes, obs_rgb, goal_rgb,
            goal_pred_m[..., :2], explore_pred_m[..., :2],
            gt_local_m, goal_xy_local_m,
            t=t, goal_t=goal_t, plot_lim=PLOT_LIM,
        )
        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open writer for {out_path}")
        writer.write(frame_bgr)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(frames)}  elapsed={time.time()-t0:.0f}s")
    if writer is not None:
        writer.release()
    plt.close(fig)
    print(f"  -> wrote {out_path}  ({time.time()-t0:.0f}s)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = load_model(CKPT, device, cfg, num_samples=NUM_SAMPLES)
    print(f"model loaded. clips: {len(CLIPS)}")
    for traj_name, start, n in CLIPS:
        folder = os.path.join(DATA_ROOT, traj_name)
        out_path = os.path.join(OUT_DIR, f"clip_{traj_name}.mp4")
        if not os.path.isdir(folder):
            print(f"missing: {folder}")
            continue
        render_clip(model, cfg, folder, start, n, out_path, device)


if __name__ == "__main__":
    main()
