"""smoke test for vjepa2 + diffusion-policy: dataset, model fwd/bwd, inference.

usage: python3 smoke_test.py [config_path]
"""
import os
import sys
import time
import yaml
import torch

from vjepa_dataset import VJEPADataset
from model_vjepa_diffusion import VJepa2NavModel
from train_vjepa import normalize_batch, diffusion_loss


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/vjepa.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    name, ds_cfg = next(iter(cfg["datasets"].items()))
    with open(os.path.join(ds_cfg["train"], "traj_names.txt")) as f:
        traj_names = [l.strip() for l in f if l.strip()][:3]
    print(f"[smoke] using {len(traj_names)} trajectories from dataset '{name}'")

    dataset = VJEPADataset(
        data_folder=ds_cfg["data_folder"],
        traj_names=traj_names,
        image_size=tuple(cfg["image_size"]),
        context_size=cfg["context_size"],
        len_traj_pred=cfg["len_traj_pred"],
        waypoint_spacing=ds_cfg.get("waypoint_spacing", 1),
        min_action_dist=cfg["action"]["min_dist_cat"],
        max_action_dist=cfg["action"]["max_dist_cat"],
        max_goal_dist=cfg["distance"]["max_dist_cat"],
        end_slack=ds_cfg.get("end_slack", 0),
        normalize=cfg["normalize"],
        metric_waypoint_spacing=ds_cfg.get("metric_waypoint_spacing", 0.25),
        learn_angle=cfg["learn_angle"],
        dataset_name=name,
        rectify_images=ds_cfg.get("rectify_images", False),
        use_lmdb_cache=False,
        data_split_folder=ds_cfg["train"],
    )
    print(f"[smoke] dataset size: {len(dataset)}")

    item = dataset[0]
    obs_image, goal_image, goal_xy, actions, mask = item
    H, W = cfg["image_size"][1], cfg["image_size"][0]
    Tc = cfg["context_size"]
    A = 3 if cfg["learn_angle"] else 2
    Lt = cfg["len_traj_pred"]
    assert obs_image.shape == (3 * Tc, H, W), obs_image.shape
    assert goal_image.shape == (3, H, W), goal_image.shape
    assert goal_xy.shape == (2,), goal_xy.shape
    assert actions.shape == (Lt, A), actions.shape
    print(f"[smoke] dataset item shapes OK: obs={obs_image.shape}, goal={goal_image.shape}, "
          f"goal_xy={goal_xy.shape}, actions={actions.shape}")

    bs = 2
    batch = [dataset[i] for i in range(bs)]
    obs = torch.stack([b[0] for b in batch]).float()
    goal_image = torch.stack([b[1] for b in batch]).float()
    goal_xy = torch.stack([b[2] for b in batch])
    actions = torch.stack([b[3] for b in batch])
    mask = torch.stack([b[4] for b in batch])
    print(f"[smoke] batch shapes: obs={obs.shape}, goal_image={goal_image.shape}, "
          f"goal_xy={goal_xy.shape}, actions={actions.shape}")

    model = VJepa2NavModel(
        context_size=Tc,
        len_traj_pred=Lt,
        action_dim=A,
        hidden_dim=cfg["vision_hidden_dim"],
        inference_num_samples=2,
        num_diffusion_steps=cfg["num_diffusion_steps"],
        num_fusion_layers=cfg.get("num_fusion_layers", 2),
        unet_down_dims=cfg.get("unet_down_dims"),
        use_residual=cfg.get("use_residual", False),
        diffusion_clip_sample=cfg.get("diffusion_clip_sample", True),
        diffusion_clip_sample_range=cfg.get("diffusion_clip_sample_range", 1.0),
        diffusion_beta_schedule=cfg.get("diffusion_beta_schedule", "squaredcos_cap_v2"),
        diffusion_prediction_type=cfg.get("diffusion_prediction_type", "epsilon"),
        cfg_dropout_prob=cfg.get("cfg_dropout_prob", 0.0),
        cfg_guidance_scale=cfg.get("cfg_guidance_scale", 1.0),
    ).to(device)
    model.train()
    model.encoder.eval()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] params: trainable={n_train:,}  total={n_total:,}")

    obs_norm = normalize_batch(obs, device)
    goal_norm = normalize_batch(goal_image, device)
    goal_xy = goal_xy.to(device)
    actions = actions.to(device)
    mask = mask.to(device)

    t0 = time.time()
    noise_pred, noise_target = model(obs_norm, goal_norm, goal_xy, target=actions)
    t1 = time.time()
    print(f"[smoke] forward(train) {t1-t0:.2f}s  pred={tuple(noise_pred.shape)} target={tuple(noise_target.shape)}")
    assert noise_pred.shape == noise_target.shape

    loss, _ = diffusion_loss(noise_pred, noise_target, mask)
    print(f"[smoke] loss={loss.item():.4f}")

    t0 = time.time()
    loss.backward()
    t1 = time.time()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"[smoke] backward {t1-t0:.2f}s  grad_norm={grad_norm:.4f}")

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        samples, backbone_t, sample_ts = model(obs_norm, goal_norm, goal_xy)
    t1 = time.time()
    expected = (bs, 2, Lt, A)
    print(f"[smoke] inference {t1-t0:.2f}s  samples={tuple(samples.shape)} expected={expected}  "
          f"backbone={backbone_t*1000:.1f}ms")
    assert samples.shape == expected, (samples.shape, expected)

    print("[smoke] OK")


if __name__ == "__main__":
    main()
