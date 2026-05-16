import argparse
import os
import time
import yaml
import numpy as np
from typing import Dict, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

import wandb

from vjepa_dataset import VJEPADataset
from model_vjepa_diffusion import VJepa2NavModel, IMAGENET_MEAN, IMAGENET_STD


def diffusion_loss(noise_pred: torch.Tensor, noise_target: torch.Tensor, mask: torch.Tensor):
    """mse loss on noise prediction."""
    mse = F.mse_loss(noise_pred, noise_target, reduction="none").mean(dim=-1)
    loss = mse.mean()
    return loss, {"diffusion_loss": loss.detach()}


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    b, c, h, w = images.shape
    n_frames = c // 3
    if n_frames > 1:
        images = images.view(b * n_frames, 3, h, w)
        images = (images - mean) / std
        return images.view(b, c, h, w)
    return (images - mean) / std


def visualize_multi_trajectories(samples: torch.Tensor, target: torch.Tensor, n_batch: int = 4):
    """visualize multiple trajectory samples vs ground truth."""
    samples = samples[:n_batch].cpu().numpy()  # [n_batch, num_samples, len_traj, 2]
    target = target[:n_batch].cpu().numpy()
    
    fig, axes = plt.subplots(1, n_batch, figsize=(4 * n_batch, 4))
    if n_batch == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(target[i, :, 0], target[i, :, 1], "g.-", label="gt", linewidth=2, markersize=8)
        ax.plot(target[i, -1, 0], target[i, -1, 1], "bo", label="goal", markersize=10)
        for s in range(samples.shape[1]):
            alpha = 0.5 if s > 0 else 1.0
            label = "samples" if s == 0 else None
            ax.plot(samples[i, s, :, 0], samples[i, s, :, 1], "r.--", alpha=alpha, linewidth=1, markersize=4, label=label)
        ax.plot(0, 0, "ko", markersize=10, label="start")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"sample {i}")
    
    plt.tight_layout()
    return fig


def train_epoch(model, dataloader, optimizer, device, context_size, log_freq, use_wandb, max_norm, vis_freq=500, eval_step_freq=None, test_loaders=None, global_step=0, max_eval_batches=None, checkpoint_step_freq=None, save_checkpoint_fn=None):
    model.train()
    enc = model.module.encoder if hasattr(model, "module") else model.encoder
    enc.eval()  # keep encoder frozen

    total_loss = 0.0
    total_data_time = 0.0
    for step, batch_data in enumerate(tqdm(dataloader, desc="training")):
        data_start = time.time()
        obs, goal_image, goal_xy, actions, mask = batch_data
        data_time = time.time() - data_start
        total_data_time += data_time

        obs_norm = normalize_batch(obs, device)
        goal_norm = normalize_batch(goal_image, device)
        goal_xy = goal_xy.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        noise_pred, noise_target = model(obs_norm, goal_norm, goal_xy, target=actions)
        loss, metrics = diffusion_loss(noise_pred, noise_target, mask)
        loss.backward()

        # grad norm computed before clipping for accurate logging
        base_model = model.module if hasattr(model, "module") else model
        grad_norm = sum(p.grad.norm().item() for p in base_model.parameters() if p.grad is not None)
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss += loss.item()
        current_step = global_step + step
        if step % log_freq == 0:
            avg_data_time = total_data_time / (step + 1)
            print(f"train step {step}/{len(dataloader)-1} loss {loss.item():.4f} grad_norm {grad_norm:.4f} data_time {data_time*1000:.2f}ms avg_data_time {avg_data_time*1000:.2f}ms")
        
        if use_wandb and wandb:
            log_dict = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm,
                "train/goal_xy_norm": goal_xy.norm(dim=-1).mean().item(),
            }

            if step % 50 == 0:
                avg_data_time = total_data_time / (step + 1)
                log_dict.update({
                    "train/data_time_ms": data_time * 1000,
                    "train/avg_data_time_ms": avg_data_time * 1000,
                })

            if step % vis_freq == 0:
                with torch.no_grad():
                    samples, _, _ = model(obs_norm, goal_norm, goal_xy)
                    traj_fig = visualize_multi_trajectories(samples.detach(), actions, n_batch=min(4, samples.shape[0]))
                    log_dict["train/trajectories"] = wandb.Image(traj_fig)
                    plt.close(traj_fig)
            
            wandb.log(log_dict)

        if eval_step_freq and test_loaders and current_step > 0 and current_step % eval_step_freq == 0:
            model.eval()
            for name, loader in test_loaders.items():
                eval_loss = eval_epoch(model, loader, device, context_size, use_wandb, name=f"eval/{name}", max_batches=max_eval_batches)
                print(f"eval (step {current_step}) {name}: {eval_loss:.4f}")
            model.train()
            enc.eval()  # keep encoder frozen

        if checkpoint_step_freq and save_checkpoint_fn and current_step > 0 and current_step % checkpoint_step_freq == 0:
            save_checkpoint_fn(current_step)
    
    avg_data_time = total_data_time / max(len(dataloader), 1)
    print(f"epoch data loading: total {total_data_time:.2f}s, avg per batch {avg_data_time*1000:.2f}ms")

    try:
        dataset = dataloader.dataset
        if hasattr(dataset, 'print_stats'):
            dataset.print_stats()
        elif hasattr(dataset, 'datasets'):  # ConcatDataset
            for i, ds in enumerate(dataset.datasets):
                if hasattr(ds, 'print_stats'):
                    print(f"dataset {i}:")
                    ds.print_stats()
    except Exception as e:
        pass
    
    if use_wandb and wandb:
        wandb.log({
            "train/epoch_data_time_total_s": total_data_time,
            "train/epoch_avg_data_time_ms": avg_data_time * 1000,
        })
    return total_loss / max(len(dataloader), 1)


def eval_epoch(model, dataloader, device, context_size, use_wandb, name="eval", max_batches=None):
    model.eval()
    total_loss = 0.0
    total_data_time = 0.0
    vis_logged = False
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=name):
            if max_batches is not None and num_batches >= max_batches:
                break
            data_start = time.time()
            obs, goal_image, goal_xy, actions, mask = batch_data
            data_time = time.time() - data_start
            total_data_time += data_time

            obs_norm = normalize_batch(obs, device)
            goal_norm = normalize_batch(goal_image, device)
            goal_xy = goal_xy.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            noise_pred, noise_target = model(obs_norm, goal_norm, goal_xy, target=actions)
            loss, metrics = diffusion_loss(noise_pred, noise_target, mask)
            total_loss += loss.item()
            num_batches += 1

            if use_wandb and wandb and not vis_logged:
                samples, _, _ = model(obs_norm, goal_norm, goal_xy)
                traj_fig = visualize_multi_trajectories(samples, actions, n_batch=min(4, samples.shape[0]))
                wandb.log({
                    f"{name}/trajectories": wandb.Image(traj_fig)
                })
                plt.close(traj_fig)
                vis_logged = True
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_data_time = total_data_time / max(num_batches, 1)
    if num_batches > 0:
        print(f"{name} data loading: total {total_data_time:.2f}s, avg per batch {avg_data_time*1000:.2f}ms")
    if use_wandb and wandb:
        wandb.log({
            f"{name}/loss": avg_loss,
            f"{name}/data_time_ms": avg_data_time * 1000,
        })
    return avg_loss


def _worker_init_fn(worker_id):
    """initialize worker process - set different seed for each worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloaders(config: Dict):
    train_datasets, train_weights, test_loaders = [], [], {}

    for name, cfg in config["datasets"].items():
        for split in ["train", "test"]:
            if split not in cfg:
                continue
            with open(os.path.join(cfg[split], "traj_names.txt")) as f:
                traj_names = [l.strip() for l in f if l.strip()]

            dataset = VJEPADataset(
                data_folder=cfg["data_folder"],
                traj_names=traj_names,
                image_size=tuple(config["image_size"]),
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                waypoint_spacing=cfg.get("waypoint_spacing", 1),
                min_action_dist=config["action"]["min_dist_cat"],
                max_action_dist=config["action"]["max_dist_cat"],
                max_goal_dist=config["distance"]["max_dist_cat"],
                end_slack=cfg.get("end_slack", 0),
                normalize=config["normalize"],
                metric_waypoint_spacing=cfg.get("metric_waypoint_spacing", 0.25),
                action_normalization_factor=config.get("action_normalization_factor"),
                learn_angle=config["learn_angle"],
                dataset_name=name,
                rectify_images=cfg.get("rectify_images", False),
                use_lmdb_cache=cfg.get("use_lmdb_cache", False),
                data_split_folder=cfg[split],
            )

            if split == "train":
                train_datasets.append(dataset)
                w = cfg.get("dataset_weight", 1.0)
                # per-sample weight = dataset_weight / dataset_size so each
                # dataset's total probability mass equals its dataset_weight
                train_weights.extend([w / len(dataset)] * len(dataset))
            else:
                test_loaders[f"{name}_{split}"] = DataLoader(
                    dataset, batch_size=config["eval_batch_size"], shuffle=False, num_workers=0
                )

    prefetch_factor = config.get("prefetch_factor", 2)
    num_workers = config.get("num_workers", 0)

    if num_workers > 0:
        # scale prefetch with batch_size/num_workers to keep GPU fed on HDD-backed datasets
        prefetch_factor = max(prefetch_factor, min(4, max(2, config["batch_size"] // max(1, num_workers // 2))))
        print(f"dataloader: {num_workers} workers, prefetch_factor={prefetch_factor}")

    concat = ConcatDataset(train_datasets)
    all_weights = torch.tensor(train_weights, dtype=torch.double)
    use_weighted = any(cfg.get("dataset_weight", 1.0) != 1.0 for cfg in config["datasets"].values())
    if use_weighted:
        sampler = WeightedRandomSampler(all_weights, num_samples=len(concat), replacement=True)
        print(f"dataset weights: { {n: cfg.get('dataset_weight', 1.0) for n, cfg in config['datasets'].items()} }")
        shuffle, sampler_arg = False, sampler
    else:
        shuffle, sampler_arg = True, None

    train_loader = DataLoader(
        concat,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        sampler=sampler_arg,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    return train_loader, test_loaders


def _expand_env_vars(obj):
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def prepare_config(config_path: str) -> Dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _expand_env_vars(cfg)

    defaults = {
        "use_wandb": False, "batch_size": 32, "epochs": 50, "gpu_ids": [0],
        "num_workers": 4, "prefetch_factor": 2, "lr": 1e-4, "optimizer": "adamw", "scheduler": None,
        "clipping": False, "max_norm": 1.0, "context_size": 6, "len_traj_pred": 8,
        "learn_angle": False, "vision_hidden_dim": 1024, "normalize": False,
        "num_samples": 1,  # number of samples for training visualization
        "inference_num_samples": 8,  # number of samples for inference
        "num_diffusion_steps": 50,
                "distance": {"min_dist_cat": 0, "max_dist_cat": 20},
        "action": {"min_dist_cat": 2, "max_dist_cat": 20},
        "image_size": [256, 256], "print_log_freq": 100, "eval_freq": 1,
        "eval_step_freq": 500,  # evaluate every N training steps
        "image_log_freq": 500,  # visualization frequency
        "max_eval_batches": 10,  # limit number of batches evaluated (None for all)
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    cfg.setdefault("eval_batch_size", cfg["batch_size"])
    cfg.setdefault("action_dim", 3 if cfg["learn_angle"] else 2)
    cfg["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    cfg["project_folder"] = os.path.join("logs", cfg["project_name"], cfg["run_name"])
    os.makedirs(cfg["project_folder"], exist_ok=True)
    return cfg


def main(config: Dict):
    """
    main training loop for vjepa2 navigation model.
    
    handles:
    - dataset loading and batching
    - model initialization (frozen vjepa2 + trainable diffusion components)
    - training with diffusion loss (mse on noise prediction)
    - periodic evaluation and visualization
    - checkpoint saving (step-based and epoch-based)
    - wandb logging (if enabled)
    """
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if isinstance(config["gpu_ids"], int):
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config["gpu_ids"]))
        print("using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])

    device = torch.device(f"cuda:{config['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
    cudnn.benchmark = True

    train_loader, test_loaders = build_dataloaders(config)

    total_dataset_size = len(train_loader.dataset)
    steps_per_epoch = len(train_loader)
    total_epochs = config["epochs"]
    total_steps = steps_per_epoch * total_epochs
    
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Total dataset size: {total_dataset_size:,} samples")
    print(f"Batch size: {config['batch_size']}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total epochs: {total_epochs}")
    print(f"Total training steps: {total_steps:,}")
    print("=" * 60)

    if test_loaders:
        print("\nTest datasets:")
        for name, loader in test_loaders.items():
            print(f"  {name}: {len(loader.dataset):,} samples")
    print()
    
    model = VJepa2NavModel(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        action_dim=config["action_dim"],
        hidden_dim=config["vision_hidden_dim"],
        inference_num_samples=config.get("inference_num_samples", 8),
        num_diffusion_steps=config["num_diffusion_steps"],
        num_fusion_layers=config.get("num_fusion_layers", 2),
        unet_down_dims=config.get("unet_down_dims"),
        use_residual=config.get("use_residual", True),
        diffusion_clip_sample=config.get("diffusion_clip_sample", True),
        diffusion_clip_sample_range=config.get("diffusion_clip_sample_range", 1.0),
        diffusion_beta_schedule=config.get("diffusion_beta_schedule", "squaredcos_cap_v2"),
        diffusion_prediction_type=config.get("diffusion_prediction_type", "epsilon"),
        cfg_dropout_prob=config.get("cfg_dropout_prob", 0.0),
        cfg_guidance_scale=config.get("cfg_guidance_scale", 1.0),
    )
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = (torch.optim.Adam if config["optimizer"].lower() == "adam" else torch.optim.AdamW)(
        trainable, lr=float(config["lr"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"]) if config["scheduler"] else None

    start_epoch = 0
    global_step = 0
    if config.get("resume_checkpoint"):
        checkpoint_path = config["resume_checkpoint"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        print(f"loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        base_model = model.module if hasattr(model, "module") else model
        missing, unexpected = base_model.load_state_dict(checkpoint["model"], strict=False)
        print(f"loaded model: {len(missing)} missing, {len(unexpected)} unexpected keys")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", start_epoch * steps_per_epoch)
        print(f"resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}, global_step {global_step}")

    use_wandb = config["use_wandb"] and wandb
    if use_wandb:
        wandb.login()
        wandb.init(project=config["project_name"], name=config["run_name"], settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config)

    print("config:", config)
    
    def save_checkpoint(step, epoch=None):
        base_model = model.module if hasattr(model, "module") else model
        # skip frozen vjepa2 encoder weights to keep checkpoints small
        sd = {k: v for k, v in base_model.state_dict().items() if not k.startswith("encoder.")}
        ckpt = {
            "epoch": epoch if epoch is not None else -1,
            "model": sd,
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "global_step": step,
        }
        if scheduler:
            ckpt["scheduler_state"] = scheduler.state_dict()
        torch.save(ckpt, os.path.join(config["project_folder"], "latest.pth"))
        torch.save(ckpt, os.path.join(config["project_folder"], f"step_{step}.pth"))
        print(f"saved checkpoint at step {step}")
    
    for epoch in range(start_epoch, config["epochs"]):
        print(f"epoch {epoch}/{config['epochs'] - 1}")
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            config["context_size"], config["print_log_freq"],
            use_wandb, config["max_norm"] if config["clipping"] else 0.0,
            vis_freq=config["image_log_freq"],
            eval_step_freq=config.get("eval_step_freq"),
            test_loaders=test_loaders,
            global_step=global_step,
            max_eval_batches=config.get("max_eval_batches"),
            checkpoint_step_freq=config.get("checkpoint_step_freq"),
            save_checkpoint_fn=save_checkpoint,
        )
        global_step += len(train_loader)
        if scheduler:
            scheduler.step()

        if (epoch + 1) % config["eval_freq"] == 0:
            for name, loader in test_loaders.items():
                eval_loss = eval_epoch(model, loader, device, config["context_size"], use_wandb, name=f"eval/{name}", max_batches=config.get("max_eval_batches"))
                print(f"eval {name}: {eval_loss:.4f}")

        save_checkpoint(global_step, epoch=epoch)
        print(f"saved checkpoint for epoch {epoch}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Train navigation with frozen VJEPA2 Large")
    parser.add_argument("--config", "-c", default="config/vjepa.yaml", help="config file path")
    parser.add_argument("--resume", "-r", default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()
    cfg = prepare_config(args.config)
    if args.resume:
        cfg["resume_checkpoint"] = args.resume
    main(cfg)
