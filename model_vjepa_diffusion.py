import torch
import torch.nn as nn
import time

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as DiffusersDDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler


VJEPA2_LARGE_EMBED_DIM = 1024
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VJepa2NavModel(nn.Module):
    """
    navigation model: frozen vjepa2 (obs clip + goal frame) -> token fusion -> diffusion policy.

    pipeline:
    - obs context [B, 3, Tc, H, W] and goal frame (duplicated to [B, 3, 2, H, W])
      are encoded by the same frozen vjepa2 once each.
    - obs/goal tokens get learned type embeddings, then a small transformer fuses them.
    - a learned CLS query cross-attends over fused tokens to produce a single conditioning vector.
    - goal_xy is encoded by an MLP and merged into the conditioning (auxiliary signal).
    - ConditionalUnet1D denoiser predicts noise/epsilon over action chunks.
    - classifier-free guidance: random goal-token dropout during training; null-cond
      hoisted out of the diffusion loop at inference.
    """

    def __init__(
        self,
        context_size: int,
        len_traj_pred: int,
        action_dim: int,
        hidden_dim: int,
        goal_xy_dim: int = 2,
        inference_num_samples: int = 8,
        num_diffusion_steps: int = 50,
        num_attn_heads: int = 8,
        num_fusion_layers: int = 2,
        unet_down_dims: list = None,
        use_residual: bool = True,
        diffusion_clip_sample: bool = True,
        diffusion_clip_sample_range: float = 1.0,
        diffusion_beta_schedule: str = "squaredcos_cap_v2",
        diffusion_prediction_type: str = "epsilon",
        cfg_dropout_prob: float = 0.0,
        cfg_guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.inference_num_samples = inference_num_samples
        self.num_diffusion_steps = num_diffusion_steps
        self.use_residual = use_residual
        self.cfg_dropout_prob = cfg_dropout_prob
        self.cfg_guidance_scale = cfg_guidance_scale
        self._cfg_enabled = cfg_dropout_prob > 0.0 or cfg_guidance_scale > 1.0

        # frozen vjepa2 encoder (shared for obs and goal)
        print("loading vjepa2_vit_large...")
        loaded = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
        self.encoder = loaded[0] if isinstance(loaded, tuple) else loaded
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # token projections + type embeddings (0=obs, 1=goal)
        self.obs_proj = nn.Linear(VJEPA2_LARGE_EMBED_DIM, hidden_dim)
        self.goal_proj = nn.Linear(VJEPA2_LARGE_EMBED_DIM, hidden_dim)
        self.type_embed = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.type_embed, std=0.02)

        # learned null token replacing goal tokens for cfg
        if self._cfg_enabled:
            self.null_goal_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.null_goal_token, std=0.02)

        # fusion transformer over [obs_tokens; goal_tokens]
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attn_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=num_fusion_layers)

        # learned CLS-style query that cross-attends over fused tokens to produce cond
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_attn_heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # auxiliary goal_xy branch (kept so use_residual keeps working)
        self.goal_xy_mlp = nn.Sequential(
            nn.Linear(goal_xy_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_merge = nn.Linear(hidden_dim * 2, hidden_dim)

        # diffusion denoiser (unet1d only)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        cond_dim = hidden_dim + hidden_dim  # cond + time_embed
        self.denoiser = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            down_dims=unet_down_dims or [256, 512, 1024],
            cond_predict_scale=False,
        )

        self.scheduler = DiffusersDDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule=diffusion_beta_schedule,
            clip_sample=diffusion_clip_sample,
            clip_sample_range=diffusion_clip_sample_range,
            prediction_type=diffusion_prediction_type,
        )
        self._scheduler_config = {
            "num_train_timesteps": num_diffusion_steps,
            "beta_schedule": diffusion_beta_schedule,
            "clip_sample": diffusion_clip_sample,
            "clip_sample_range": diffusion_clip_sample_range,
            "prediction_type": diffusion_prediction_type,
        }

    def set_inference_scheduler(self, scheduler_type: str = "ddpm", num_inference_steps: int = None):
        """swap scheduler for faster inference (ddpm/ddim/dpmsolver)."""
        if num_inference_steps is None:
            num_inference_steps = self.num_diffusion_steps
        scheduler_type = scheduler_type.lower()
        if scheduler_type == "ddpm":
            self.scheduler = DiffusersDDPMScheduler(**self._scheduler_config)
        elif scheduler_type == "ddim":
            self.scheduler = DDIMScheduler(**self._scheduler_config)
        elif scheduler_type == "dpmsolver":
            self.scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self._scheduler_config["num_train_timesteps"],
                beta_schedule=self._scheduler_config["beta_schedule"],
                algorithm_type="dpmsolver++",
                solver_order=2,
                prediction_type=self._scheduler_config["prediction_type"],
            )
        else:
            raise ValueError(f"unknown scheduler type: {scheduler_type}")
        self._num_inference_steps = num_inference_steps
        print(f"switched to {scheduler_type.upper()} scheduler with {num_inference_steps} inference steps")

    def _base_traj(self, goal_xy: torch.Tensor) -> torch.Tensor:
        """deterministic straight-line path from (0,0) to goal; yaw stays zero."""
        b = goal_xy.shape[0]
        device, dtype = goal_xy.device, goal_xy.dtype
        steps = torch.arange(1, self.len_traj_pred + 1, device=device, dtype=dtype).view(1, -1, 1)
        frac = steps / float(self.len_traj_pred)
        base = torch.zeros(b, self.len_traj_pred, self.action_dim, device=device, dtype=dtype)
        base[:, :, :2] = goal_xy[:, None, :2] * frac
        return base

    @torch.no_grad()
    def _encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """[B, C, T, H, W] -> [B, N, D]. Pads T to be even (vjepa2 tubelet=2)."""
        t = clip.shape[2]
        if t < 2:
            clip = clip.repeat(1, 1, 2, 1, 1)
        elif t % 2 == 1:
            clip = torch.cat([clip, clip[:, :, -1:].clone()], dim=2)
        return self.encoder(clip)

    def get_condition(self, obs: torch.Tensor, goal_image: torch.Tensor, goal_xy: torch.Tensor,
                      drop_goal_mask: torch.Tensor = None) -> torch.Tensor:
        """
        obs:        [B, 3*Tc, H, W] flattened context
        goal_image: [B, 3, H, W]
        goal_xy:    [B, goal_xy_dim]
        drop_goal_mask: [B] bool; True -> replace goal tokens with null (cfg)
        """
        b, _, h, w = obs.shape
        obs_clip = obs.view(b, self.context_size, 3, h, w).permute(0, 2, 1, 3, 4)  # [B,3,Tc,H,W]
        goal_clip = goal_image.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [B,3,2,H,W]

        with torch.no_grad():
            obs_tokens = self._encode_clip(obs_clip)    # [B, No, D_enc]
            goal_tokens = self._encode_clip(goal_clip)  # [B, Ng, D_enc]

        obs_tokens = self.obs_proj(obs_tokens) + self.type_embed[0]
        goal_tokens = self.goal_proj(goal_tokens) + self.type_embed[1]

        if drop_goal_mask is not None and self._cfg_enabled:
            null = self.null_goal_token.expand(b, goal_tokens.shape[1], -1) + self.type_embed[1]
            mask = drop_goal_mask.view(b, 1, 1)
            goal_tokens = torch.where(mask, null, goal_tokens)

        tokens = torch.cat([obs_tokens, goal_tokens], dim=1)
        tokens = self.fusion(tokens)

        q = self.pool_query.expand(b, -1, -1)
        pooled, _ = self.pool_attn(q, tokens, tokens)
        pooled = self.pool_norm(pooled.squeeze(1))  # [B, H]

        gxy = self.goal_xy_mlp(goal_xy.view(b, -1))
        return self.cond_merge(torch.cat([pooled, gxy], dim=-1))  # [B, H]

    def predict_noise(self, noisy_traj: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = noisy_traj.shape[0]
        if noisy_traj.dim() == 2:
            noisy_traj = noisy_traj.reshape(b, self.len_traj_pred, self.action_dim)
        t_embed = self.time_embed(t.float().view(-1, 1) / self.num_diffusion_steps)
        global_cond = torch.cat([cond, t_embed], dim=-1)
        return self.denoiser(sample=noisy_traj, timestep=t.long(), global_cond=global_cond)

    def forward(self, obs: torch.Tensor, goal_image: torch.Tensor, goal_xy: torch.Tensor,
                target: torch.Tensor = None):
        backbone_start = time.time()
        b = obs.shape[0]

        if target is not None:
            # training: cfg dropout per-sample
            drop_mask = None
            if self.cfg_dropout_prob > 0.0 and self.training:
                drop_mask = torch.rand(b, device=obs.device) < self.cfg_dropout_prob
            cond = self.get_condition(obs, goal_image, goal_xy, drop_goal_mask=drop_mask)
            backbone_time = time.time() - backbone_start

            base_traj = self._base_traj(goal_xy) if self.use_residual else None
            traj_to_noise = (target - base_traj) if self.use_residual else target

            noise = torch.randn_like(traj_to_noise)
            t = torch.randint(0, self.num_diffusion_steps, (b,), device=obs.device)
            noisy = self.scheduler.add_noise(traj_to_noise, noise, t)
            noise_pred = self.predict_noise(noisy, cond, t)
            return noise_pred, noise

        # inference: precompute cond and (optionally) null cond once
        cond = self.get_condition(obs, goal_image, goal_xy)
        null_cond = None
        if self.cfg_guidance_scale > 1.0 and self._cfg_enabled:
            drop_all = torch.ones(b, dtype=torch.bool, device=obs.device)
            null_cond = self.get_condition(obs, goal_image, goal_xy, drop_goal_mask=drop_all)
        backbone_time = time.time() - backbone_start

        base_traj = self._base_traj(goal_xy) if self.use_residual else None
        num_inference_steps = getattr(self, "_num_inference_steps", self.num_diffusion_steps)
        self.scheduler.set_timesteps(num_inference_steps, device=obs.device)
        timesteps = self.scheduler.timesteps

        num_samples = getattr(self, "_override_num_samples", self.inference_num_samples)
        samples, sample_times = [], []
        for _ in range(num_samples):
            sample_start = time.time()
            x = torch.randn(b, self.len_traj_pred, self.action_dim, device=obs.device)
            for t in timesteps:
                t_batch = t.unsqueeze(0).repeat(b).to(obs.device)
                if null_cond is not None:
                    nc = self.predict_noise(x, cond, t_batch)
                    nu = self.predict_noise(x, null_cond, t_batch)
                    noise_pred = nu + self.cfg_guidance_scale * (nc - nu)
                else:
                    noise_pred = self.predict_noise(x, cond, t_batch)
                x = self.scheduler.step(model_output=noise_pred, timestep=t, sample=x).prev_sample

            # with randomized goal frame, goal_xy != actions[-1] in general,
            # so we no longer force the predicted last waypoint onto the goal.
            traj = (x + base_traj) if self.use_residual else x
            samples.append(traj)
            sample_times.append(time.time() - sample_start)
        return torch.stack(samples, dim=1), backbone_time, sample_times
