#!/usr/bin/env python3
"""Open-loop eval that renders *what the policy is responding to* over each camera frame.

Same replay loop as ``gr00t/eval/open_loop_eval.py`` (ground-truth states in, predicted
actions out, no robot), with a saliency heatmap painted over the input frames as a video.

Two modes, answering genuinely different questions — ``--saliency``:

    attention   Where the model LOOKED. DiT action->image cross-attention. Forward-only and
                cheap, but 8x8 per camera and attention is not causation.
    gradcam     What the output is SENSITIVE TO. Gradient of the predicted action chunk
                w.r.t. pixel_values. Needs a backward pass through the denoising loop and the
                VLM (much heavier), but resolves at the vision input grid — typically 256x256,
                32x the spatial detail.

Neither proves the model *used* a region correctly. Attention shows allocation, gradient shows
local sensitivity; a high gradient means the prediction moves when that pixel moves, which is
still not evidence of deliberate, correct use. Treat both as hypothesis-generating.

An empirical caution from this repo: the better-scoring `dobot_optical_gear_pick` policy
(MSE 0.000082) showed MORE background attention than the worse cotrain checkpoint (0.000896).
Heatmap tidiness does not predict accuracy.

Reproducibility
---------------
The flow-matching loop starts from ``torch.randn`` (gr00t_n1d7.py:335), so predictions — and
the reported MSE — vary run to run unless seeded. Both modes call ``torch.manual_seed(--seed)``
before every inference. Measured drift when unseeded: 0.000064 -> 0.000089 MSE on a 96-step
trajectory, ~20%. Small MSE differences between runs mean nothing without a fixed seed.

With a matched ``--seed`` and ``--denoise-steps``, both modes produce *identical* predictions
(verified: 0.000064 / 0.005259 either way) — the saliency machinery only observes.

What the attention heatmap actually is
--------------------------------------
``AlternateVLDiT`` (gr00t/model/modules/dit.py) interleaves its 32 blocks:

    even idx, idx % (2*attend_text_every_n_blocks) == 0  -> cross-attend to TEXT tokens
    even idx, otherwise                                  -> cross-attend to IMAGE tokens
    odd idx                                              -> self-attention

For this checkpoint that puts image cross-attention at blocks [2, 6, 10, 14, 18, 22, 26, 30].
On each of those we recompute ``softmax(Q Kᵀ / sqrt(d))`` from the block's own ``to_q``/``to_k``
(the model runs SDPA, which never materializes the probabilities), average over heads and over
the action-chunk queries, and keep the column per image token. That vector is the model's
attention mass per image patch — higher = the action prediction drew more from that patch.

Scores are accumulated over every denoising step of every inference call, then averaged.

Mapping tokens back to pixels
-----------------------------
Qwen3VL emits ``image_grid_thw`` patches per image and merges ``spatial_merge_size**2`` of them
into one token, so a 480x640 frame becomes an 8x8 token grid here. The heatmap is therefore
genuinely 8x8, bilinearly upscaled to the frame — it shows coarse regions, not fine edges. Do
not read pixel-level precision into it.

Image tokens for the N cameras appear as N contiguous runs in the sequence, in the same order
as ``modality_config['video'].modality_keys``. The runs are located from ``image_mask`` and
split by the per-image token counts derived from ``image_grid_thw``.

What the gradcam heatmap actually is
------------------------------------
``d/d(pixel_values) sum(action_pred**2)``, i.e. how strongly the whole predicted chunk responds
to each input pixel. ``pixel_values`` is ``(n_patches, C*temporal*p*p)`` in Qwen2/3VL's
*merge-aware* order (not raster): flat index n decodes as ``(t, h_block, w_block, mh, mw)`` with
true position ``row = h_block*merge + mh``, ``col = w_block*merge + mw``. Verified by rebuilding
the image from ``pixel_values`` — ``--debug-reconstruct`` writes it out, and a coherent photo
proves the de-permutation (a wrong one is visibly block-shuffled).

``--cam-level patch`` (default) reduces each patch to one value, giving a ``grid_h x grid_w``
map — coarse, like classic Grad-CAM, which operates at feature-map not pixel resolution.
``--cam-level pixel`` keeps the full 256x256, which is really vanilla gradient saliency and is
markedly noisier.

Gradient magnitudes are heavy-tailed here: p99 is only ~15-25% of the max, so scaling the colour
map to the max renders the overlay almost uniformly flat. ``--vmax-percentile`` (default 99)
exists for this; set it to 100 for plain min-max.

Caveats worth keeping in mind
-----------------------------
- Attention is not causation. A bright patch is where the model attended, not proof it used
  that information correctly.
- Gradients are not importance either — they are *local* sensitivity, valid only for
  infinitesimal perturbations, and are known to be noisy for large models. In practice on this
  checkpoint the gradient maps are markedly more scattered and harder to read than the
  attention maps; do not assume the higher resolution means more insight.
- Only image-attending layers are captured for attention; text/state contributions are
  invisible in both modes.
- Averaging over 8 layers and all heads hides disagreement between them. Use --layers to
  inspect a single layer if a specific one is of interest.

Examples
--------
    cd /home/sharan/Isaac-GR00T

    # attention (default, cheap)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval/attention_overlay_eval.py \
        --model-path /home/sharan/G1_gear_cotrain/checkpoint-20000 \
        --dataset-path /home/sharan/cotrain/g1_gear_tcp_2.1_optical \
        --embodiment-tag new_embodiment \
        --traj-ids 0 1 2 \
        --steps 350 --action-horizon 16 \
        --output-dir /home/sharan/attn_overlay

    # grad-cam (high resolution; --denoise-steps 1 keeps activation memory down)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval/attention_overlay_eval.py \
        --model-path /home/sharan/G1_gear_cotrain/checkpoint-20000 \
        --dataset-path /home/sharan/cotrain/g1_gear_tcp_2.1_optical \
        --embodiment-tag new_embodiment \
        --traj-ids 0 --steps 350 --action-horizon 16 \
        --saliency gradcam --denoise-steps 1 --debug-reconstruct \
        --output-dir /home/sharan/gradcam_overlay

    Note: this script is tyro-based, so booleans are bare flags (``--debug-reconstruct``,
    ``--no-show-raw``) — unlike the draccus-based inference clients, which need
    ``--flag true``.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any
import warnings

import cv2
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
import numpy as np
import torch
import tyro


warnings.simplefilter("ignore", category=FutureWarning)


# --------------------------------------------------------------------------------------
# Attention capture
# --------------------------------------------------------------------------------------


class ImageAttentionCapture:
    """Hooks the DiT's image cross-attention blocks and accumulates per-image-token scores.

    Usage:
        cap = ImageAttentionCapture(policy, layers=None)
        cap.attach()
        cap.reset()
        policy.get_action(obs)          # may run several denoising steps; all are accumulated
        scores = cap.mean_scores()      # (S,) float32, one score per backbone token
        cap.detach()
    """

    def __init__(self, policy: Gr00tPolicy, layers: list[int] | None = None):
        self.policy = policy
        self.dit = self._find_dit(policy.model)
        self.image_layers = self._image_layer_indices(self.dit)
        if layers:
            unknown = sorted(set(layers) - set(self.image_layers))
            if unknown:
                raise ValueError(
                    f"--layers {unknown} are not image cross-attention layers. "
                    f"Available: {self.image_layers}"
                )
            self.layers = sorted(layers)
        else:
            self.layers = list(self.image_layers)

        self._handles: list[Any] = []
        self._sum: torch.Tensor | None = None
        self._n = 0
        # Captured from the backbone so tokens can be mapped back to cameras.
        self.image_mask: torch.Tensor | None = None
        self.image_grid_thw: torch.Tensor | None = None
        self._backbone_forward = None

    @staticmethod
    def _find_dit(model: torch.nn.Module) -> torch.nn.Module:
        for _, mod in model.named_modules():
            if type(mod).__name__ == "AlternateVLDiT":
                return mod
        raise RuntimeError("AlternateVLDiT not found; this script targets the GR00T N1.7 DiT.")

    @staticmethod
    def _image_layer_indices(dit: torch.nn.Module) -> list[int]:
        """Mirrors AlternateVLDiT.forward's block routing (dit.py:378-399)."""
        n = int(getattr(dit, "attend_text_every_n_blocks", 2))
        total = len(dit.transformer_blocks)
        return [i for i in range(total) if i % 2 == 0 and i % (2 * n) != 0]

    def attach(self) -> None:
        for idx in self.layers:
            attn = self.dit.transformer_blocks[idx].attn1
            self._handles.append(
                attn.register_forward_pre_hook(self._make_hook(attn), with_kwargs=True)
            )
        # Spy on the backbone to grab image_mask / image_grid_thw for this observation.
        bb = self.policy.model.backbone
        self._backbone_forward = bb.forward

        def spy(vl_input):
            # grid = vl_input.get("image_grid_thw") if isinstance(vl_input, dict) else None
            grid = vl_input["image_grid_thw"] if "image_grid_thw" in vl_input else None
            out = self._backbone_forward(vl_input)
            self.image_mask = out["image_mask"].detach()
            if grid is not None:
                self.image_grid_thw = grid.detach()
            return out

        bb.forward = spy

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._backbone_forward is not None:
            self.policy.model.backbone.forward = self._backbone_forward
            self._backbone_forward = None

    def reset(self) -> None:
        self._sum = None
        self._n = 0

    def _make_hook(self, attn: torch.nn.Module):
        def hook(module, args, kwargs):
            hidden = kwargs.get("encoder_hidden_states")
            if hidden is None:
                return  # self-attention call; nothing to do
            query_in = args[0] if args else kwargs.get("hidden_states")
            if query_in is None:
                return
            mask = kwargs.get("attention_mask")
            with torch.no_grad():
                self._accumulate(module, query_in, hidden, mask)

        return hook

    def _accumulate(
        self,
        attn: torch.nn.Module,
        query_in: torch.Tensor,
        encoder_hidden: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        heads = attn.heads
        q = attn.to_q(query_in)  # (B, T, inner)
        k = attn.to_k(encoder_hidden)  # (B, S, inner)
        b, t, inner = q.shape
        s = k.shape[1]
        head_dim = inner // heads
        q = q.view(b, t, heads, head_dim).transpose(1, 2).float()  # (B, H, T, d)
        k = k.view(b, s, heads, head_dim).transpose(1, 2).float()  # (B, H, S, d)

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)  # (B,H,T,S)
        if mask is not None:
            m = mask
            if m.dtype == torch.bool:
                while m.dim() < 4:
                    m = m.unsqueeze(1)
                logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
            else:
                while m.dim() < 4:
                    m = m.unsqueeze(1)
                logits = logits + m.to(logits.dtype)

        probs = torch.softmax(logits, dim=-1)
        # Average over heads and over the action-chunk queries -> one score per token.
        scores = probs.mean(dim=1).mean(dim=1)[0]  # (S,)
        scores = torch.nan_to_num(scores, nan=0.0)

        if self._sum is None:
            self._sum = scores.detach().clone()
        else:
            self._sum += scores.detach()
        self._n += 1

    def mean_scores(self) -> np.ndarray | None:
        if self._sum is None or self._n == 0:
            return None
        return (self._sum / self._n).cpu().numpy().astype(np.float32)

    # -- token -> camera mapping ---------------------------------------------------------

    def per_camera_maps(self, scores: np.ndarray, n_cameras: int) -> list[np.ndarray] | None:
        """Split token scores into one (h, w) grid per camera, in video modality-key order."""
        if scores is None or self.image_mask is None:
            return None
        idx = torch.where(self.image_mask[0])[0].cpu().numpy()
        if idx.size == 0:
            return None

        if self.image_grid_thw is not None:
            merge = self._spatial_merge_size()
            grid = self.image_grid_thw.cpu().numpy()
            shapes, counts = [], []
            for row in grid[:n_cameras]:
                _t, h, w = int(row[0]), int(row[1]), int(row[2])
                shapes.append((h // merge, w // merge))
                counts.append((h // merge) * (w // merge))
        else:  # fall back to equal split into squares
            per = idx.size // n_cameras
            side = int(round(math.sqrt(per)))
            shapes = [(side, side)] * n_cameras
            counts = [per] * n_cameras

        maps, cursor = [], 0
        for (h, w), c in zip(shapes, counts):
            take = idx[cursor : cursor + c]
            cursor += c
            if take.size != c:
                return None
            maps.append(scores[take].reshape(h, w))
        return maps

    def _spatial_merge_size(self) -> int:
        cfg = getattr(self.policy.model.backbone, "model", None)
        cfg = getattr(cfg, "config", None)
        vcfg = getattr(cfg, "vision_config", None)
        return int(getattr(vcfg, "spatial_merge_size", 2) or 2)


# --------------------------------------------------------------------------------------
# Saliency backends
# --------------------------------------------------------------------------------------


class SaliencyBackend:
    """Given one parsed observation, return per-camera 2-D maps plus the action chunk."""

    def compute(
        self, parsed_obs: dict[str, Any], n_cameras: int
    ) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class AttentionBackend(SaliencyBackend):
    """DiT action->image cross-attention. Forward-only; unchanged from the original script."""

    default_map_shape = (8, 8)

    def __init__(self, policy: Gr00tPolicy, layers: list[int] | None, seed: int = 0):
        self.policy = policy
        self.seed = seed
        self.capture = ImageAttentionCapture(policy, layers)
        self.capture.attach()
        self.image_layers = self.capture.image_layers
        self.layers = self.capture.layers

    def compute(self, parsed_obs, n_cameras):
        self.capture.reset()
        # The flow-matching loop starts from torch.randn (gr00t_n1d7.py:335). Without seeding,
        # predictions — and therefore the reported MSE — vary run to run by ~20% on short
        # trajectories, which is enough to swamp small comparisons.
        torch.manual_seed(self.seed)
        raw_action, _ = self.policy.get_action(parsed_obs)
        scores = self.capture.mean_scores()
        return self.capture.per_camera_maps(scores, n_cameras), parse_action_gr00t(raw_action)

    def close(self) -> None:
        self.capture.detach()


class GradCAMBackend(SaliencyBackend):
    """Gradient of the predicted action chunk w.r.t. ``pixel_values``.

    Measures *sensitivity*: how much the predicted action moves when a pixel moves. That is a
    different (and arguably more decision-relevant) question than attention, which only says
    where the model looked. It is still not proof of deliberate use — see the module docstring.

    Resolution is the vision-tower input grid (``grid_h*patch`` x ``grid_w*patch``, typically
    256x256), i.e. 32x the spatial detail of the 8x8 attention map.

    Requires bypassing three grad blockers (all in the upstream inference path):
      - ``torch.inference_mode()``           gr00t/policy/gr00t_policy.py:407
      - ``@torch.no_grad()`` get_action      gr00t/model/gr00t_n1d7/gr00t_n1d7.py:429
      - ``@torch.no_grad()`` ..._with_features                              :311
    The first is avoided by not calling ``policy.get_action`` at all; the other two by invoking
    the undecorated ``__wrapped__`` function. ``torch.enable_grad()`` alone does NOT work — the
    decorators re-disable grad inside the callee.
    """

    def __init__(
        self,
        policy: Gr00tPolicy,
        denoise_steps: int | None,
        seed: int,
        method: str,
        smooth: float,
        level: str = "patch",
    ):
        self.policy = policy
        self.seed = seed
        self.method = method
        self.smooth = smooth
        self.level = level
        self.action_head = policy.model.action_head

        fn = type(self.action_head).get_action_with_features
        if not hasattr(fn, "__wrapped__"):
            raise RuntimeError(
                "action_head.get_action_with_features has no __wrapped__ attribute, so its "
                "@torch.no_grad() decorator cannot be bypassed (PyTorch's decorator normally "
                "sets it via functools.wraps). Grad-CAM needs gradients through the denoising "
                "loop. Fix: inline the ~40-line Euler loop from gr00t_n1d7.py:381-421 here, or "
                "use --saliency attention."
            )
        self._get_action_nograd_free = fn.__wrapped__

        self._orig_steps = self.action_head.num_inference_timesteps
        if denoise_steps is not None:
            self.action_head.num_inference_timesteps = denoise_steps
            logging.info(
                f"Denoising steps {self._orig_steps} -> {denoise_steps} for the backward pass "
                "(lower = less activation memory)."
            )

        # Patch geometry, needed to scatter per-patch gradients back into an image.
        vcfg = policy.model.backbone.model.config.vision_config
        self.merge = int(getattr(vcfg, "spatial_merge_size", 2) or 2)
        img_proc = self._find_image_processor(policy)
        self.patch = int(getattr(img_proc, "patch_size", 16) or 16)
        self.temporal = int(getattr(img_proc, "temporal_patch_size", 2) or 2)
        self.last_debug: dict[str, Any] = {}

    @staticmethod
    def _find_image_processor(policy: Gr00tPolicy):
        for attr in ("image_processor", "processor"):
            o = getattr(policy.processor, attr, None)
            if o is not None:
                return getattr(o, "image_processor", o)
        return None

    def close(self) -> None:
        self.action_head.num_inference_timesteps = self._orig_steps

    # -- forward + backward --------------------------------------------------------------

    def compute(self, parsed_obs, n_cameras):
        policy = self.policy
        ah = self.action_head

        # Steps 1-3 of Gr00tPolicy._get_action: pure data prep, safe to reuse verbatim.
        unbatched = policy._unbatch_observation(parsed_obs)
        processed, states = [], []
        for o in unbatched:
            d = policy._to_vla_step_data(o)
            states.append(d.states)
            processed.append(
                policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": d}])
            )
        collated = _rec_to_dtype(policy.collate_fn(processed), dtype=torch.bfloat16)
        model_inputs = collated["inputs"] if "inputs" in collated else collated

        backbone_inputs, action_inputs = policy.model.prepare_input(model_inputs)
        pv = backbone_inputs["pixel_values"].detach().clone().requires_grad_(True)
        backbone_inputs["pixel_values"] = pv
        grid = backbone_inputs["image_grid_thw"].detach().cpu().numpy()

        # get_action_with_features seeds from torch.randn (gr00t_n1d7.py:335); without a fixed
        # seed the saliency would vary run to run for reasons unrelated to the input.
        torch.manual_seed(self.seed)

        with torch.enable_grad():
            backbone_out = policy.model.backbone(backbone_inputs)
            feats = ah._encode_features(backbone_out, action_inputs)
            out = self._get_action_nograd_free(
                ah,
                feats.backbone_features,
                feats.state_features,
                action_inputs.embodiment_id,
                backbone_out,
                action_inputs,
                None,
            )
            action_pred = out["action_pred"]
            # Target: the whole predicted chunk. Sum of squares, so every step and dim
            # contributes and signs cannot cancel.
            scalar = (action_pred.float() ** 2).sum()
            scalar.backward()

        if pv.grad is None or not torch.isfinite(pv.grad).all() or pv.grad.abs().sum() == 0:
            raise RuntimeError(
                "pixel_values gradient is missing, non-finite, or identically zero — the "
                "backward pass did not reach the vision input. Nothing meaningful can be "
                "rendered from this; not falling back silently."
            )

        grad = pv.grad.detach().float().cpu().numpy()
        pix = pv.detach().float().cpu().numpy()

        # Decode actions for the MSE HUD (steps 5-6 of _get_action).
        normalized = action_pred.detach().float().cpu().numpy()
        batched_states = {
            k: np.stack([s[k] for s in states], axis=0)
            for k in policy.modality_configs["state"].modality_keys
        }
        unnorm = policy.processor.decode_action(normalized, policy.embodiment_tag, batched_states)
        action_chunk = parse_action_gr00t(
            {k: v.astype(np.float32) for k, v in unnorm.items()}
        )

        maps = self._grad_to_maps(grad, pix, grid, n_cameras)

        del out, feats, backbone_out, pv, action_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return maps, action_chunk

    # -- pixel_values -> spatial maps ----------------------------------------------------

    def _grad_to_maps(self, grad, pix, grid, n_cameras) -> list[np.ndarray]:
        """Scatter per-patch gradient magnitude into one image-resolution map per camera.

        Qwen2/3VL emits patches in *merge-aware* order, not raster order: the processor
        permutes to (grid_t, gh//m, gw//m, m, m, C, temporal, p, p), so flat index n decodes
        as (t, h_block, w_block, mh, mw) with true position row = h_block*m + mh,
        col = w_block*m + mw. Verified empirically by reconstructing the image from
        pixel_values (see --debug-reconstruct).
        """
        maps, cursor = [], 0
        for cam in range(min(n_cameras, len(grid))):
            t, gh, gw = (int(x) for x in grid[cam])
            n = t * gh * gw
            g = grad[cursor : cursor + n]
            x = pix[cursor : cursor + n]
            cursor += n

            if self.method == "gradxinput":
                val = np.abs(g * x)
            else:  # "grad"
                val = np.abs(g)

            # (n, C*temporal*p*p) -> (n, p, p): collapse channel and temporal
            val = val.reshape(n, 3, self.temporal, self.patch, self.patch).sum(axis=(1, 2))

            # Classic Grad-CAM is computed at *feature-map* resolution, not pixel resolution —
            # that coarseness is what makes it readable. 'patch' reduces each 16x16 patch to a
            # single value (a gh x gw map, e.g. 16x16); 'pixel' keeps full resolution, which is
            # truer to vanilla gradient saliency and considerably noisier.
            if self.level == "patch":
                cell, val = 1, val.sum(axis=(1, 2))[:, None, None]
            else:
                cell = self.patch

            canvas = np.zeros((gh * cell, gw * cell), np.float32)
            m, hb_n, wb_n = self.merge, gh // self.merge, gw // self.merge
            for i in range(n):
                idx = i
                mw = idx % m; idx //= m
                mh = idx % m; idx //= m
                wb = idx % wb_n; idx //= wb_n
                hb = idx % hb_n
                row, col = hb * m + mh, wb * m + mw
                canvas[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = val[i]

            if self.smooth > 0:
                k = max(3, int(self.smooth * 4) | 1)  # odd kernel
                canvas = cv2.GaussianBlur(canvas, (k, k), self.smooth)
            maps.append(canvas)

        self.last_debug = {"grid": grid, "pix": pix}
        return maps

    def reconstruct_image(self, cam: int = 0) -> np.ndarray | None:
        """Rebuild the model's actual input image from pixel_values, to prove the ordering.

        A wrong de-permutation produces a visibly block-shuffled image; a correct one produces
        a coherent photo. Also shows the true resize/crop the model sees, which is *not* the
        raw dataset frame.
        """
        if not self.last_debug:
            return None
        grid, pix = self.last_debug["grid"], self.last_debug["pix"]
        cursor = sum(int(np.prod(grid[c])) for c in range(cam))
        t, gh, gw = (int(x) for x in grid[cam])
        n = t * gh * gw
        img = pix[cursor : cursor + n]
        canvas = np.zeros((gh * self.patch, gw * self.patch, 3), np.float32)
        m, hb_n, wb_n = self.merge, gh // self.merge, gw // self.merge
        for i in range(n):
            idx = i
            mw = idx % m; idx //= m
            mh = idx % m; idx //= m
            wb = idx % wb_n; idx //= wb_n
            hb = idx % hb_n
            row, col = hb * m + mh, wb * m + mw
            pt = img[i].reshape(3, self.temporal, self.patch, self.patch)[:, 0]
            canvas[
                row * self.patch : (row + 1) * self.patch,
                col * self.patch : (col + 1) * self.patch,
            ] = pt.transpose(1, 2, 0)
        proc = self._find_image_processor(self.policy)
        mean = np.array(getattr(proc, "image_mean", [0.5] * 3), np.float32)
        std = np.array(getattr(proc, "image_std", [0.5] * 3), np.float32)
        return np.clip(canvas * std + mean, 0, 1)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def render_overlay(
    frame_rgb: np.ndarray,
    attn_map: np.ndarray,
    alpha: float,
    colormap: int,
    normalize: str,
    vmax_ref: float | None,
    pct: float = 100.0,
) -> np.ndarray:
    """Blend an (h, w) attention grid over an RGB frame. Returns BGR for cv2 writers."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    m = attn_map.astype(np.float32)
    if normalize == "frame":
        lo = float(m.min())
        hi = float(np.percentile(m, pct)) if pct < 100 else float(m.max())
    else:  # "global" — comparable across frames of a trajectory
        lo, hi = 0.0, float(vmax_ref if vmax_ref else m.max())
    m = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
    m = np.clip(m, 0.0, 1.0)

    heat = cv2.resize((m * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(heat, colormap)
    return cv2.addWeighted(heat, alpha, frame_bgr, 1.0 - alpha, 0.0)


def label(img: np.ndarray, text: str, org: tuple[int, int] = (8, 22)) -> np.ndarray:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------------------
# Eval loop
# --------------------------------------------------------------------------------------


def evaluate_trajectory(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    backend: SaliencyBackend,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    steps: int,
    action_horizon: int,
    out_dir: Path,
    cfg: "ArgsConfig",
) -> dict[str, float]:
    traj = loader[traj_id]
    actual_steps = min(steps, len(traj))
    logging.info(f"[traj {traj_id}] {actual_steps} steps (trajectory length {len(traj)})")

    video_keys = loader.modality_configs["video"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys
    mc = deepcopy(loader.modality_configs)
    mc.pop("action")

    frames: list[np.ndarray] = []          # RGB frames per camera, per inference point
    attn_per_point: list[list[np.ndarray]] = []
    pred_actions: list[np.ndarray] = []
    infer_points: list[int] = []

    for step in range(0, actual_steps, action_horizon):
        dp = extract_step_data(traj, step, mc, embodiment_tag)
        obs: dict[str, Any] = {}
        for k, v in dp.states.items():
            obs[f"state.{k}"] = v
        images_this_step = {}
        for k, v in dp.images.items():
            arr = np.array(v)
            obs[f"video.{k}"] = arr
            images_this_step[k] = arr
        for lk in loader.modality_configs["language"].modality_keys:
            obs[lk] = dp.text
        parsed = parse_observation_gr00t(obs, loader.modality_configs)

        maps, action_chunk = backend.compute(parsed, len(video_keys))
        if maps is None:
            logging.warning(f"[traj {traj_id}] no saliency captured at step {step}")
            maps = [np.zeros((8, 8), np.float32) for _ in video_keys]

        if cfg.debug_reconstruct and step == 0 and isinstance(backend, GradCAMBackend):
            recon = backend.reconstruct_image(0)
            if recon is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"traj_{traj_id}_model_input_recon.png"
                cv2.imwrite(
                    str(path),
                    cv2.cvtColor((recon * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                )
                logging.info(
                    f"[traj {traj_id}] wrote {path} — this is the actual resized/cropped image "
                    "the model sees; a coherent photo confirms the patch de-permutation."
                )

        # (T, H, W, C) -> take the single timestep
        frames.append([images_this_step[k][0] for k in video_keys])
        attn_per_point.append(maps)
        infer_points.append(step)

        for j in range(action_horizon):
            pred_actions.append(
                np.concatenate(
                    [np.atleast_1d(np.atleast_1d(action_chunk[f"action.{k}"])[j]) for k in action_keys],
                    axis=0,
                )
            )

    # Ground truth for the error HUD
    def stack(cols: list[str]) -> np.ndarray:
        d = {c: np.vstack([a for a in traj[c]]) for c in cols}
        return np.concatenate([d[c] for c in cols], axis=-1)

    gt = stack([f"action.{k}" for k in action_keys])[:actual_steps]
    pred = np.asarray(pred_actions)[:actual_steps]
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(np.abs(gt - pred)))

    # Global colour scale so brightness is comparable across the trajectory.
    # Use a percentile, not the max: raw gradient magnitudes are heavy-tailed (p99 is only
    # ~15-25% of the max on this model), so min-max scaling crushes 99% of the image into the
    # bottom of the colormap and the overlay renders flat.
    if attn_per_point:
        allvals = np.concatenate([m.reshape(-1) for maps in attn_per_point for m in maps])
        vmax = (
            float(np.percentile(allvals, cfg.vmax_percentile))
            if cfg.vmax_percentile < 100
            else float(allvals.max())
        )
    else:
        vmax = 1.0

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    video_path = out_dir / f"traj_{traj_id}_attention.mp4"
    frame_dir = out_dir / f"traj_{traj_id}_frames"
    if cfg.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    for i, (cam_frames, maps) in enumerate(zip(frames, attn_per_point)):
        panels = []
        for key, frame_rgb, m in zip(video_keys, cam_frames, maps):
            panel = render_overlay(
                frame_rgb, m, cfg.alpha, getattr(cv2, f"COLORMAP_{cfg.colormap.upper()}"),
                cfg.normalize, vmax, cfg.vmax_percentile,
            )
            if cfg.show_raw:
                raw = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                panel = np.vstack([label(raw.copy(), f"{key} (raw)"), label(panel, f"{key} (attention)")])
            else:
                panel = label(panel, key)
            panels.append(panel)
        # Cameras may differ in resolution (e.g. the Dobot head is 430px tall, its wrist 480),
        # so match heights before stacking side by side.
        if len({p.shape[0] for p in panels}) > 1:
            target_h = max(p.shape[0] for p in panels)
            panels = [
                p if p.shape[0] == target_h
                else cv2.resize(p, (int(round(p.shape[1] * target_h / p.shape[0])), target_h),
                                interpolation=cv2.INTER_LINEAR)
                for p in panels
            ]
        canvas = np.hstack(panels)
        canvas = label(
            canvas,
            f"traj {traj_id}  step {infer_points[i]}  MSE {mse:.5f}",
            org=(8, canvas.shape[0] - 12),
        )

        if cfg.save_frames:
            cv2.imwrite(str(frame_dir / f"{infer_points[i]:05d}.png"), canvas)
        if writer is None:
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps,
                (canvas.shape[1], canvas.shape[0]),
            )
            if not writer.isOpened():
                logging.error(f"Could not open video writer for {video_path}; frames only.")
                writer = False  # sentinel: don't retry
        if writer:
            writer.write(canvas)

    if writer:
        writer.release()
        logging.info(f"[traj {traj_id}] wrote {video_path}")
    if cfg.save_frames:
        logging.info(f"[traj {traj_id}] wrote {len(frames)} frames to {frame_dir}")

    logging.info(f"[traj {traj_id}] MSE {mse:.6f}  MAE {mae:.6f}")
    return {"mse": mse, "mae": mae}


@dataclass
class ArgsConfig:
    """Render policy attention over camera frames during an open-loop replay."""

    model_path: str
    """Path to the model checkpoint directory."""

    dataset_path: str
    """Path to the LeRobot dataset to replay."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """Trajectory ids to render."""

    steps: int = 350
    """Max steps per trajectory (capped by trajectory length)."""

    action_horizon: int = 16
    """Steps consumed per inference — also the sampling interval for rendered frames."""

    output_dir: str = "/home/sharan/attn_overlay"
    """Where videos/frames are written."""

    saliency: str = "attention"
    """'attention' = DiT action->image cross-attention (forward-only, 8x8 per camera).
    'gradcam' = gradient of the predicted action chunk w.r.t. pixel_values (needs a backward
    pass, much heavier, but resolves at the vision input grid — typically 256x256)."""

    layers: list[int] | None = None
    """attention only: image cross-attention block indices to average. Default: all of them."""

    denoise_steps: int | None = None
    """gradcam only: override num_inference_timesteps for the differentiated forward.
    Backprop holds activations for every step, so 1 cuts memory ~4x vs the default 4 and
    barely changes the map's shape. Restored afterwards."""

    seed: int = 0
    """Seed for the flow-matching initial noise (gr00t_n1d7.py:335). Applies to BOTH modes —
    without it the predicted actions, and so the reported MSE, vary ~20% run to run."""

    cam_method: str = "grad"
    """gradcam only: 'grad' = |d action / d pixel|; 'gradxinput' = |grad * input|
    (suppresses response in near-black regions, often visually cleaner)."""

    cam_level: str = "patch"
    """gradcam only: 'patch' = one value per 16x16 vision patch (a 16x16 map) — coarse like
    classic Grad-CAM, which is what makes it readable. 'pixel' = full 256x256 gradient
    resolution, truer to vanilla gradient saliency but markedly noisier."""

    cam_smooth: float = 2.0
    """gradcam only: Gaussian sigma applied to the raw gradient map. Pixel gradients are
    speckly; 0 disables."""

    debug_reconstruct: bool = False
    """gradcam only: also write the image rebuilt from pixel_values at step 0. A coherent
    photo proves the merge-aware patch de-permutation is right; a block-shuffled mess proves
    it is wrong."""

    alpha: float = 0.5
    """Heatmap opacity, 0-1."""

    colormap: str = "jet"
    """OpenCV colormap name, e.g. jet, inferno, turbo, viridis."""

    normalize: str = "global"
    """'global' = one colour scale per trajectory (frames comparable);
    'frame' = rescale each frame independently (structure clearer, brightness not comparable)."""

    vmax_percentile: float = 99.0
    """Percentile used as the top of the colour scale instead of the raw max. Gradient
    magnitudes are heavy-tailed — on this model p99 is only ~15-25% of the max, so scaling to
    the max renders the overlay almost uniformly flat. 100 restores plain min-max."""

    fps: int = 4
    """Video frame rate. Frames are emitted once per inference, not per control step."""

    save_frames: bool = False
    """Also write each rendered frame as a PNG."""

    show_raw: bool = True
    """Stack the unmodified frame above the overlay for comparison."""

    device: str = "cuda"
    """Device to run the model on."""


def main(cfg: ArgsConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tag = EmbodimentTag.resolve(cfg.embodiment_tag)

    policy = Gr00tPolicy(embodiment_tag=tag, model_path=cfg.model_path, device=cfg.device)
    modality = policy.get_modality_config()
    logging.info(f"Video modality keys: {modality['video'].modality_keys}")

    dataset = LeRobotEpisodeLoader(
        dataset_path=cfg.dataset_path,
        modality_configs=modality,
        # video_backend="torchcodec",
        # video_backend_kwargs=None,
    )
    logging.info(f"Dataset length: {len(dataset)}")

    if cfg.saliency not in {"attention", "gradcam"}:
        raise ValueError(f"--saliency must be 'attention' or 'gradcam', got {cfg.saliency!r}")

    if cfg.saliency == "attention":
        backend: SaliencyBackend = AttentionBackend(policy, cfg.layers, seed=cfg.seed)
        logging.info(
            f"Saliency: attention | image cross-attention layers available "
            f"{backend.image_layers}; using {backend.layers}"
        )
    else:
        backend = GradCAMBackend(
            policy,
            denoise_steps=cfg.denoise_steps,
            seed=cfg.seed,
            method=cfg.cam_method,
            smooth=cfg.cam_smooth,
            level=cfg.cam_level,
        )
        logging.info(
            f"Saliency: gradcam | level={cfg.cam_level} method={cfg.cam_method} smooth={cfg.cam_smooth} "
            f"seed={cfg.seed} patch={backend.patch} merge={backend.merge}"
        )

    out_dir = Path(cfg.output_dir)
    results = {}
    try:
        for traj_id in cfg.traj_ids:
            if traj_id >= len(dataset):
                logging.warning(f"Trajectory {traj_id} out of range; skipping.")
                continue
            results[traj_id] = evaluate_trajectory(
                policy, dataset, backend, traj_id, tag, cfg.steps, cfg.action_horizon, out_dir, cfg
            )
    finally:
        backend.close()

    if results:
        logging.info("")
        logging.info(f"{'traj':>6}  {'MSE':>10}  {'MAE':>10}")
        for tid, r in results.items():
            logging.info(f"{tid:>6}  {r['mse']:>10.6f}  {r['mae']:>10.6f}")
        logging.info(
            f"{'mean':>6}  {np.mean([r['mse'] for r in results.values()]):>10.6f}  "
            f"{np.mean([r['mae'] for r in results.values()]):>10.6f}"
        )
    logging.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
