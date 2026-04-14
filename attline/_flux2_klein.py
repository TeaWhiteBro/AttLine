"""Flux2KleinPipeline adapter for attention capture."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .capture import CaptureState, get_capture_state
from .layouts import build_flux2_klein_layout, compute_token_hw, normalize_to_multiple
from .selectors import compute_text_meta, has_text_phrase_selectors
from .patch import PipelineAdapter, register_adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_to_target_area_if_needed(
    img: Image.Image, max_area: int = 1024 * 1024
) -> Image.Image:
    w, h = img.size
    area = w * h
    if area <= max_area:
        return img
    scale = (max_area / float(area)) ** 0.5
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Patched __call__ for Flux2KleinPipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def flux2klein_call_with_layout(
    self,
    image: Optional[Union[List[Image.Image], Image.Image]] = None,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: Optional[float] = 4.0,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[Union[str, List[str]]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    text_encoder_out_layers: Tuple[int, ...] = (9, 18, 27),
):
    from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        XLA_AVAILABLE,
        compute_empirical_mu,
        retrieve_timesteps,
    )

    state = get_capture_state()

    self.check_inputs(
        prompt=prompt,
        height=height,
        width=width,
        prompt_embeds=prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        guidance_scale=guidance_scale,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
    )

    if (
        state is not None
        and prompt is not None
        and has_text_phrase_selectors(state.attention_pairs)
    ):
        text_meta = compute_text_meta(
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )
        state.set_text_meta(text_meta)

    if self.do_classifier_free_guidance:
        negative_prompt = ""
        if prompt is not None and isinstance(prompt, list):
            negative_prompt = [negative_prompt] * len(prompt)

        negative_prompt_embeds, negative_text_ids = self.encode_prompt(
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

    if image is not None and not isinstance(image, list):
        image = [image]

    condition_images = None
    condition_overlay_images: List[Image.Image] = []
    condition_token_hws: List[Tuple[int, int]] = []
    if image is not None:
        for img in image:
            self.image_processor.check_image_input(img)

        condition_images = []
        for img in image:
            img = _resize_to_target_area_if_needed(img)
            image_width, image_height = img.size
            multiple_of = self.vae_scale_factor * 2
            image_width, image_height = normalize_to_multiple(image_width, image_height, multiple_of)
            overlay_base = img.convert("RGB").resize((image_width, image_height), Image.LANCZOS)
            processed = self.image_processor.preprocess(
                img,
                height=image_height,
                width=image_width,
                resize_mode="crop",
            )
            condition_images.append(processed)
            condition_overlay_images.append(overlay_base)

            if state is not None:
                condition_token_hws.append(compute_token_hw(image_height, image_width, self.vae_scale_factor * 2))

            height = height or image_height
            width = width or image_width

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_ids = self.prepare_latents(
        batch_size=batch_size * num_images_per_prompt,
        num_latents_channels=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=latents,
    )

    image_latents = None
    image_latent_ids = None
    if condition_images is not None:
        image_latents, image_latent_ids = self.prepare_image_latents(
            images=condition_images,
            batch_size=batch_size * num_images_per_prompt,
            generator=generator,
            device=device,
            dtype=self.vae.dtype,
        )

    if state is not None:
        noise_hw = compute_token_hw(height, width, self.vae_scale_factor * 2)
        layout = build_flux2_klein_layout(
            text_count=prompt_embeds.shape[1],
            noise_hw=noise_hw,
            image_hws=condition_token_hws,
        )
        state.set_layout(layout)
        state.set_condition_images(condition_overlay_images)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
        sigmas = None

    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)

    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )

    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    self.scheduler.set_begin_index(0)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            latent_model_input = latents.to(self.transformer.dtype)
            latent_image_ids = latent_ids

            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]

            noise_pred = noise_pred[:, : latents.size(1) :]

            if self.do_classifier_free_guidance:
                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                import torch_xla.core.xla_model as xm
                xm.mark_step()

    self._current_timestep = None

    latents = self._unpack_latents_with_ids(latents, latent_ids)
    latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(
        self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)

    latents = latents * latents_bn_std + latents_bn_mean
    latents = self._unpatchify_latents(latents)

    if output_type == "latent":
        image_out = latents
    else:
        image_out = self.vae.decode(latents, return_dict=False)[0]
        image_out = self.image_processor.postprocess(image_out, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image_out,)

    return Flux2PipelineOutput(images=image_out)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

register_adapter(
    PipelineAdapter(
        name="Flux2KleinPipeline",
        pipeline_class_names=("Flux2KleinPipeline",),
        patched_call=flux2klein_call_with_layout,
    )
)
