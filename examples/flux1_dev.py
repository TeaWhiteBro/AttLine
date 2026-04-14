"""Per-word attention visualization demo for FluxPipeline (Flux.1-dev).

Usage:
    python examples/flux1_dev.py
"""
import torch
from diffusers import FluxPipeline

from attline import attach

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

# One line — attach attention capture
attach(
    pipe,
    words=["cat", "holding", "sign", "hello", "world"],
    save_dir="./attn_out_flux1",
    heatmap_upscale=8,
)

# Normal pipeline usage, completely unchanged
image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("flux-dev.png")
# Attention heatmaps and overlays are already saved in ./attn_out_flux1/
