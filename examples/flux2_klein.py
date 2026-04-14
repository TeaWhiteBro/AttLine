"""Per-word attention visualization demo for Flux2KleinPipeline.

Usage:
    python examples/flux2_klein.py
"""
import torch
from diffusers import Flux2KleinPipeline

from attline import attach

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    torch_dtype=torch.bfloat16,
).to("cuda")

# One line — attach attention capture
attach(
    pipe,
    words=["cat", "holding", "sign", "hello", "world"],
    save_dir="./attn_out",
    heatmap_upscale=8,
)

# Normal pipeline usage, completely unchanged
image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cuda").manual_seed(0),
).images[0]

image.save("flux-klein.png")
# Attention heatmaps and overlays are already saved in ./attn_out/
