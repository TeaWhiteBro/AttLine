"""Per-word attention visualization demo for QwenImagePipeline (Qwen-Image-2512).

Usage:
    python examples/qwen_image_2512.py
"""
import torch
from diffusers import DiffusionPipeline

from attline import attach

pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-2512",
    torch_dtype=torch.bfloat16,
).to("cuda")

# One line — attach attention capture
attach(
    pipe,
    words=["cat", "holding", "sign", "hello", "world"],
    save_dir="./attn_out_qwen",
    heatmap_upscale=8,
)

# Normal pipeline usage, completely unchanged
image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(0),
).images[0]

image.save("qwen_image_2512.png")
# Attention heatmaps and overlays are already saved in ./attn_out_qwen/
