# Attline: Visualize Attention at a Line in Diffusers

> Capture and visualize joint-attention maps from diffusion pipelines — per segment, per word, or per bounding box. Lightweight, no fork of `diffusers` needed.

**Languages:** English · [中文](README_ch.md)

Since `diffusers>=0.35` reworked its attention dispatch interface, existing attention-visualization tools (which hook into the old attention-processor layer) no longer work on the new transformer stacks. **Attline** adapts to the new interface and wraps the whole thing behind a single line of code — `attach(pipe, words=[...])` — so you can keep your normal pipeline usage completely unchanged and still get per-word heatmaps out of any supported pipeline, with minimal intrusion into your code.

![Attline teaser: per-word attention across FLUX.2-klein, FLUX.1-dev, and Qwen-Image-2512](assets/comparison.png)

## Currently supported pipelines

| Pipeline class        | Model family                                                 | Example                                                     |
| --------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| `Flux2KleinPipeline`  | `black-forest-labs/FLUX.2-klein-9B`                          | [`examples/flux2_klein.py`](examples/flux2_klein.py)        |
| `FluxPipeline`        | `black-forest-labs/FLUX.1-dev`, `FLUX.1-schnell`             | [`examples/flux1_dev.py`](examples/flux1_dev.py)            |
| `QwenImagePipeline`   | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512`                    | [`examples/qwen_image_2512.py`](examples/qwen_image_2512.py) |

## Install

```bash
pip install -e .
# or, for the optional diffusers dep pin:
pip install -e ".[diffusers]"
```

## Quickstart

The minimum to enable attention capture is a single line:

```python
import torch
from diffusers import Flux2KleinPipeline
from attline import attach

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
).to("cuda")

attach(pipe, words=["cat", "hello", "world"])

image = pipe(prompt="A cat holding a sign that says hello world").images[0]
image.save("out.png")
# Heatmaps saved in ./attn_out/
```

Call `detach(pipe)` to restore the original pipeline.

## Usage — `attach(pipe, ...)`

**Minimum required:**

| Argument | Description |
| --- | --- |
| `pipe` | A supported diffusion pipeline instance. |
| `words=[...]` **or** `attention_pairs=[...]` | At least one target for which to save a heatmap. |

**Optional:**

| Argument | Default | Description |
| --- | --- | --- |
| `words` | `None` | List of words/phrases from the prompt. Each produces a spatial heatmap + overlay. |
| `attention_pairs` | `None` | Low-level `(query_selector, key_selector)` tuples. |
| `save_dir` | `"./attn_out"` | Output directory for heatmaps and overlays (created if missing). |
| `heatmap_upscale` | `8` | Integer upscale applied to the saved heatmap PNG for readability (purely cosmetic; does not change capture). |
| `fallback_to_sdpa` | `True` | On CUDA OOM, silently fall back to plain SDPA for that call and increment `skipped_capture_calls`. |
| `capture_chunk_size` | `256` | Query-chunk size for streaming softmax. Lower → less VRAM, slower. |
| `pipeline_type` | `None` | Explicit adapter name. Auto-detected from the pipeline class when `None`. |

## Acknowledgement

This project is inspired by [wooyeolbaek/attention-map-diffusers](https://github.com/wooyeolbaek/attention-map-diffusers), which pioneered per-token joint-attention visualization for diffusion pipelines.

## Citation

If you find this tool useful, please consider leaving a ⭐ on GitHub.

## License

MIT. See [LICENSE](LICENSE) for details.
