"""
Generate images from text using Stable Diffusion 3.5 Large Turbo.
"""

import io
import base64
import random
from typing import Optional, List
from PIL import Image
from pydantic import BaseModel
import modal

app = modal.App(name="diffusion_create_suite")

# Configure the container image with necessary dependencies

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
CACHE_DIR = "/cache"

cache_volume = modal.Volume.from_name("flux-model-cache", create_if_missing=True)

class GenerateRequest(BaseModel):
    model_type: str = "sdxl"
    prompt: str
    negative_prompt: Optional[str] = ""
    controlnet_id: Optional[str] = ""
    control_image_b64: Optional[str] = ""
    controlnet_scale: float = 0.6
    steps: int = 25
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: int = None
    max_sequence_length: int = 512
    
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]",
        "huggingface-hub==0.36.0",
        "hf_transfer",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
        "pydantic"
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

@app.cls(
    image=base_image,
    gpu="A10G",
    timeout=600,
    volumes={CACHE_DIR: cache_volume},
    scaledown_window=100
)
class SDXLRunner:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import StableDiffusionXLPipeline 

        """Load the model into GPU memory on container startup."""
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR
        ).to("cuda")
        self.loaded_controlnet = {}
        cache_volume.commit()

    def _get_controlnet(self, repo_id: str):
        import torch
        from diffusers import ControlNetModel
        if repo_id not in self.loaded_controlnet:
            cnet = ControlNetModel.from_pretrained(
                repo_id,
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR
            ).to("cuda")
            self.loaded_controlnet[repo_id] = cnet
            cache_volume.commit()
        return self.loaded_controlnet[repo_id]

    @modal.method()
    def run(self, req: GenerateRequest) -> List[Image.Image]:
        import torch
        from diffusers import StableDiffusionXLControlNetPipeline

        """Generate images from a text prompt."""

        if req.controlnet_id and req.control_image_b64:
            cnet = self._get_controlnet(req.controlnet_id)
            cnet_pipe = StableDiffusionXLControlNetPipeline(
                **self.pipe.components, controlnet=cnet
            )
            raw = base64.b64decode(req.control_image_b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB").resize((1024, 1024))
            
            return cnet_pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                image=img,
                controlnet_conditioning_scale=req.controlnet_scale,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                num_images_per_prompt=req.num_images,
            ).images
        
        seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
        print(f"Generating image with seed: {seed}")
        
        torch.manual_seed(seed)

        return self.pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            num_images_per_prompt=req.num_images,
        ).images

@app.cls(
    image=base_image,
    gpu="A100",
    timeout=600,
    volumes={CACHE_DIR: cache_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
    scaledown_window=300
)
class FluxRunner:
    @modal.enter()
    def load_model(self):
        import torch
        import os
        from diffusers import FluxPipeline 

        """Load the model into GPU memory on container startup."""
        self.pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN")
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.loaded_controlnet = {}
        cache_volume.commit()

    def _get_controlnet(self, repo_id: str):
        import torch
        from diffusers import FluxControlNetModel
        if repo_id not in self.loaded_controlnet:
            cnet = FluxControlNetModel.from_pretrained(
                repo_id,
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR
            ).to("cuda")
            self.loaded_controlnet[repo_id] = cnet
            cache_volume.commit()
        return self.loaded_controlnet[repo_id]

    @modal.method()
    def run(self, req: GenerateRequest) -> List[Image.Image]:
        import torch
        from diffusers import FluxControlNetPipeline

        """Generate images from a text prompt."""

        if req.controlnet_id and req.control_image_b64:
            cnet = self._get_controlnet(req.controlnet_id)
            cnet_pipe = FluxControlNetPipeline(
                **self.pipe.components, controlnet=cnet
            )
            cnet_pipe.enable_model_cpu_offload()

            raw = base64.b64decode(req.control_image_b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB").resize((1024, 1024))
            
            return cnet_pipe(
                prompt=req.prompt,
                control_image=img,
                controlnet_conditioning_scale=req.controlnet_scale,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                num_images_per_prompt=req.num_images,
                height=1024,
                width=1024
            ).images
        
        seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
        print(f"Generating image with seed: {seed}")
        
        torch.manual_seed(seed)

        return self.pipe(
            prompt=req.prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            num_images_per_prompt=req.num_images,
            height=1024,
            width=1024
        ).images

@app.function(image=base_image)
@modal.fastapi_endpoint(method="POST", docs=True)
def generate(req: GenerateRequest):
    """
    Web endpoint for generating images via HTTP.
    Visit the /docs endpoint for interactive API documentation.
    """
    if req.model_type.lower() == "flux":
        output_images = FluxRunner().run.remote(req)
    else:
        output_images = SDXLRunner().run.remote(req)
    b64_image_list = []
    for img in output_images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_image_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
                              
    return {"images": b64_image_list}
