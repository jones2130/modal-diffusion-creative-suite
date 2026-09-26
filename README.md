# Diffusers Backend Platform & Studio

A serverless generative AI platform pairing high-performance diffusion models (**SDXL 1.0** and **FLUX.1-dev**) deployed on [Modal](https://modal.com) with a modern [Astro](https://astro.build) frontend client.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   Astro Frontend (./frontend)                    │
│  - Prompt & Parameter Controls (Steps, Guidance, Seed)           │
│  - ControlNet Preset Switcher (SDXL <-> FLUX Auto-Mapping)       │
│  - Image Dropzone & Base64 Encoder                               │
│  - Live Inference Timer & Full-Resolution Lightbox/Download      │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ HTTP POST (JSON)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Modal Backend (./src/...)                       │
│  App: "diffusion_create_suite"                                   │
│                                                                  │
│  ┌─────────────────────────┐         ┌─────────────────────────┐ │
│  │       SDXLRunner        │         │       FluxRunner        │ │
│  │  - StabilityAI SDXL 1.0 │         │  - Black Forest FLUX.1  │ │
│  │  - A10G GPU             │         │  - A100 GPU             │ │
│  │  - ControlNet SDXL      │         │  - CPU offload + tiling │ │
│  └─────────────────────────┘         └─────────────────────────┘ │
│                                                                  │
│  - Shared Modal Cache Volume: "flux-model-cache"                 │
│  - Secure Token Injection: modal.Secret.from_name("hf-secret")   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites & Requirements

### 1. Python Environment
- **Python**: `>=3.12`
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### 2. Node.js Environment
- **Node.js**: `>=22.12.0`
- **npm**: `>=10.0.0`

### 3. Modal Account & CLI
- Sign up at [modal.com](https://modal.com).
- Install the Modal client and authenticate your machine:
  ```bash
  uv pip install modal
  modal setup
  ```
  *(This will open a browser window to generate and securely store your Modal credentials locally in `~/.modal.toml`)*.

### 4. Hugging Face Access (for FLUX.1-dev)
- FLUX.1-dev is a gated model requiring Hugging Face agreement acceptance.
- Accept the model terms on Hugging Face: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- Create a secret in your Modal workspace named `hf-secret`:
  ```bash
  modal secret create hf-secret HF_TOKEN="hf_your_actual_token_here"
  ```
  > **Note**: Never commit your `HF_TOKEN` or Modal API keys to git. Modal manages this securely in the cloud.

---

## 🚀 Backend Setup & Deployment

### 1. Install Local Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd diffusers-backend-platform

# Install dependencies using uv
uv sync
```

### 2. Deploy to Modal

Deploy the inference application to your Modal cloud workspace:

```bash
modal deploy src/diffusers_backend_platform/text_to_image/app.py
```

When deployment finishes, Modal prints the live HTTPS endpoint URL for the `generate` function:
```text
✓ Created objects:
├── App: diffusion_create_suite
└── Function generate => https://<your-modal-workspace>--diffusion-create-suite-generate.modal.run
```
Save this URL; you will paste it into the frontend.

### 3. Optional: Adding CORS to Modal Endpoint

If calling the Modal endpoint directly from a web browser (`http://localhost:4321`), ensure your FastAPI endpoint allows Cross-Origin Resource Sharing (CORS).

In `src/diffusers_backend_platform/text_to_image/app.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.post("/")
def generate(req: GenerateRequest):
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

@app.function(image=base_image)
@modal.asgi_app()
def fastapi_app():
    return web_app
```

---

## 🎨 Frontend Setup & Execution

The frontend is an Astro application located in the `./frontend` directory.

### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 2. Start the Development Server

```bash
npm run dev
```

Open your browser at:
```text
http://localhost:4321/
```

### 3. Connect to the Modal Backend

1. In the top header bar, enter your Modal HTTPS URL:
   `https://<your-workspace>--diffusion-create-suite-generate.modal.run`
2. The URL is automatically saved to your browser's `localStorage` for future sessions.
3. Select an engine (**SDXL 1.0** or **FLUX.1-dev**), enter your prompt, and click **⚡ Generate Image**.

---

## 🧩 ControlNet Presets & Model Mappings

The frontend includes quick-selection presets defined in [`frontend/src/lib/controlnets.ts`](file:///Users/jones220/Work/experiments/ai-testing/modal/diffusers-backend-platform/frontend/src/lib/controlnets.ts). Selecting a preset automatically configures the appropriate Hugging Face model repository and optimal conditioning scale for whichever engine is selected:

| Preset | Description | SDXL Model ID | FLUX Model ID | Default Scale |
| :--- | :--- | :--- | :--- | :--- |
| **None** | Pure text-to-image | `null` | `null` | `0.00` |
| **Scribble** | Hand sketches / line art | `xinsir/controlnet-scribble-sdxl-1.0` | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` | `0.60` |
| **Canny** | Edge detection / outlines | `diffusers/controlnet-canny-sdxl-1.0` | `InstantX/FLUX.1-dev-Controlnet-Canny` | `0.65` |
| **Depth** | 3D scene geometry | `diffusers/controlnet-depth-sdxl-1.0` | `InstantX/FLUX.1-dev-Controlnet-Depth` | `0.70` |
| **Pose** | Human OpenPose keypoints | `xinsir/controlnet-openpose-sdxl-1.0` | `XLabs-AI/flux-controlnet-openpose-diffusers` | `0.75` |
| **Tile** | Detail enhancement & upscaling | `xinsir/controlnet-tile-sdxl-1.0` | `jasperai/Flux.1-dev-Controlnet-Upscaler` | `0.50` |

*You can also manually override the input field with any custom HuggingFace ControlNet repository at any time.*

---

## 🔒 Security & Privacy Audit Checklist

Before pushing this repository to a public or shared remote:

- [x] **No hardcoded secrets or tokens**: All Hugging Face and API tokens are dynamically resolved via Modal Cloud Secrets (`modal.Secret.from_name("hf-secret")`).
- [x] **No Modal credentials tracked**: Modal credentials reside exclusively in `~/.modal.toml` (outside the repository).
- [x] **Gitignore protection**: `.gitignore` blocks `.env*`, `.modal.toml`, `node_modules/`, `.astro/`, `.venv/`, and generated images (`*.png`, `bootstrap-results-*`).
- [ ] **Author information in `pyproject.toml`**: Check lines 6-8 of [`pyproject.toml`](file:///Users/jones220/Work/experiments/ai-testing/modal/diffusers-backend-platform/pyproject.toml) if you wish to anonymize your personal name or institutional email before publishing.
