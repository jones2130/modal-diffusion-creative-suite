# Diffusers Studio — Frontend

An Astro-based web interface to generate images using the Modal deployment in `src/diffusers_backend_platform`.

## ✨ Features

- **Model Engine Switching**: Toggle between `SDXL 1.0` (A10G GPU, fast) and `FLUX.1-dev` (A100 GPU, high quality & photorealism).
- **Prompt Engineering**: Prompt, negative prompt, and randomized creative prompt suggestions.
- **Inference Tuning**: Sliders for inference steps, guidance scale, seed randomization, and batch image count.
- **ControlNet Integration**: Optional upload area for conditioning images and model repo ID (`diffusers/controlnet-canny-sdxl-1.0`, etc.).
- **Live Diagnostics**: Real-time timer tracking inference duration (including cold-start estimation).
- **Interactive Gallery**: Multi-image view, full-screen lightbox preview, and one-click full-resolution PNG downloads.
- **Persistent Endpoint URL**: Automatically saves your Modal endpoint URL in `localStorage`.

---

## 🚀 Getting Started

### 1. Start the Astro Development Server

From the `frontend/` directory:

```bash
cd frontend
npm run dev
```

Open `http://localhost:4321` in your browser.

---

### 2. Connect to your Modal Deployment

1. Deploy the backend from the project root:
   ```bash
   modal deploy -m src.diffusers_backend_platform.text_to_image.app
   ```
2. Copy the resulting endpoint URL printed in the terminal (e.g. `https://<username>--diffusion-create-suite-generate.modal.run`).
3. Paste the URL into the **Modal Endpoint** field in the top header bar of the Astro app.

---

### 3. Note on CORS (Cross-Origin Resource Sharing)

If you test the frontend on `http://localhost:4321` and make requests directly to your Modal endpoint, browsers may block the request unless CORS is enabled on the FastAPI endpoint in Modal.

To enable CORS in `src/diffusers_backend_platform/text_to_image/app.py`:

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
