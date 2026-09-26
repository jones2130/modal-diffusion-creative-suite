export interface ControlNetConfig {
  label: string;
  sdxl: string | null;
  flux: string | null;
  defaultScale: number;
}

export const CONTROLNET_REGISTRY: Record<string, ControlNetConfig> = {
  none: {
    label: "None",
    sdxl: null,
    flux: null,
    defaultScale: 0.0,
  },
  scribble: {
    label: "Scribble / Hand Sketch",
    sdxl: "xinsir/controlnet-scribble-sdxl-1.0",
    flux: "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
    defaultScale: 0.6,
  },
  canny: {
    label: "Canny / Hard Outlines",
    sdxl: "diffusers/controlnet-canny-sdxl-1.0",
    flux: "InstantX/FLUX.1-dev-Controlnet-Canny",
    defaultScale: 0.65,
  },
  depth: {
    label: "Depth / 3D Layout",
    sdxl: "diffusers/controlnet-depth-sdxl-1.0",
    flux: "InstantX/FLUX.1-dev-Controlnet-Depth",
    defaultScale: 0.7,
  },
  pose: {
    label: "Human Pose (OpenPose)",
    sdxl: "xinsir/controlnet-openpose-sdxl-1.0",
    flux: "XLabs-AI/flux-controlnet-openpose-diffusers",
    defaultScale: 0.75,
  },
  tile: {
    label: "Tile / Detailer",
    sdxl: "xinsir/controlnet-tile-sdxl-1.0",
    flux: "jasperai/Flux.1-dev-Controlnet-Upscaler",
    defaultScale: 0.5,
  },
};

export default CONTROLNET_REGISTRY;

