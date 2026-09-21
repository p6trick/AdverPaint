#!/usr/bin/env python3
"""Run AdverPaint image inpainting on every image in a directory."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor

from model import InpaintGenerator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="AdverPaint inference")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to G0100000.pt")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory of input images")
    parser.add_argument("--mask", required=True, type=Path, help="White=area to inpaint mask image")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output images")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu (default: auto)")
    parser.add_argument("--size", type=int, default=512, help="Square inference size (default: 512)")
    parser.add_argument("--save-intermediates", action="store_true", help="Also save masked input and raw prediction")
    return parser.parse_args()


def get_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    return device


def to_image(tensor):
    tensor = tensor.detach().cpu().clamp(-1, 1)
    array = ((tensor + 1) / 2 * 255).permute(1, 2, 0).byte().numpy()
    return Image.fromarray(array)


def load_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = InpaintGenerator().to(device)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def main():
    args = parse_args()
    if args.size < 8 or args.size % 4:
        raise ValueError("--size must be a multiple of 4 and at least 8.")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
    if not args.mask.is_file():
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    image_paths = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.input_dir}")

    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask = to_tensor(resize(Image.open(args.mask).convert("L"), [args.size, args.size], InterpolationMode.NEAREST)).unsqueeze(0).to(device)
    mask = (mask > 0.5).float()

    print(f"Running {len(image_paths)} image(s) on {device}; outputs: {args.output_dir}")
    with torch.inference_mode():
        for image_path in image_paths:
            source = Image.open(image_path).convert("RGB")
            image = to_tensor(resize(source, [args.size, args.size], InterpolationMode.BILINEAR))
            image = (image * 2 - 1).unsqueeze(0).to(device)
            masked = image * (1 - mask) + mask
            prediction = model(masked, mask)
            composite = image * (1 - mask) + prediction * mask
            stem = image_path.stem
            to_image(composite[0]).save(args.output_dir / f"{stem}_inpainted.png")
            if args.save_intermediates:
                to_image(masked[0]).save(args.output_dir / f"{stem}_masked.png")
                to_image(prediction[0]).save(args.output_dir / f"{stem}_prediction.png")
            print(f"Saved {stem}_inpainted.png")


if __name__ == "__main__":
    main()
