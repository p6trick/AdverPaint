# AdverPaint: Inpainting-Based Adversarial Attack on Face Recognition Systems

Official implementation of **AdverPaint: Inpainting-Based Adversarial Attack on Face Recognition Systems**.

## Requirements

- Python 3.8 or newer
- PyTorch and torchvision (CUDA is recommended, but CPU inference is supported and slower)

Install the Python dependencies. Install the PyTorch build appropriate for the machine first if CUDA is needed; see [PyTorch's install selector](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt
```

## AdverPaint checkpoint

Download `G0450000.pt` from **[Google Drive — model, mask, and example image](https://drive.google.com/drive/folders/1z4pT3xSWvsfmtiSl4D61rSos2EJIzu8C?usp=sharing)** and place it at `checkpoints/G0450000.pt`.


## Input format

- `input/`: input images (`jpg`, `jpeg`, `png`, `bmp`, or `webp`)
- `mask.png`: a grayscale mask.

Images and the mask are resized to 512 × 512 by default. The output is therefore 512 × 512.

The Google Drive folder also includes `mask.png` and `example.jpg`, so the following directory layout can be used for a quick test:

```
AdverPaint/
├── checkpoints/G0100000.pt
├── input/example.jpg
└── mask.png
```

## Run inference

From this directory:

```bash
python inference.py \
  --checkpoint checkpoints/G0100000.pt \
  --input-dir input \
  --mask mask.png \
  --output-dir output
```

Outputs are named `<input-name>_inpainted.png`. To also save the masked input and the model's raw prediction:

```bash
python inference.py \
  --checkpoint checkpoints/G0100000.pt \
  --input-dir input \
  --mask mask.png \
  --output-dir output \
  --save-intermediates
```

The program automatically uses CUDA when available. Use `--device cpu` to force CPU, or `--device cuda:0` to select a particular GPU.
