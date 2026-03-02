import argparse
import os
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T


def list_image_files(dir_path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    return sorted(
        [
            os.path.basename(p)
            for p in glob(os.path.join(dir_path, "*"))
            if os.path.splitext(p)[1].lower() in exts
        ]
    )


def load_image(path: str, image_size: int = 256) -> torch.Tensor:
    """
    Load image as float tensor in [0, 1], shape (3, H, W).
    """
    img = Image.open(path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ]
    )
    return transform(img)


def compute_ssim_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Very simple SSIM implementation for RGB images.
    x, y: (3, H, W), values in [0, 1]
    """
    # constants from original SSIM paper
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # convert to grayscale for simplicity
    rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(3, 1, 1)
    x = (x * rgb_weights).sum(dim=0, keepdim=True)
    y = (y * rgb_weights).sum(dim=0, keepdim=True)

    mu_x = torch.nn.functional.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
    mu_y = torch.nn.functional.avg_pool2d(y, kernel_size=11, stride=1, padding=5)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.nn.functional.avg_pool2d(x * x, 11, 1, 5) - mu_x2
    sigma_y2 = torch.nn.functional.avg_pool2d(y * y, 11, 1, 5) - mu_y2
    sigma_xy = torch.nn.functional.avg_pool2d(x * y, 11, 1, 5) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean().item()


def compute_lpips_model(device: torch.device):
    """
    Build LPIPS model if lpips is installed.
    """
    try:
        import lpips  # type: ignore
    except ImportError:
        raise ImportError(
            "lpips is not installed. Install it with:\n"
            "  pip install lpips\n"
        )

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()
    return loss_fn


def compute_fid_model(device: torch.device):
    """
    Build FID metric if torchmetrics is installed.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore
    except ImportError:
        raise ImportError(
            "torchmetrics is not installed. Install it with:\n"
            "  pip install torchmetrics\n"
        )

    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.eval()
    return fid


def evaluate_metrics(
    real_dir: str,
    gen_dir: str,
    image_size: int = 256,
    device: str = "cuda",
) -> Tuple[float, float, float]:
    """
    Compute FID, SSIM, LPIPS between images in two folders.

    - real_dir: reference / ground-truth images
    - gen_dir: generated images
    - Images are paired by **filename**; only intersection of filenames is used.
    """
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    real_files = list_image_files(real_dir)
    gen_files = list_image_files(gen_dir)

    common_files = sorted(set(real_files).intersection(set(gen_files)))
    if not common_files:
        raise ValueError(f"No common image filenames between {real_dir} and {gen_dir}.")

    print(f"Found {len(common_files)} paired images.")

    # Build models
    fid_metric = compute_fid_model(device_t)
    lpips_model = compute_lpips_model(device_t)

    ssim_values: List[float] = []
    lpips_values: List[float] = []

    with torch.no_grad():
        for name in common_files:
            real_path = os.path.join(real_dir, name)
            gen_path = os.path.join(gen_dir, name)

            real_img = load_image(real_path, image_size=image_size)
            gen_img = load_image(gen_path, image_size=image_size)

            # FID expects batches in (B, 3, H, W)
            fid_metric.update(real_img.unsqueeze(0).to(device_t), real=True)
            fid_metric.update(gen_img.unsqueeze(0).to(device_t), real=False)

            # SSIM on [0, 1]
            ssim_values.append(compute_ssim_torch(real_img.to(device_t), gen_img.to(device_t)))

            # LPIPS expects [-1, 1]
            real_lp = real_img.to(device_t) * 2 - 1
            gen_lp = gen_img.to(device_t) * 2 - 1
            lp = lpips_model(real_lp.unsqueeze(0), gen_lp.unsqueeze(0)).item()
            lpips_values.append(lp)

    fid_score = float(fid_metric.compute().cpu().item())
    ssim_mean = float(np.mean(ssim_values))
    lpips_mean = float(np.mean(lpips_values))

    return fid_score, ssim_mean, lpips_mean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID, SSIM, LPIPS between two folders of images."
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        required=True,
        help="Directory with reference / ground-truth images.",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="Directory with generated images to evaluate.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Images will be resized to (image_size, image_size) before metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to run metrics on, e.g. "cuda" or "cpu".',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fid, ssim_val, lpips_val = evaluate_metrics(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        image_size=args.image_size,
        device=args.device,
    )

    print("==== Metrics ====")
    print(f"FID   : {fid:.4f}  (lower is better)")
    print(f"SSIM  : {ssim_val:.4f}  (higher is better)")
    print(f"LPIPS : {lpips_val:.4f}  (lower is better)")

