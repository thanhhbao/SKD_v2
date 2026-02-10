import argparse
import torch
from PIL import Image

from models import ModelWrapper
from utils import load_config
from preprocessor.registry import build_preprocessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to config yaml")
    ap.add_argument("--ckpt", required=True, help="path to .ckpt")
    ap.add_argument("--img", required=True, help="path to input image")
    ap.add_argument("--th", type=float, default=0.4, help="threshold for class=1")
    ap.add_argument("--device", default=None, help="cuda/cpu (override config)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # --- model config ---
    model_cfg = cfg["model"].copy()
    model_cfg["num_labels"] = cfg["data"]["num_classes"]
    model_cfg["_metrics_config"] = cfg.get("metrics", {})

    device = args.device if args.device is not None else model_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # --- load model ---
    model = ModelWrapper.load_from_checkpoint(args.ckpt, **model_cfg).to(device)
    model.eval()

    # --- build preprocessor ---
    data_cfg = cfg.get("data", {})
    preproc_name = data_cfg.get("preprocessor", "BasicImagePreprocessor")
    image_size = data_cfg.get("image_size", 224)  # nếu config không có, mặc định 224

    preproc = build_preprocessor(preproc_name, image_size=image_size)

    # --- load + preprocess image ---
    img = Image.open(args.img).convert("RGB")
    x = preproc(img, is_train=False)  # returns Tensor (C,H,W)
    if x.ndim == 3:
        x = x.unsqueeze(0)  # (1,C,H,W)
    x = x.to(device)

    # --- forward ---
    with torch.no_grad():
        logits = model(x)  # (1,2)
        prob_pos = torch.softmax(logits, dim=1)[0, 1].item()

    pred = int(prob_pos >= args.th)

    print(f"prob(class=1) = {prob_pos:.6f}")
    print(f"threshold = {args.th:.2f}")
    print(f"prediction = {pred}  ({'cancer' if pred==1 else 'non-cancer'})")


if __name__ == "__main__":
    main()
