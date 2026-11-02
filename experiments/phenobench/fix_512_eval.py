import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from test_highres_512 import (
    BASE_DIR, HIGH_RES, BATCH_SIZE, SEED,
    set_seed, PhenoBenchDataset, SimpleUNet,
    dice_loss, dilate_prob, erode_prob, evaluate_512
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"


def topo_fp_suppression_loss_512(logits, imgs, labels):
    """512x512 전용: 과민 반응을 유발한 loss_recall을 제거하고,
    오탐(FP) 발생 픽셀 위주로만 정규화하여 정밀 타격하는 위상 손실"""
    probs = F.softmax(logits, dim=1)
    p_crop, p_weed = probs[:, 1], probs[:, 2]

    # 조명 변화에 강건한 정규화 ExG (Normalized Excess Green)
    r, g, b = imgs[:, 0], imgs[:, 1], imgs[:, 2]
    exg_norm = (2.0 * g - r - b) / (r + g + b + 1e-5)
    non_weed = (labels != 2).float()

    # 1) 맨땅(Soil) 오탐 억제: 비초록 영역에서 튀는 p_weed를 집중 억제
    soil_gate = F.relu(0.05 - exg_norm) / 0.05
    loss_soil_fp = (soil_gate * (p_weed ** 2) * non_weed).sum() / (non_weed.sum() + 1e-5)

    # 2) 작물(Crop) 경계·내부 오탐 억제: 512 해상도에 맞춰 커널을 k=9로 확대
    gt_crop = (labels == 1).float().unsqueeze(1)
    dil_crop = dilate_prob(gt_crop, k=9).squeeze(1)
    closed_crop = erode_prob(dilate_prob(p_crop.unsqueeze(1), k=9), k=9).squeeze(1).detach()
    crop_zone = torch.clamp(0.6 * dil_crop + 0.4 * closed_crop, 0.0, 1.0) * non_weed
    loss_crop_fp = (crop_zone * p_weed).sum() / (crop_zone.sum() + 1e-5)

    return 2.0 * loss_soil_fp + 2.0 * loss_crop_fp


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PhenoBenchDataset(BASE_DIR, "train", img_size=HIGH_RES)
    val_ds = PhenoBenchDataset(BASE_DIR, "val", img_size=HIGH_RES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 1. 기존 Baseline 가중치 로드 및 결과 확인
    base_model = SimpleUNet().to(device)
    base_ckpt = torch.load(WEIGHTS_DIR / "ckpt_512_base.pth", map_location=device)
    base_model.load_state_dict(base_ckpt["model"])
    iou_b, fp_b = evaluate_512("1. 기존 512x512 Baseline (로드됨)", base_model, val_loader, device)

    # 2. Topo 모델: Baseline 가중치에서 시작해 낮은 학습률(1e-4)로 2에폭 정밀 위상 교정
    set_seed(SEED)
    topo_model = SimpleUNet().to(device)
    topo_model.load_state_dict(base_ckpt["model"])
    opt = torch.optim.Adam(topo_model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print("\n[복구 학습] 낮은 학습률(1e-4) + 512 전용 위상 오탐 억제(k=9, loss_recall 제거) 2에폭 실행...")
    for ep in range(2):
        t0 = time.perf_counter()
        topo_model.train()
        tot_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = topo_model(imgs)
                loss = F.cross_entropy(logits, labels) + dice_loss(logits, labels)
                loss = loss + topo_fp_suppression_loss_512(logits.float(), imgs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item()
        print(f"  [Topo-Fix] Epoch {ep+1}/2 | loss = {tot_loss/len(train_loader):.4f} ({time.perf_counter()-t0:.1f}s)")

    torch.save({"epoch": 11, "model": topo_model.state_dict()}, WEIGHTS_DIR / "ckpt_512_topo_fixed.pth")
    iou_t, fp_t = evaluate_512("2. 교정된 512x512 Topo-ExG U-Net", topo_model, val_loader, device)

    print(f"\n[최종 요약] 잡초 IoU: {iou_b:.4f} -> {iou_t:.4f} ({(iou_t-iou_b)*100:+.2f}%p) | FP 블롭: {fp_b} -> {fp_t} ({fp_t-fp_b:+d}개)")


if __name__ == "__main__":
    main()