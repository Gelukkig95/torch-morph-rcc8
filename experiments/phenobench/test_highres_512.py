import random
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from torch_morph_rcc8.diagnostics import (
    DifferentiableFuzzyRCC8Loss,
    morph_dilation,
    morph_erosion,
)

BASE_DIR = r"D:\Download\PhenoBench-v110\PhenoBench"
HIGH_RES = 512
BATCH_SIZE = 4
EPOCHS = 12
SEED = 42
WEIGHTS_DIR = ROOT_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PhenoBenchDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=HIGH_RES):
        self.img_dir = Path(root_dir) / split / "images"
        self.sem_dir = Path(root_dir) / split / "semantics"
        self.files = sorted([f.name for f in self.img_dir.glob("*.png")])
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / fname)), cv2.COLOR_BGR2RGB)
        sem = cv2.imread(str(self.sem_dir / fname), cv2.IMREAD_UNCHANGED)
        sem_clean = np.zeros_like(sem, dtype=np.uint8)
        sem_clean[(sem == 1) | (sem == 3)] = 1
        sem_clean[(sem == 2) | (sem == 4)] = 2
        if img.shape[0] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            sem_clean = cv2.resize(sem_clean, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return (
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(sem_clean).long(),
        )


class SimpleUNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.enc1 = self._block(3, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def dice_loss(logits, targets, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, 3).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


class PhenoBenchRCC8LossWrapper(nn.Module):
    """
    torch_morph_rcc8.diagnostics의 DifferentiableFuzzyRCC8Loss, morph_dilation, morph_erosion을
    직접 호출하여 작물(Crop) 및 토양(Soil) 오탐 블롭과 고립 점 노이즈를 억제하는 손실 모듈
    """
    def __init__(self):
        super().__init__()
        self.rcc8 = DifferentiableFuzzyRCC8Loss(kernel_size=3)

    def forward(self, logits: torch.Tensor, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        p_crop = probs[:, 1:2]
        p_weed = probs[:, 2:3]

        non_weed = (labels != 2).float().unsqueeze(1)
        gt_crop = (labels == 1).float().unsqueeze(1)
        gt_soil = (labels == 0).float().unsqueeze(1)

        # 진짜 잡초(labels == 2) 영역은 0으로 분리하고, 오탐 확률에 3x3 팽창을 적용하여
        # DifferentiableFuzzyRCC8Loss 내부의 morph_erosion(nu, 3)과 함께 닫힘 연산(Closing)을 형성
        p_weed_fp = p_weed * non_weed
        nu_fp = morph_dilation(p_weed_fp, kernel_size=3)

        # 활성 오탐 후보 픽셀(p_weed > 0.15) 주변 5x5 영역 마스크 추출
        active_fp = morph_dilation((p_weed.detach() > 0.15).float() * non_weed, kernel_size=5)

        # 1. 작물 내부(NTPP) 및 외곽 경계(EC, k=9) 침범 오탐 억제
        closed_crop = morph_erosion(morph_dilation(p_crop.detach(), kernel_size=5), kernel_size=5)
        crop_zone = torch.clamp(morph_dilation(gt_crop, kernel_size=9) + 0.5 * closed_crop, 0.0, 1.0) * non_weed
        mu_crop = (active_fp * crop_zone + 1e-3 * crop_zone + 1e-4).detach()
        loss_rcc8_crop = self.rcc8(mu_crop, nu_fp)["fuzzy_NTPP"]

        # 2. 비식생 토양(Soil) 오탐 억제 (정규화 ExG 게이트 범위 [0.2, 1.0] 엄밀 제한)
        imgs_safe = torch.clamp(imgs, 0.0, 1.0)
        r, g, b = imgs_safe[:, 0:1], imgs_safe[:, 1:2], imgs_safe[:, 2:3]
        exg_norm = (2.0 * g - r - b) / (r + g + b + 1e-5)
        soil_gate = torch.clamp((0.10 - exg_norm) / 0.10, 0.2, 1.0) * gt_soil
        mu_soil = (active_fp * soil_gate + 1e-3 * soil_gate + 1e-4).detach()
        loss_rcc8_soil = self.rcc8(mu_soil, nu_fp)["fuzzy_NTPP"]

        # 3. 수리형태학 White Top-Hat (p_weed - Opening(p_weed)): 1~3px 고립 점 노이즈 타격
        opened_weed = morph_dilation(morph_erosion(p_weed.detach(), kernel_size=5), kernel_size=5)
        tophat_speckle = F.relu(p_weed - opened_weed) * non_weed
        loss_speckle = (tophat_speckle * (active_fp + 1e-3)).sum() / (active_fp.sum() + 256.0)

        return 0.30 * loss_rcc8_crop + 0.30 * loss_rcc8_soil + 0.30 * loss_speckle


@torch.no_grad()
def evaluate_512(name, model, val_loader, device):
    model.eval()
    tp_w, fp_w, fn_w = 0, 0, 0
    fp_blobs_soil, fp_blobs_crop, tp_blobs = 0, 0, 0
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        gts = labels.numpy()
        for b in range(preds.shape[0]):
            p, g = preds[b], gts[b]
            tp_w += int(((p == 2) & (g == 2)).sum())
            fp_w += int(((p == 2) & (g != 2)).sum())
            fn_w += int(((p != 2) & (g == 2)).sum())
            num_l, l_map = cv2.connectedComponents((p == 2).astype(np.uint8))
            for lid in range(1, num_l):
                mask = l_map == lid
                if (g[mask] == 2).sum() == 0:
                    if (g[mask] == 1).sum() > 0:
                        fp_blobs_crop += 1
                    else:
                        fp_blobs_soil += 1
                else:
                    tp_blobs += 1
    iou = tp_w / (tp_w + fp_w + fn_w + 1e-6)
    prec = tp_w / (tp_w + fp_w + 1e-6)
    rec = tp_w / (tp_w + fn_w + 1e-6)
    tot_fp = fp_blobs_soil + fp_blobs_crop
    print(f"\n=== [{name}] ===")
    print(f"  * 잡초 IoU = {iou:.4f} (Precision = {prec:.4f}, Recall = {rec:.4f})")
    print(f"  * 오탐(FP) 블롭 = 총 {tot_fp}개 (맨땅: {fp_blobs_soil}, 작물: {fp_blobs_crop}) | 진짜 잡초(TP): {tp_blobs}개")
    return iou, tot_fp


def train_model(tag, model, train_loader, device, ckpt_path, rcc8_loss_fn=None):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    start_ep = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_ep = ckpt["epoch"] + 1
        print(f"  -> [{tag}] 체크포인트 로드 완료 (Epoch {start_ep}/{EPOCHS})")

    for ep in range(start_ep, EPOCHS):
        t0 = time.perf_counter()
        model.train()
        tot_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels) + dice_loss(logits, labels)
                if rcc8_loss_fn is not None:
                    loss = loss + rcc8_loss_fn(logits.float(), imgs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item()
        dt = time.perf_counter() - t0
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt_path)
        print(f"  [{tag}] Epoch {ep+1:2d}/{EPOCHS} | loss = {tot_loss/len(train_loader):.4f} ({dt:.1f}s) [저장됨]")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PhenoBenchDataset(BASE_DIR, "train", img_size=HIGH_RES)
    val_ds = PhenoBenchDataset(BASE_DIR, "val", img_size=HIGH_RES)

    # 1. Baseline (기존 12에폭 체크포인트 재사용)
    set_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("\n[1/2] 512x512 Baseline U-Net 확인...")
    base_model = SimpleUNet().to(device)
    train_model("base", base_model, train_loader, device, WEIGHTS_DIR / "ckpt_512_base.pth", rcc8_loss_fn=None)
    iou_b, fp_b = evaluate_512("1. 512x512 Baseline U-Net", base_model, val_loader, device)

    # 2. Topo-RCC8 U-Net (diagnostics.py의 DifferentiableFuzzyRCC8Loss로 1~12에폭 정석 학습)
    set_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print("\n[2/2] 512x512 Topo-RCC8 U-Net (diagnostics.DifferentiableFuzzyRCC8Loss) 1~12에폭 학습 시작...")
    topo_model = SimpleUNet().to(device)
    rcc8_loss_fn = PhenoBenchRCC8LossWrapper().to(device)
    train_model("topo_diag", topo_model, train_loader, device, WEIGHTS_DIR / "ckpt_512_topo_diag.pth", rcc8_loss_fn=rcc8_loss_fn)
    iou_t, fp_t = evaluate_512("2. 512x512 Topo-RCC8 U-Net", topo_model, val_loader, device)

    print(f"\n[최종 요약] 잡초 IoU: {iou_b:.4f} -> {iou_t:.4f} ({(iou_t-iou_b)*100:+.2f}%p) | FP 블롭: {fp_b} -> {fp_t} ({fp_t-fp_b:+d}개)")


if __name__ == "__main__":
    main()