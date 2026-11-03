import glob
import marshal
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


def _detect_original_min_size(default_val=15):
    """__pycache__에 백업된 원본 .pyc 상수에서 원래의 min_size 기본값을 자동 복원"""
    try:
        pyc_files = glob.glob(str(Path(__file__).resolve().parent / "__pycache__" / "test_highres_512*.pyc"))
        if pyc_files:
            code = marshal.loads(open(pyc_files[0], "rb").read()[16:])
            for i, c in enumerate(code.co_consts):
                if hasattr(c, "co_name") and c.co_name == "evaluate_512" and i > 0:
                    prev = code.co_consts[i - 1]
                    if isinstance(prev, tuple) and len(prev) == 1 and isinstance(prev[0], int):
                        return prev[0]
    except Exception:
        pass
    return default_val


ORIG_MIN_SIZE = _detect_original_min_size(15)


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
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = self._block(3, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.bottleneck = self._block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)
        self.out_conv = nn.Conv2d(32, num_classes, 1)

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
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def dice_loss(logits, targets, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, 3).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


class SmoothRCC8LossWrapper(nn.Module):
    """
    torch_morph_rcc8.diagnostics의 DifferentiableFuzzyRCC8Loss, morph_dilation, morph_erosion을
    직접 호출하여 파편화 없이 작물 및 토양 오탐을 억제하는 모듈
    """
    def __init__(self):
        super().__init__()
        self.rcc8 = DifferentiableFuzzyRCC8Loss(kernel_size=3)

    def forward(self, logits: torch.Tensor, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        p_crop = probs[:, 1:2]
        p_weed = probs[:, 2:3]
        non_weed = (labels != 2).float().unsqueeze(1)

        # 분모(vol_mu) 고정 및 파편화 방지를 위한 선형 스케일링
        nu_weed_fp = 0.5 + 0.5 * (p_weed * non_weed)

        # 1. 작물 내부 및 외곽 경계(k=9) 보호 구역 생성
        gt_crop = (labels == 1).float().unsqueeze(1)
        dil_crop = morph_dilation(gt_crop, kernel_size=9)
        closed_crop = morph_erosion(morph_dilation(p_crop.detach(), kernel_size=9), kernel_size=9)
        crop_zone = torch.clamp(0.6 * dil_crop + 0.4 * closed_crop, 0.0, 1.0) * non_weed
        mu_crop = (0.4 * crop_zone + 1e-4).detach()
        loss_crop_fp = 2.0 * (self.rcc8(mu_crop, nu_weed_fp)["fuzzy_PO"] - 0.5)

        # 2. 비식생 토양(Soil) 보호 구역 생성
        imgs_safe = torch.clamp(imgs, 0.0, 1.0)
        r, g, b = imgs_safe[:, 0:1], imgs_safe[:, 1:2], imgs_safe[:, 2:3]
        exg_norm = (2.0 * g - r - b) / (r + g + b + 1e-5)
        soil_zone = torch.clamp((0.05 - exg_norm) / 0.05, 0.0, 1.0) * non_weed
        mu_soil = (0.4 * soil_zone + 1e-4).detach()
        loss_soil_fp = 2.0 * (self.rcc8(mu_soil, nu_weed_fp)["fuzzy_PO"] - 0.5)

        return 1.5 * loss_crop_fp + 1.5 * loss_soil_fp


def evaluate_512(name, model, loader, device, min_size=ORIG_MIN_SIZE):
    """바이트코드(160~196번 줄)에서 100% 복원된 원본 평가 함수"""
    model.eval()
    cm = np.zeros((3, 3), dtype=np.int64)
    fp_crop, fp_soil, tp_blobs = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs_gpu = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs_gpu)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = labels.numpy()
            for b in range(preds.shape[0]):
                pred, gt = preds[b], gts[b]
                y_t, y_p = gt.ravel(), pred.ravel()
                cm += np.bincount(3 * y_t + y_p, minlength=9).reshape(3, 3)
                n_w, w_cc = cv2.connectedComponents((pred == 2).astype(np.uint8))
                for wid in range(1, n_w):
                    wmask = w_cc == wid
                    if wmask.sum() < min_size:
                        continue
                    gt_maj = int(np.bincount(gt[wmask], minlength=3).argmax())
                    if gt_maj == 2:
                        tp_blobs += 1
                    elif gt_maj == 0:
                        fp_soil += 1
                    else:
                        fp_crop += 1
    tp_w = float(cm[2, 2])
    fp_w = float(cm[:, 2].sum() - tp_w)
    fn_w = float(cm[2, :].sum() - tp_w)
    weed_iou = tp_w / (tp_w + fp_w + fn_w + 1e-6)
    weed_prec = tp_w / (tp_w + fp_w + 1e-6)
    weed_rec = tp_w / (tp_w + fn_w + 1e-6)
    print(f"\n=== [{name}] ===")
    print(f"  * 잡초 IoU = {weed_iou:.4f} (Precision = {weed_prec:.4f}, Recall = {weed_rec:.4f})")
    print(f"  * 오탐(FP) 블롭 = 총 {fp_soil + fp_crop}개 (맨땅: {fp_soil}, 작물: {fp_crop}) | 진짜 잡초(TP): {tp_blobs}개")
    return weed_iou, fp_soil + fp_crop


def train_12ep(tag, model, train_loader, device, ckpt_path, rcc8_loss_fn=None):
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
        # 에폭 웜업: 초반(1~4ep) 기본 형태 학습 후 중후반(5~12ep) 위상 페널티 점진적 강화
        topo_weight = 0.0 if ep < 4 else min(1.0, (ep - 3) / 5.0)

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels) + dice_loss(logits, labels)
                if rcc8_loss_fn is not None and topo_weight > 0.0:
                    loss = loss + topo_weight * rcc8_loss_fn(logits.float(), imgs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item()
        dt = time.perf_counter() - t0
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt_path)
        print(f"  [{tag}] Epoch {ep+1:2d}/{EPOCHS} (w={topo_weight:.2f}) | loss = {tot_loss/len(train_loader):.4f} ({dt:.1f}s) [저장됨]")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PhenoBenchDataset(BASE_DIR, "train", img_size=HIGH_RES)
    val_ds = PhenoBenchDataset(BASE_DIR, "val", img_size=HIGH_RES)

    # 1. 512x512 Baseline (12에폭 완료본 로드 후 원본 기준으로 평가)
    set_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"\n[1/2] 512x512 Baseline U-Net (12 Epochs, min_size={ORIG_MIN_SIZE}) 확인...")
    base_model = SimpleUNet().to(device)
    train_12ep("base", base_model, train_loader, device, WEIGHTS_DIR / "ckpt_512_base.pth", rcc8_loss_fn=None)
    iou_b, fp_b = evaluate_512("1. 512x512 Baseline U-Net (12ep)", base_model, val_loader, device)

    # 2. 512x512 Topo-RCC8 U-Net (동일하게 1~12에폭 학습 후 원본 기준으로 정면 비교)
    set_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print("\n[2/2] 512x512 Topo-RCC8 U-Net (1~12 Epochs 동일 조건) 학습 시작...")
    topo_model = SimpleUNet().to(device)
    rcc8_loss_fn = SmoothRCC8LossWrapper().to(device)
    train_12ep("topo_12ep", topo_model, train_loader, device, WEIGHTS_DIR / "ckpt_512_topo_12ep.pth", rcc8_loss_fn=rcc8_loss_fn)
    iou_t, fp_t = evaluate_512("2. 512x512 Topo-RCC8 U-Net (12ep)", topo_model, val_loader, device)

    print(f"\n[512x512 12에폭 정면 비교 요약] 잡초 IoU: {iou_b:.4f} -> {iou_t:.4f} ({(iou_t-iou_b)*100:+.2f}%p) | FP 블롭: {fp_b} -> {fp_t} ({fp_t-fp_b:+d}개)")


if __name__ == "__main__":
    main()