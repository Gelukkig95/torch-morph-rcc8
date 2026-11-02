import os
import time
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
BASE_DIR = r"D:\Download\PhenoBench-v110\PhenoBench"
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

HIGH_RES = 512
BATCH_SIZE = 4
EPOCHS = 12


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PhenoBenchDataset(Dataset):
    def __init__(self, base_dir, split="train", img_size=512):
        self.images_dir = f"{base_dir}/{split}/images"
        self.semantics_dir = f"{base_dir}/{split}/semantics"
        self.files = sorted(os.listdir(self.images_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(f"{self.images_dir}/{fname}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        sem = cv2.imread(f"{self.semantics_dir}/{fname}", cv2.IMREAD_UNCHANGED)
        sem = cv2.resize(sem, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        label = np.zeros_like(sem, dtype=np.int64)
        label[(sem == 1) | (sem == 3)] = 1
        label[(sem == 2) | (sem == 4)] = 2
        return img, torch.from_numpy(label)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )
        self.enc1 = block(in_ch, base_ch)
        self.enc2 = block(base_ch, base_ch * 2)
        self.enc3 = block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(base_ch * 4, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = block(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def dice_loss(logits, target, num_classes=3, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    inter = torch.sum(probs * target_oh, (0, 2, 3))
    card = torch.sum(probs + target_oh, (0, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (card + eps)).mean()


def dilate_prob(p, k=5):
    return F.max_pool2d(p, kernel_size=k, stride=1, padding=k // 2)


def erode_prob(p, k=5):
    return -F.max_pool2d(-p, kernel_size=k, stride=1, padding=k // 2)


def topo_exg_guided_loss(logits, imgs, labels):
    probs = F.softmax(logits, dim=1)
    p_crop, p_weed = probs[:, 1], probs[:, 2]
    exg = 2.0 * imgs[:, 1] - imgs[:, 0] - imgs[:, 2]
    non_weed = (labels != 2).float()

    soil_gate = F.relu(0.10 - exg) / 0.10
    loss_soil_fp = (soil_gate * p_weed * non_weed).mean()

    gt_crop = (labels == 1).float().unsqueeze(1)
    dil_crop = dilate_prob(gt_crop, 5).squeeze(1)
    closed_crop = erode_prob(dilate_prob(p_crop.unsqueeze(1), 5), 5).squeeze(1).detach()
    crop_weight = torch.clamp(0.5 * dil_crop + 0.5 * closed_crop, 0.0, 1.0)
    loss_crop_fp = (crop_weight * p_weed * non_weed).mean()

    weed_mask = (labels == 2).float()
    green_gate = torch.clamp(exg / 0.15, 0.0, 1.0)
    loss_recall = (green_gate * (1.0 - p_weed) * weed_mask).mean()
    return loss_soil_fp + loss_crop_fp + 0.5 * loss_recall


def train_with_resume(tag, model, loader, device, use_topo=False):
    ckpt_path = WEIGHTS_DIR / f"ckpt_512_{tag}.pth"
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    start_ep = 0

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_ep = ckpt["epoch"] + 1
        print(f"  -> 체크포인트 로드 완료: Epoch {start_ep + 1}부터 재개합니다.")

    for ep in range(start_ep, EPOCHS):
        t0 = time.perf_counter()
        model.train()
        tot_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels) + dice_loss(logits, labels)
                if use_topo:
                    loss = loss + 0.6 * topo_exg_guided_loss(logits.float(), imgs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item()

        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt_path)
        print(f"  [{tag}] Epoch {ep+1:2d}/{EPOCHS} | loss = {tot_loss/len(loader):.4f} ({time.perf_counter()-t0:.1f}s) [저장됨]")


def evaluate_512(name, model, loader, device, min_size=15):
    model.eval()
    cm = np.zeros((3, 3), dtype=np.int64)
    tp_blobs, fp_soil, fp_crop = 0, 0, 0
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
                    wmask = (w_cc == wid)
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PhenoBenchDataset(BASE_DIR, "train", img_size=HIGH_RES)
    val_ds = PhenoBenchDataset(BASE_DIR, "val", img_size=HIGH_RES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    set_seed(SEED)
    base_model = SimpleUNet().to(device)
    print("\n[1/2] 512x512 Baseline U-Net 학습 시작...")
    train_with_resume("base", base_model, train_loader, device, use_topo=False)
    iou_b, fp_b = evaluate_512("1. 512x512 Baseline U-Net", base_model, val_loader, device)

    set_seed(SEED)
    topo_model = SimpleUNet().to(device)
    print("\n[2/2] 512x512 Topo-ExG U-Net 학습 시작...")
    train_with_resume("topo", topo_model, train_loader, device, use_topo=True)
    iou_t, fp_t = evaluate_512("2. 512x512 Topo-ExG U-Net", topo_model, val_loader, device)

    print(f"\n[최종 요약] 잡초 IoU: {iou_b:.4f} -> {iou_t:.4f} ({(iou_t-iou_b)*100:+.2f}%p) | FP 블롭: {fp_b} -> {fp_t} ({fp_t-fp_b:+d}개)")


if __name__ == "__main__":
    main()