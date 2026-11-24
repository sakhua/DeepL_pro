# -*- coding: utf-8 -*-
"""Nhánh viên thuốc: M6 YOLO11 (tìm viên) → M7 ConvNeXt + ArcFace (vector đặc trưng) → M8 đối chiếu theo đơn."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

YOLO_LONG_SIDE = 1280      # notebook 01 thu ảnh về cạnh dài 1280 trước khi đưa vào YOLO


class Encoder(nn.Module):
    """Backbone timm + lớp chiếu ra vector đặc trưng đã chuẩn hóa L2 (giống notebook 05)."""

    def __init__(self, backbone, emb_dim=512):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.proj = nn.Sequential(nn.Linear(self.backbone.num_features, emb_dim), nn.BatchNorm1d(emb_dim))

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)).float(), dim=1)


def square_crop(img, box, size, margin):
    """Cắt vuông quanh viên, lề `margin` mỗi bên (giống notebook 01/06)."""
    W, H = img.size
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    side = min(max(w, h) * (1 + 2 * margin), W, H)
    cx, cy = x1 + w / 2, y1 + h / 2
    x0 = min(max(cx - side / 2, 0), W - side)
    y0 = min(max(cy - side / 2, 0), H - side)
    return img.crop((round(x0), round(y0), round(x0 + side), round(y0 + side))).resize((size, size), Image.BICUBIC)


def suppress_nested(dets, thr=0.6):
    """Bỏ khung lồng trong khung khác (NMS của YOLO dùng IoU nên sót trường hợp khung nhỏ nằm gọn trong khung lớn).
    dets đã sắp theo độ tin cậy giảm dần; giữ khung nếu phần giao / diện tích khung nhỏ hơn < thr với mọi khung đã giữ."""
    keep = []
    for b, c in dets:
        ok = True
        for k, _ in keep:
            iw = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            ih = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            amin = min((b[2] - b[0]) * (b[3] - b[1]), (k[2] - k[0]) * (k[3] - k[1]))
            if amin > 0 and iw * ih / amin > thr:
                ok = False
                break
        if ok:
            keep.append((b, c))
    return keep


def prepare_image(pil):
    """Xoay theo EXIF (ảnh điện thoại) và thu về cạnh dài 1280 px như lúc huấn luyện."""
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    s = min(1.0, YOLO_LONG_SIDE / max(pil.size))
    if s < 1:
        pil = pil.resize((round(pil.width * s), round(pil.height * s)), Image.BILINEAR)
    return pil


class PillMatcher:
    def __init__(self, models_dir, device="cpu"):
        models_dir = Path(models_dir)
        self.device = torch.device(device)
        torch.set_grad_enabled(False)
        from ultralytics import YOLO
        self.yolo = YOLO(str(models_dir / "yolo_pill_1cls.pt"))
        edir = models_dir / "embed_arcface_imagenet"
        self.cfg = json.loads((edir / "config.json").read_text(encoding="utf-8"))
        self.enc = Encoder(self.cfg["backbone"], self.cfg["emb_dim"]).to(self.device).eval()
        self.enc.load_state_dict(torch.load(edir / "encoder.pt", map_location=self.device))
        self.G = torch.from_numpy(np.load(edir / "gallery.npy")).float()
        self.classes = json.loads((edir / "gallery_classes.json").read_text())
        self.pos = {int(c): i for i, c in enumerate(self.classes)}
        m = json.loads((models_dir / "metrics_match.json").read_text())
        self.tau = float(next(iter(m["match"].values()))["tau"])
        import torchvision.transforms as T
        s = self.cfg["img_size"]
        self.tf = T.Compose([T.Resize((s, s)), T.ToTensor(), T.Normalize(self.cfg["mean"], self.cfg["std"])])

    def detect(self, pil, conf=0.25, imgsz=1024, nest_thr=0.85):
        r = self.yolo.predict(pil, imgsz=imgsz, conf=conf, device="cpu" if self.device.type == "cpu" else 0, verbose=False)[0]
        dets = sorted(((b.tolist(), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())),
                      key=lambda x: -x[1])
        return suppress_nested(dets, nest_thr)

    def embed(self, crops):
        if not crops:
            return torch.zeros(0, self.G.shape[1])
        x = torch.stack([self.tf(c) for c in crops]).to(self.device)
        return self.enc(x).cpu()

    @torch.inference_mode()            # tắt gradient trên luồng gọi hàm (Streamlit chạy mỗi lượt trên luồng khác)
    def analyze(self, pil, cand_classes, tau=None, low_margin=0.05, det_conf=0.25):
        """Tìm và đối chiếu mọi viên trong ảnh.
        cand_classes: tập mã ứng viên của đơn (C_đơn). Rỗng → so với mọi mã (không có ngữ cảnh đơn).
        Trạng thái mỗi viên: 'khớp' (điểm ≥ τ + low_margin), 'tin cậy thấp' (τ ≤ điểm < τ + low_margin), 'ngoài đơn' (điểm < τ)."""
        tau = self.tau if tau is None else tau
        t = {}
        pil = prepare_image(pil)
        t0 = time.time(); dets = self.detect(pil, conf=det_conf); t["YOLO"] = time.time() - t0
        crops = [square_crop(pil, b, self.cfg.get("crop_size", 256), self.cfg.get("crop_margin", 0.1)) for b, _ in dets]
        t0 = time.time(); Z = self.embed(crops); t["ConvNeXt+ArcFace"] = time.time() - t0
        S = Z @ self.G.T                                                     # cosin với mọi prototype
        cand = [c for c in cand_classes if c in self.pos] or [int(c) for c in self.classes]
        idx = torch.tensor([self.pos[c] for c in cand])
        pills = []
        for k, (b, conf) in enumerate(dets):
            s_c = S[k, idx]
            j = int(s_c.argmax()); best_c, best_s = cand[j], float(s_c[j])
            g = int(S[k].argmax()); glob_c, glob_s = int(self.classes[g]), float(S[k, g])
            if best_s < tau:
                status = "ngoài đơn"
            elif best_s < tau + low_margin:
                status = "tin cậy thấp"
            else:
                status = "khớp"
            top = sorted(zip(cand, s_c.tolist()), key=lambda x: -x[1])[:3]
            pills.append(dict(box=b, det_conf=conf, cls=best_c if status != "ngoài đơn" else None, best_cls=best_c,
                              score=best_s, status=status, global_cls=glob_c, global_score=glob_s, top3=top, crop=crops[k]))
        return dict(image=pil, pills=pills, tau=tau, timing=t)
