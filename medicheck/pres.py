# -*- coding: utf-8 -*-
"""Nhánh đơn thuốc: M2 DBNet (tìm dòng) → M3 VietOCR (đọc chữ) → M4 PhoBERT (phân loại trường) → M5 (mã thuốc)."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageOps

from .text_utils import clean_text, DrugIndex

FIELD_VI = {"drugname": "tên thuốc", "quantity": "số lượng", "usage": "cách dùng",
            "diagnose": "chẩn đoán", "date": "ngày", "other": "khác"}


class FieldClassifier(nn.Module):
    """PhoBERT + đặc trưng vị trí (giống notebook 10)."""

    def __init__(self, bert, n_cls, n_pos=5):
        super().__init__()
        self.bert = bert
        h = self.bert.config.hidden_size
        self.pos = nn.Sequential(nn.Linear(n_pos, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h + 64, 256), nn.ReLU(), nn.Linear(256, n_cls))

    def forward(self, ids, mask, pos):
        cls = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([cls, self.pos(pos)], 1))


def order_lines(boxes):
    """Thứ tự đọc: từ trên xuống (theo hàng ~12 px), trái sang phải."""
    return sorted(range(len(boxes)), key=lambda i: (round(((boxes[i][1] + boxes[i][3]) / 2) / 12), boxes[i][0]))


class PrescriptionReader:
    def __init__(self, models_dir, device="cpu"):
        models_dir = Path(models_dir)
        self.device = torch.device(device)
        torch.set_grad_enabled(False)

        # M2 – DBNet (docTR)
        from doctr.models import detection_predictor
        self.det = detection_predictor("db_resnet50", pretrained=False, assume_straight_pages=True, batch_size=1)
        self.det.model.load_state_dict(torch.load(models_dir / "dbnet_vaipe.pt", map_location="cpu"))
        self.det.model.eval()

        # M3 – VietOCR
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        vcfg = Cfg.load_config_from_file(str(models_dir / "vietocr_config.yml"))
        vcfg["weights"] = str(models_dir / "vietocr_vaipe.pth")
        vcfg["device"] = "cpu" if self.device.type == "cpu" else "cuda:0"
        vcfg.setdefault("cnn", {})["pretrained"] = False      # không tải VGG ImageNet, trọng số đã có trong file .pth
        vcfg.setdefault("predictor", {})["beamsearch"] = False
        self.ocr = Predictor(vcfg)

        # M4 – PhoBERT phân loại trường
        from transformers import AutoModel, PhobertTokenizer, RobertaConfig
        pdir = models_dir / "phobert_out"
        self.pcfg = json.loads((pdir / "config.json").read_text(encoding="utf-8"))
        name = self.pcfg["model_name"]
        local_tok = models_dir / "phobert_tokenizer"          # nếu đã lưu sẵn bộ tách từ thì chạy được không cần mạng
        # Dùng đúng bộ tách từ PhoBERT bản "chậm" (vocab.txt + bpe.codes) như lúc huấn luyện; lần đầu sẽ tải ~2 MB từ Hugging Face.
        self.tok = PhobertTokenizer.from_pretrained(str(local_tok) if local_tok.exists() else name)
        # Cấu hình PhoBERT-base (giống vinai/phobert-base). Trọng số đã fine-tune nằm trọn trong field_cls.pt nên không cần tải.
        bcfg = RobertaConfig(vocab_size=64001, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                             intermediate_size=3072, max_position_embeddings=258, type_vocab_size=1,
                             pad_token_id=1, bos_token_id=0, eos_token_id=2, layer_norm_eps=1e-5)
        bert = AutoModel.from_config(bcfg)
        self.labels = self.pcfg["labels"]
        self.fcls = FieldClassifier(bert, len(self.labels), self.pcfg.get("n_pos", 5)).to(self.device).eval()
        self.fcls.load_state_dict(torch.load(pdir / "field_cls.pt", map_location=self.device))
        from pyvi import ViTokenizer
        self.seg = lambda t: ViTokenizer.tokenize(t) if t.strip() else ""

        # M5 – từ điển tên thuốc
        self.index = DrugIndex(json.loads((models_dir / "drug_dict.json").read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ các bước
    def detect(self, img_np):
        h, w = img_np.shape[:2]
        r = self.det([img_np])[0]
        words = r.get("words", next(iter(r.values()), [])) if isinstance(r, dict) else r
        words = np.asarray(words)
        if not len(words):
            return []
        return [[float(b[0] * w), float(b[1] * h), float(b[2] * w), float(b[3] * h)] for b in words[:, :4]]

    def recognize(self, pil, boxes, pad=2):
        W, H = pil.size
        crops = [pil.crop((max(0, b[0] - pad), max(0, b[1] - pad), min(W, b[2] + pad), min(H, b[3] + pad))) for b in boxes]
        out = []
        for i in range(0, len(crops), 16):
            out += self.ocr.predict_batch(crops[i:i + 16])
        return [clean_text(t) for t in out]

    def classify(self, texts, boxes, W, H):
        if not texts:
            return []
        idx = order_lines(boxes)
        rank = {i: o for o, i in enumerate(idx)}
        prev = {idx[o]: (texts[idx[o - 1]] if o > 0 else "") for o in range(len(idx))}
        enc = self.tok([self.seg(t) for t in texts], [self.seg(prev[i]) for i in range(len(texts))], truncation=True,
                       max_length=self.pcfg["max_len"], padding="max_length", return_tensors="pt")
        pos = torch.tensor([[b[0] / W, b[1] / H, b[2] / W, b[3] / H, rank[i] / max(len(texts), 1)]
                            for i, b in enumerate(boxes)], dtype=torch.float32)
        p = self.fcls(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device), pos.to(self.device))
        prob = torch.softmax(p.float(), 1).cpu().numpy()
        return [(self.labels[k], float(prob[i, k])) for i, k in enumerate(prob.argmax(1))]

    # ------------------------------------------------------------------ toàn bộ
    # set_grad_enabled(False) ở __init__ chỉ có tác dụng trên luồng đã nạp mô hình; Streamlit chạy mỗi lượt trên luồng khác
    # nên phải tắt gradient ngay tại hàm suy luận.
    @torch.inference_mode()
    def read(self, pil):
        """Đọc một ảnh đơn thuốc. Trả về dict gồm các dòng chữ, danh sách thuốc và thời gian từng bước."""
        pil = ImageOps.exif_transpose(pil).convert("RGB")      # ảnh chụp điện thoại: xoay đúng chiều trước khi tìm dòng chữ
        W, H = pil.size
        t = {}
        t0 = time.time(); boxes = self.detect(np.array(pil)); t["DBNet"] = time.time() - t0
        order = order_lines(boxes); boxes = [boxes[i] for i in order]
        t0 = time.time(); texts = self.recognize(pil, boxes); t["VietOCR"] = time.time() - t0
        t0 = time.time(); fields = self.classify(texts, boxes, W, H); t["PhoBERT"] = time.time() - t0
        lines = [dict(idx=i, box=b, text=tx, field=f, field_prob=p) for i, (b, tx, (f, p)) in enumerate(zip(boxes, texts, fields))]

        # M5 + ghép số lượng / cách dùng với thuốc đứng ngay trước
        t0 = time.time()
        drugs = []
        for ln in lines:
            if ln["field"] == "drugname":
                m = self.index.match(ln["text"])
                drugs.append(dict(line=ln["idx"], text=ln["text"], match=m, quantity=[], usage=[]))
            elif ln["field"] in ("quantity", "usage") and drugs:
                drugs[-1][ln["field"]].append(ln["text"])
        t["M5"] = time.time() - t0
        cand = sorted({c for d in drugs for c in d["match"]["cand"]})
        return dict(size=(W, H), lines=lines, drugs=drugs, c_don_cand=cand, timing=t)
