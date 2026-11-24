# -*- coding: utf-8 -*-
"""Xử lý chữ dùng chung: chuẩn hóa văn bản và chuẩn hóa / so khớp tên thuốc (M5).
Giữ đúng như notebook 11 để kết quả demo khớp với kết quả đánh giá."""
import re
import unicodedata

from rapidfuzz import fuzz, process

CHAR_MAP = {"–": "-", "—": "-", "‐": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...", "×": "x",
            " ": " ", "\t": " ", "µ": "u", "²": "2", "³": "3", "°": "o", "•": "-", "·": ".", "≤": "<", "≥": ">"}


def clean_text(t):
    """Chuẩn hóa NFC (dấu tiếng Việt dựng sẵn), thay ký tự lạ thường gặp, gộp khoảng trắng."""
    t = unicodedata.normalize("NFC", "" if t is None else str(t))
    t = "".join(CHAR_MAP.get(ch, ch) for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def _g2mg(m):
    v = float(m.group(1).replace(",", "."))
    return f"{int(round(v * 1000))}mg"


def norm_drug(t):
    """Chuẩn hóa dòng tên thuốc: bỏ số thứ tự '1)', viết thường, đổi 0,5g → 500mg, bỏ số lặp ('500 500mg' → '500mg')."""
    t = clean_text(t).lower()
    t = re.sub(r"^\s*\d+\s*[\)\.\-:/]\s*", "", t)
    t = t.replace("(", " ").replace(")", " ")
    t = re.sub(r"(\d+(?:[.,]\d+)?)\s*g\b", _g2mg, t)
    t = re.sub(r"(\d+(?:[.,]\d+)?)\s*mg\b", lambda m: m.group(1).replace(",", ".") + "mg", t)
    t = re.sub(r"[^\w\s\+\.%/-]", " ", t)
    toks, out = t.split(), []
    for i, tok in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if re.fullmatch(r"\d+(\.\d+)?", tok) and nxt.startswith(tok) and nxt != tok:
            continue
        if out and tok == out[-1]:
            continue
        out.append(tok)
    return " ".join(out)


class DrugIndex:
    """Bảng ánh xạ hai chiều tên thuốc ↔ mã thuốc VAIPE (đọc từ drug_dict.json của notebook 11)."""

    def __init__(self, drug_dict):
        self.thr = float(drug_dict.get("thr", 60))
        self.cand_thr = float(drug_dict.get("cand_thr", 100))
        self.name2cls = {k: {int(c): n for c, n in v.items()} for k, v in drug_dict["name2cls"].items()}
        self.cls2names = {int(c): v for c, v in drug_dict["cls2names"].items()}
        self.keys = list(self.name2cls)

    def match(self, text):
        """Trả về dict: name (tên khớp nhất trong từ điển), score, single (1 mã), cand (tập mã ứng viên)."""
        q = norm_drug(text)
        if not q:
            return dict(query=q, name=None, score=0.0, single=None, cand=[])
        hits = process.extract(q, self.keys, scorer=fuzz.token_set_ratio, limit=None,
                               score_cutoff=min(self.thr, self.cand_thr))
        if not hits or hits[0][1] < self.thr:
            return dict(query=q, name=hits[0][0] if hits else None, score=float(hits[0][1]) if hits else 0.0,
                        single=None, cand=[])
        best = hits[0][0]
        counts = self.name2cls[best]
        single = max(counts, key=counts.get)
        cand = set(counts)
        for k, s, _ in hits:
            if s >= self.cand_thr:
                cand |= set(self.name2cls[k])
        return dict(query=q, name=best, score=float(hits[0][1]), single=single, cand=sorted(cand))
