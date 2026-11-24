# -*- coding: utf-8 -*-
"""Chép trọng số đã huấn luyện từ các file results_*.zip (tải về từ Kaggle) vào thư mục models/ của demo.

Cách chạy (trong thư mục Demo):
    python setup_models.py                       # mặc định tìm zip ở ../Results
    python setup_models.py --results "D:/duong/dan/Results"
Chỉ cần chạy một lần. Chạy lại sẽ bỏ qua file đã có (thêm --force để chép đè).
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (file zip, đường dẫn trong zip, đích trong models/)
FILES = [
    ("results_03_yolo_pill.zip", "yolo_out/weights/yolo_pill_1cls.pt", "yolo_pill_1cls.pt"),
    ("results_03_yolo_pill.zip", "yolo_out/metrics_yolo.json", "metrics_yolo.json"),
    ("results_05_embed_arcface.zip", "embed_arcface_imagenet/config.json", "embed_arcface_imagenet/config.json"),
    ("results_05_embed_arcface.zip", "embed_arcface_imagenet/encoder.pt", "embed_arcface_imagenet/encoder.pt"),
    ("results_05_embed_arcface.zip", "embed_arcface_imagenet/gallery.npy", "embed_arcface_imagenet/gallery.npy"),
    ("results_05_embed_arcface.zip", "embed_arcface_imagenet/gallery_classes.json", "embed_arcface_imagenet/gallery_classes.json"),
    ("results_06_match_eval.zip", "eval_out/metrics_match.json", "metrics_match.json"),
    ("results_08_text_det_dbnet.zip", "dbnet_out/dbnet_vaipe.pt", "dbnet_vaipe.pt"),
    ("results_08_text_det_dbnet.zip", "dbnet_out/metrics_det.json", "metrics_det.json"),
    ("results_09_text_rec_vietocr.zip", "vietocr_out/vietocr_vaipe.pth", "vietocr_vaipe.pth"),
    ("results_09_text_rec_vietocr.zip", "vietocr_out/vietocr_config.yml", "vietocr_config.yml"),
    ("results_09_text_rec_vietocr.zip", "vietocr_out/metrics_rec.json", "metrics_rec.json"),
    ("results_10_field_cls_phobert.zip", "phobert_out/config.json", "phobert_out/config.json"),
    ("results_10_field_cls_phobert.zip", "phobert_out/field_cls.pt", "phobert_out/field_cls.pt"),
    ("results_10_field_cls_phobert.zip", "phobert_out/metrics_field.json", "phobert_out/metrics_field.json"),
    ("results_11_pres_eval.zip", "pres_eval_out/drug_dict.json", "drug_dict.json"),
    ("results_11_pres_eval.zip", "pres_eval_out/metrics_pres.json", "metrics_pres.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "Results"), help="thư mục chứa các file results_*.zip")
    ap.add_argument("--out", default=str(HERE / "models"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    missing = sorted({z for z, _, _ in FILES if not (res / z).exists()})
    if missing:
        print("Không tìm thấy trong", res, ":", ", ".join(missing))
        sys.exit(1)
    for z, src, dst in FILES:
        d = out / dst
        with zipfile.ZipFile(res / z) as zf:
            size = zf.getinfo(src).file_size
            if d.exists() and d.stat().st_size == size and not a.force:
                print("  đã có   ", dst)
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(src) as fi, open(d, "wb") as fo:      # chép dở dang → lần chạy sau tự chép lại (so kích thước)
                shutil.copyfileobj(fi, fo, 1 << 20)
        print("  đã chép ", dst, f"({d.stat().st_size / 1e6:.1f} MB)")
    print("Xong. Thư mục models:", out)


if __name__ == "__main__":
    main()
