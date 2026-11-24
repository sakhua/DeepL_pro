# MediCheck — demo Streamlit (NHÓM 03, Học sâu)

Demo chạy toàn bộ hệ thống trên **CPU** (laptop 16 GB RAM, không GPU):

| Tab | Mô hình | Việc làm |
|---|---|---|
| ① Đọc đơn thuốc | DBNet → VietOCR → PhoBERT → M5 | tìm dòng chữ, đọc chữ, phân loại trường, khớp tên thuốc → tập mã ứng viên **C_đơn** |
| ② Đối chiếu viên thuốc | YOLO11s → ConvNeXt-T + ArcFace → so prototype | tìm từng viên, so với các thuốc trong đơn, kết luận *khớp / tin cậy thấp / ngoài đơn* theo ngưỡng τ |
| ③ Tương tác thuốc | DDInter 2.0 | tra cặp hoạt chất có tương tác, mức độ nghiêm trọng / trung bình / nhẹ |

> Công cụ phục vụ học tập, nghiên cứu — **không thay thế dược sĩ, bác sĩ**.

## 1. Cài đặt (làm một lần)

Mở terminal trong VS Code tại thư mục `Demo`:

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps vietocr==0.3.13
```

`vietocr` cài với `--no-deps` vì gói này ghim các thư viện cũ (imgaug…) không cần khi chạy dự đoán.

## 2. Chép trọng số từ các file kết quả Kaggle

Các file `results_*.zip` nằm ở `../Results`. Chạy:

```bash
python setup_models.py
```

Lệnh này lấy đúng các file cần thiết (~950 MB) vào `models/`. Nếu zip để chỗ khác: `python setup_models.py --results "D:/duong/dan/Results"`.

## 3. Chạy

```bash
python -m streamlit run app.py
```

Nên chạy bằng `python -m streamlit` (sau khi đã kích hoạt `.venv` của thư mục `Demo`) thay vì gõ `streamlit` trực tiếp. Lý do: nếu máy có nhiều môi trường ảo, lệnh `streamlit` có thể lấy nhầm bản của project khác. Kiểm tra bằng `python -c "import sys; print(sys.executable)"`: đường dẫn in ra phải nằm trong `Demo\.venv`.

Trình duyệt tự mở `http://localhost:8501`.

- **Lần chạy đầu cần mạng** để tải bộ tách từ PhoBERT (`vinai/phobert-base`, ~2 MB: `vocab.txt`, `bpe.codes`). Các lần sau dùng bản đã lưu trong bộ nhớ đệm của Hugging Face.
  Muốn chạy hoàn toàn không mạng: chép hai file đó vào `models/phobert_tokenizer/`.
- Nạp mô hình lần đầu mất khoảng 20–40 giây. Sau đó mỗi đơn thuốc mất khoảng 10–20 giây (phần lớn là VietOCR), mỗi ảnh viên thuốc khoảng 1–2 giây.

## 4. Bộ ảnh mẫu (`data/samples/`)

Ba đơn trong **tập test** của nhóm (VAIPE), kèm nhãn đúng trong `meta.json` để so sánh:

| Bộ | Đơn | Ảnh viên | Dùng để thử |
|---|---|---|---|
| `don_1094` | Novoxim-500, Hoạt huyết dưỡng não, Mypara 500 | 2 ảnh, mỗi ảnh 6 viên | trường hợp chuẩn, mọi viên khớp |
| `don_798` | Glucofast 500, Enalapril 5mg | 2 ảnh | tương tác metformin – enalapril (mức trung bình) |
| `don_281` | Amoxicilin, Paracetamol, Metronidazol, Hoạt huyết dưỡng não | 3 ảnh | ảnh 2 và 3 toàn viên **ngoài đơn** (nhãn 107) → thử cảnh báo |

Ảnh viên đã được xoay đúng chiều theo EXIF và thu về cạnh dài 1024 px.

Có thể tải ảnh của mình ở mục **Tải ảnh lên** trong thanh bên.

## 5. Giao diện và các tùy chọn

Đầu trang là **kết luận tổng hợp** của cả ca: không phát hiện vấn đề, cần lưu ý, hoặc cần kiểm tra ngay. Ngay dưới là 4 chỉ số chính. Ba thẻ bên dưới cho xem chi tiết đơn thuốc, từng ảnh viên thuốc và các cặp tương tác. Số liệu kỹ thuật (điểm cosin, YOLO conf, thời gian chạy, nhãn đúng) nằm trong các mục **Chi tiết kỹ thuật** thu gọn.

Mục **Cài đặt nâng cao** trong thanh bên có các tùy chọn sau:

- **τ**: ngưỡng cosin để chấp nhận một viên là thuốc trong đơn. Mặc định 0,36, chọn trên tập val ở notebook 06. Kéo thanh trượt không làm chạy lại mô hình.
- **Vùng cần kiểm tra lại**: viên có điểm trong [τ, τ + x) được tô cam để người dùng tự kiểm tra.
- **Dùng danh sách thuốc đúng (nhãn)**: chỉ có ở ca mẫu. Dùng tập thuốc đúng của đơn thay cho kết quả đọc đơn, để tách riêng sai số của nhánh viên thuốc.

Nếu không đọc được thuốc nào trong đơn, app **không** kết luận "đúng thuốc". Khi đó các viên được đánh dấu *chưa đối chiếu được*.

## 6. Dữ liệu tra cứu

- `data/drug_info.csv` — bảng **mã viên VAIPE → tên thuốc → hoạt chất → tên trong DDInter** (107 mã).
  Nhóm lập bảng này. Các dòng có `can_kiem_tra = x` chưa chắc hoạt chất và cần kiểm tra lại với tờ hướng dẫn sử dụng.
  Thuốc thảo dược (Hoạt huyết dưỡng não…) phần lớn không có trong DDInter.
- `data/ddinter_vaipe.csv` — 342 cặp tương tác giữa 39 hoạt chất có trong VAIPE.
  Nguồn là bản tải của DDInter 2.0 (nhóm ATC A, B, D, H, L, P, R, V). Bản tải không có nhóm C, J, M, N…, nên một số cặp có thể bị thiếu.
  Mức `Unknown` nghĩa là DDInter có ghi nhận nhưng chưa phân mức; app để riêng các cặp này trong mục thu gọn.

## 7. Cấu trúc thư mục

```
Demo/
├── app.py                 giao diện Streamlit
├── setup_models.py        chép trọng số từ ../Results/*.zip
├── requirements.txt
├── medicheck/
│   ├── pres.py            nhánh đơn thuốc (M2–M5)
│   ├── pill.py            nhánh viên thuốc (M6–M8)
│   ├── ddi.py             tra tương tác (M9)
│   └── text_utils.py      chuẩn hóa chữ, so khớp tên thuốc (giống notebook 11)
├── data/
│   ├── drug_info.csv, ddinter_vaipe.csv
│   └── samples/don_*/     ảnh đơn, ảnh viên, meta.json
└── models/                tạo bởi setup_models.py
```

## 8. Lỗi thường gặp

- `OSError: Can't load tokenizer for 'vinai/phobert-base'`: máy chưa có mạng ở lần chạy đầu. Xem mục 3.
- `ModuleNotFoundError: vietocr`: chạy lại `pip install --no-deps vietocr==0.3.13`.
- App báo thiếu trọng số: chạy `python setup_models.py`.
