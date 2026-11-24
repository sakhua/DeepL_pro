# -*- coding: utf-8 -*-
"""M9 – Tra cứu tương tác thuốc trên DDInter 2.0 (bản đã lọc cho các hoạt chất có trong VAIPE)."""
import csv
from itertools import combinations
from pathlib import Path

LEVEL_ORDER = {"Major": 0, "Moderate": 1, "Minor": 2, "Unknown": 3}
LEVEL_VI = {"Major": "Nghiêm trọng", "Moderate": "Trung bình", "Minor": "Nhẹ", "Unknown": "Có ghi nhận, chưa rõ mức độ"}


class DrugInfo:
    """Bảng mã thuốc VAIPE → tên hiển thị, hoạt chất, tên trong DDInter (data/drug_info.csv)."""

    def __init__(self, path):
        self.rows = {}
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r["ddinter"] = [x for x in r["ddinter_names"].split(";") if x]
                self.rows[int(r["class_id"])] = r

    def name(self, c):
        r = self.rows.get(int(c))
        return r["ten_thuoc"] if r else f"mã {c}"

    def ingredient(self, c):
        r = self.rows.get(int(c))
        return r["hoat_chat"] if r else "?"

    def ddinter(self, c):
        r = self.rows.get(int(c))
        return r["ddinter"] if r else []


class DDIChecker:
    def __init__(self, ddi_csv, info: DrugInfo):
        self.info = info
        self.pairs = {}
        with open(ddi_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.pairs[frozenset((r["drug_a"], r["drug_b"]))] = r["level"]

    def check(self, drug_groups):
        """drug_groups: danh sách (nhãn hiển thị, tập mã thuốc ứng viên) — mỗi phần tử là một thuốc trong đơn.
        Trả về (danh sách tương tác tìm thấy, danh sách cặp chưa có dữ liệu)."""
        groups = []
        for label, classes in drug_groups:
            names = sorted({n for c in classes for n in self.info.ddinter(c)})
            groups.append((label, names))
        found, nodata = [], []
        for (la, na), (lb, nb) in combinations(groups, 2):
            hits = []
            for a in na:
                for b in nb:
                    if a == b:
                        continue
                    lv = self.pairs.get(frozenset((a, b)))
                    if lv:
                        hits.append((a, b, lv))
            if hits:
                for a, b, lv in hits:
                    found.append(dict(thuoc_1=la, thuoc_2=lb, hoat_chat_1=a, hoat_chat_2=b, level=lv,
                                      muc_do=LEVEL_VI.get(lv, lv)))
            elif not na or not nb:
                nodata.append(dict(thuoc_1=la, thuoc_2=lb, ly_do="chưa có hoạt chất trong DDInter"))
        found.sort(key=lambda r: LEVEL_ORDER.get(r["level"], 9))
        return found, nodata
