#!/bin/bash
set -e

echo "========================================================"
echo "      HE THONG NHAN DIEN VA THEO DOI PHUONG TIEN"
echo "========================================================"

# 1. Kiem tra Python
if ! command -v python3 &> /dev/null; then
    echo "[LOI] May chua cai dat python3!"
    exit 1
fi

# 2. Tao venv va cai thu vien neu chua co
if [ ! -d "venv" ]; then
    echo "[1/2] Dang tao moi truong ao venv..."
    python3 -m venv venv
    echo "[2/2] Dang cai dat thu vien..."
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
    echo "[OK] Cai dat xong thu vien!"
fi

# 3. Chay chuong trinh
echo "Dang khoi chay chuong trinh..."
./venv/bin/python yolo_detect.py
