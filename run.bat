@echo off
chcp 65001 >nul
title Vehicle Detection System

echo ========================================================
echo       HE THONG NHAN DIEN VA THEO DOI PHUONG TIEN
echo ========================================================
echo.

:: 1. Kiem tra xem may da cai Python chua
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] May cua ban chua cai dat Python hoac chua them vao PATH!
    echo.
    echo Huong dan:
    echo  1. Tai Python (khuyen nghi 3.10 - 3.12) tai: https://www.python.org/downloads/
    echo  2. Khi cai dat, NHO TICH VAO O: "Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

:: 2. Kiem tra va tu dong tao venv neu chua co
if not exist "venv" (
    echo [1/3] Phat hien chua co moi truong ao. Dang tao venv...
    python -m venv venv
    if errorlevel 1 (
        echo [LOI] Khong the khoi tao venv!
        pause
        exit /b 1
    )
    echo [OK] Tao moi truong ao thanh cong!
    echo.

    echo [2/3] Dang cai dat cac thu vien can thiet tu requirements.txt...
    echo (Qua trinh nay chi dien ra 1 lan dau tien, mat khoang 1-3 phut tuy toc do mang)...
    echo.
    .\venv\Scripts\python.exe -m pip install --upgrade pip
    .\venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo [LOI] Cai dat thu vien that bai! Vui long kiem tra lai ket noi mang.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Da cai dat xong toan bo thu vien!
    echo ========================================================
    echo.
)

:: 3. Khoi chay chuong trinh
echo [3/3] Dang khoi chay chuong trinh Nhan dien...
echo.
.\venv\Scripts\python.exe yolo_detect.py

echo.
echo ========================================================
echo Chuong trinh da dung.
pause
