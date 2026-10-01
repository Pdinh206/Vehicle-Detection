@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Vehicle Detection System

echo ========================================================
echo       HE THONG NHAN DIEN VA THEO DOI PHUONG TIEN
echo ========================================================
echo.

:: 1. Kiem tra xem co Python tren may khong
python --version >nul 2>&1
if %errorlevel% neq 0 goto :NO_PYTHON

:: 2. Kiem tra xem Python hien tai da co san thu vien chua
python -c "import ultralytics, cv2" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Phat hien moi truong Python tren may da co san day du thu vien.
    echo Dang khoi chay chuong trinh...
    echo.
    python yolo_detect.py
    goto :END
)

:: 3. Neu Python he thong chua co thu vien, kiem tra moi truong ao venv
if exist "venv\Scripts\python.exe" (
    echo [OK] Dang su dung moi truong ao venv...
    echo Dang khoi chay chuong trinh...
    echo.
    .\venv\Scripts\python.exe yolo_detect.py
    goto :END
)

:: 4. Neu chua co venv, tao moi va cai thu vien
echo [1/2] May chua co thu vien. Dang tao moi truong ao venv...
python -m venv venv
if %errorlevel% neq 0 goto :VENV_ERROR
echo [OK] Tao venv thanh cong!
echo.

echo [2/2] Dang cai dat cac thu vien can thiet tu requirements.txt...
echo Qua trinh nay mat khoang 1-3 phut tuy toc do mang...
echo.
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 goto :INSTALL_ERROR

echo.
echo [OK] Cai dat xong thu vien!
echo Dang khoi chay chuong trinh...
echo.
.\venv\Scripts\python.exe yolo_detect.py
goto :END

:NO_PYTHON
echo [LOI] May ban chua cai dat Python hoac chua them Python vao PATH!
echo.
echo Huong dan:
echo  1. Tai Python tai: https://www.python.org/downloads/
echo  2. Khi cai dat, nho tich vao o: "Add python.exe to PATH"
echo.
goto :END

:VENV_ERROR
echo.
echo [LOI] Khong the tao moi truong ao venv!
goto :END

:INSTALL_ERROR
echo.
echo [LOI] Co loi khi cai dat thu vien! Vui long kiem tra lai ket noi mang.
goto :END

:END
echo.
echo ========================================================
echo Chuong trinh da ket thuc. Nhan phim bat ky de dong cua so...
pause >nul
endlocal
