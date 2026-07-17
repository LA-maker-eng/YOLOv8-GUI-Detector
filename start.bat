@echo off
cd /d "%~dp0"
echo 激活虚拟环境...
call venv\Scripts\activate.bat
echo 虚拟环境激活成功
echo.
echo 启动程序...
python Detect_GUI.py
echo.
echo 程序已退出
pause
