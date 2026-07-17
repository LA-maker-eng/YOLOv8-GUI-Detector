@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo   YOLOv8-PyQt5 视觉检测上位机启动器
echo ========================================
echo.
echo 正在激活虚拟环境...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo 错误：虚拟环境激活失败！
    pause
    exit /b 1
)
echo ✓ 虚拟环境激活成功
echo.
echo 正在启动程序...
echo 提示：请检查桌面或任务栏是否有窗口弹出
echo.
python Detect_GUI.py
echo.
echo ========================================
echo   程序已退出
echo ========================================
pause
