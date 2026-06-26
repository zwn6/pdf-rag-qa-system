@echo off
chcp 65001
echo 正在启动项目虚拟环境...
cd /d %~dp0
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo 虚拟环境激活成功
    echo 启动Streamlit网页程序
    streamlit run app.py
) else (
    echo 未找到虚拟环境.venv，请先执行pip安装依赖！
    pause
)
pause
