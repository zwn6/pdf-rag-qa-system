@echo off
chcp 65001
echo 正在激活虚拟环境...
call .venv\Scripts\activate
echo 启动Streamlit网页程序
streamlit run app.py
pause