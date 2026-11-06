@echo off
rem 室内视觉定位系统一键启动
cd /d %~dp0
call conda activate vpr
python app.py --config results/vlad_config_ORB_K32.json --strategy two_stage
pause
