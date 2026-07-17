@echo off
rem One-command bring-up (Windows)
cd /d "%~dp0\.."
python scripts\run_all.py %*
