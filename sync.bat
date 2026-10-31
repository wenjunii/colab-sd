@echo off
setlocal
python "%~dp0sync_to_github.py" %*
set "sync_exit=%ERRORLEVEL%"
pause
exit /b %sync_exit%
