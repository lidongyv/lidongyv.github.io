@echo off
REM Rebuild the site from its local sources and publish it.
REM Usage:  publish.cmd            (default commit message)
REM         publish.cmd "message"

setlocal
set PY="C:\Users\Admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
cd /d "%~dp0"

if "%~1"=="" (
  %PY% _build\publish.py
) else (
  %PY% _build\publish.py -m "%~1"
)
endlocal
