@echo off
set PYTHONIOENCODING=utf-8
cd /d G:\lidongyv.github.io
"C:\Users\Admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" -X utf8 _build\publish.py -m "wam-survey multi-page site" > _build\publish_wam.log 2>&1
echo EXITCODE=%ERRORLEVEL%
