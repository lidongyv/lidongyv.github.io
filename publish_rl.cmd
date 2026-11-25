@echo off
set PYTHONIOENCODING=utf-8
cd /d G:\lidongyv.github.io
"C:\Users\Admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" -X utf8 _build\publish.py -m "blog: rl post-training article" > _build\publish_rl.log 2>&1
echo EXITCODE=%ERRORLEVEL%
