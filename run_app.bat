@echo off
echo Starting AI Resume Matcher...
echo Application output will be saved to error_log.txt
echo Please open http://127.0.0.1:5000 in your browser.
"C:\Users\Rayan\AppData\Local\Programs\Python\Python313\python.exe" app.py > error_log.txt 2>&1
type error_log.txt
pause
