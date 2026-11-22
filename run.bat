@echo off
rem Live Taglish captions for whatever is playing on this PC.
rem Extra options go after the script name, e.g.:  run.bat --source mic --font-size 26
cd /d "%~dp0"
py -3.13 -X utf8 captions.py %*
