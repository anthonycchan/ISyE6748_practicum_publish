@echo off
setlocal enabledelayedexpansion

REM Path to the exact Python interpreter
set PYTHON_EXE=C:\Users\Anthony\anaconda3\envs\isye6740\python.exe

REM List of numbers
set numbers=1 2 3 5 8 13 21

for %%i in (%numbers%) do (
    echo Seed: %%i
    "%PYTHON_EXE%" Experiment_Final_4.py %%i > results4_%%i.log 2>&1
)

echo Done.
