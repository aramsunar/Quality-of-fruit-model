@echo off
REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo Compiling LaTeX document...
echo Current directory: %CD%
echo.

REM First compilation
"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex

REM Second compilation (for references, TOC, etc.)
"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex

echo.
if exist main.pdf (
    echo SUCCESS! PDF generated: main.pdf
    echo Renaming to ITRI626_Group_Mini_Project.pdf...
    copy /Y main.pdf ITRI626_Group_Mini_Project.pdf
    echo Opening PDF...
    start ITRI626_Group_Mini_Project.pdf
) else (
    echo ERROR: PDF was not generated. Check main.log for details.
)
echo.
pause
