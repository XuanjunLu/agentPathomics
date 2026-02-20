@echo off
chcp 65001 >nul
echo ========================================
echo Starting PyPathomics GUI and Viewer
echo ========================================
echo.

REM Start GUI in a new window
echo [1/2] Starting GUI...
start "PyPathomics GUI" cmd /k "cd /d F:\PyPathomics_agent_V2-20260210\PyPathomics-main_v2\agentor && call conda activate agentPathomics_viewer && python gui3.py"

REM Wait 2 seconds before starting the second program
timeout /t 2 /nobreak >nul

REM Start Viewer in a new window
echo [2/2] Starting Viewer...
start "PyPathomics Viewer" cmd /k "cd /d F:\PyPathomics_agent_V2-20260210\agentPyPathomics_viewer && call conda activate agentPathomics_viewer && python app.py"

echo.
echo ========================================
echo Both programs are starting...
echo GUI Window: PyPathomics GUI
echo Viewer Window: PyPathomics Viewer
echo ========================================
echo.
echo Press any key to exit this launcher...
pause >nul

