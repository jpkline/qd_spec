@echo off
"%PYTHON%" -m pip install . --no-deps --no-build-isolation
if errorlevel 1 exit /b 1
xcopy /E /I /Y "vendor\stellarnet_driverLibs" "%SP_DIR%\stellarnet_driverLibs"
if errorlevel 1 exit /b 1
copy /Y "VENDOR_NOTICE.md" "%SP_DIR%\stellarnet_driverLibs\VENDOR_NOTICE.md"
if errorlevel 1 exit /b 1
