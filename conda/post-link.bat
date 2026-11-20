@echo off
rem Install the system USB driver after conda has linked the vendor files.
rem Automated builds can skip this machine-wide operation explicitly.
if "%QD_SPEC_SKIP_DRIVER_INSTALL%"=="1" exit /b 0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PREFIX%\Scripts\.qd_spec-install-driver.ps1"
exit /b %errorlevel%
