@echo off
rem Detached launcher for run_stage2_matrix.ps1 (A2: not tied to any chat or
rem terminal session). Start it from PowerShell with:
rem
rem   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'cmd.exe /c ""D:\Mahidol University\SP\caries-detection-model\run_stage2_task.cmd""' }
rem
rem The parent is WmiPrvSE (a service), so the driver survives restarts of the
rem OpenCode server / shell sessions. The driver itself skips finished runs
rem and resumes partial folds, so relaunching after a reboot is safe.
rem Console output is appended to outputs\stage2_driver_stdout.log.
cd /d "D:\Mahidol University\SP\caries-detection-model"
powershell -NoProfile -ExecutionPolicy Bypass -File "run_stage2_matrix.ps1" >> "outputs\stage2_driver_stdout.log" 2>&1
