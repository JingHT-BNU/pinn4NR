@echo off
rem a2q_runner.cmd <variant> <exp_name> <steps> [extra args...]
rem 自动重启包装:训练崩溃后从断点续训,直到 model.pt 生成或重试上限。
rem 每次尝试独立日志(避免与存活进程的句柄冲突);后台无控制台用 ping 延时。
setlocal enabledelayedexpansion
set VARIANT=%1
set EXP=%2
set STEPS=%3
set LOGDIR=D:\AIs\PINN\paper\A2_parametric\runs
set N=0
:retry
set /a N+=1
if %N% GTR 20 (
  echo RETRY LIMIT REACHED for %EXP% %date% %time% >> "%LOGDIR%\%EXP%_runner.log"
  exit /b 1
)
echo === attempt %N% %EXP% %date% %time% === >> "%LOGDIR%\%EXP%_runner.log"
"D:\AIs\PINN\.venv\Scripts\python.exe" -u "D:\AIs\PINN\paper\A2_parametric\a2q_train.py" --variant %VARIANT% --steps %STEPS% --exp-name %EXP% %4 %5 %6 %7 > "%LOGDIR%\%EXP%_train_a%N%.log" 2>&1
if exist "%LOGDIR%\%EXP%\model.pt" (
  echo DONE %EXP% after %N% attempts %date% %time% >> "%LOGDIR%\%EXP%_runner.log"
  exit /b 0
)
rem 快速失败(启动即退)退避 60s,慢退(有 ckpt 进展)退避 10s
ping -n 61 127.0.0.1 >nul
goto retry
