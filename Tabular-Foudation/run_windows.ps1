# Chạy dự án trên Windows (PowerShell), tại thư mục Tabular-Foudation.
#   .\run_windows.ps1 check
#   .\run_windows.ps1 benchmark -Smoke
#   .\run_windows.ps1 all
param([Parameter(Position=0)][string]$Command = "check", [switch]$Smoke, [switch]$Resume)
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -q --upgrade pip
& $py -m pip install -q -r requirements.txt
& $py code_paysim\tests\test_no_leakage.py
$a = @($Command)
if ($Smoke) { $a += "--smoke" }
if ($Resume) { $a += "--resume" }
& $py run_all.py @a
