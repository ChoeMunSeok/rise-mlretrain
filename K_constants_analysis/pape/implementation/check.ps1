$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe'))
& $pythonExe scripts/check_preparation.py
if ($LASTEXITCODE -ne 0) { throw 'Synthetic preparation checks failed' }
