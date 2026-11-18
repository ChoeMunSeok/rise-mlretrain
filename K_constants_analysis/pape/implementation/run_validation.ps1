$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot 'check.ps1')
& (Join-Path $PSScriptRoot 'run.ps1') -Stage run-validation
Write-Output 'COMPLETE: fresh model, 2015 monthly Validation, confusion matrices and handoff bundle.'
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'outputs/handoff/latest.json') -Encoding UTF8
