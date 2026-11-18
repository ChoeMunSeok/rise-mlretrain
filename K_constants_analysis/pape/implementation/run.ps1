param(
 [ValidateSet('plan','audit','train','export','pape-validation','verify','bundle','verify-bundle','run-validation')]
 [string]$Stage='plan',
 [string]$Config='config/study.yaml', [int]$MaxChunks=0,
 [string]$ModelRun='', [string]$ExportRun='', [string]$ValidationRun='', [string]$BundlePath=''
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe'))
if (-not (Test-Path -LiteralPath $pythonExe)) {throw 'Run setup.ps1 first'}
$arguments=@('scripts/run_pipeline.py',$Stage,'--config',$Config)
if ($MaxChunks) {$arguments+=@('--max-chunks',"$MaxChunks")}
if ($ModelRun) {$arguments+=@('--model-run',$ModelRun)}
if ($ExportRun) {$arguments+=@('--export-run',$ExportRun)}
if ($ValidationRun) {$arguments+=@('--validation-run',$ValidationRun)}
if ($BundlePath) {$arguments+=@('--bundle-path',$BundlePath)}
& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {throw "Stage failed: $Stage"}
