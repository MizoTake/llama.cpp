param(
    [switch] $P100Only,
    [string] $RuntimeDir,
    [switch] $DisableThermalSwitch,
    [string] $ThermalConfig = (Join-Path $env:USERPROFILE '.unsloth\studio\p100-thermal\config.json'),
    [ValidateRange(1, 65536)] [int] $BatchSize = 512,
    [ValidateRange(1, 65536)] [int] $UBatchSize = 512
)

$ErrorActionPreference = 'Stop'
if ($UBatchSize -gt $BatchSize) { throw 'UBatchSize must not exceed BatchSize.' }
if (-not $RuntimeDir) {
    $RuntimeDir = if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'llama-server.exe')) { $PSScriptRoot } else { Join-Path $PSScriptRoot '..\..\release\p100-first' }
}
$RuntimeDir = (Resolve-Path -LiteralPath $RuntimeDir).Path
if (-not (Test-Path -LiteralPath (Join-Path $RuntimeDir 'llama-server.exe'))) { throw 'Package llama-server.exe was not found.' }
$desktop_path = Join-Path $env:LOCALAPPDATA 'Unsloth Studio (Desktop)\unsloth-studio.exe'
if (-not (Test-Path -LiteralPath $desktop_path)) {
    throw "Unsloth Desktop was not found: $desktop_path"
}
if (Get-Process -Name unsloth-studio -ErrorAction SilentlyContinue) {
    throw 'Close Unsloth Desktop before using this launcher. The running app cannot inherit the new settings.'
}
$gpus = @(& nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader,nounits | ConvertFrom-Csv -Header index,name,cc | Where-Object { $_.cc.Trim() -in @('6.0', '8.9') })
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed.' }
$p100 = $gpus | Where-Object { $_.name.Trim() -match '^Tesla P100' } | Select-Object -First 1
if (-not $p100) { throw 'No Tesla P100 was found.' }
$gpu_ids = @($p100.index.Trim())
if (-not $P100Only) {
    $gpu_ids += @($gpus | Where-Object { $_.index.Trim() -ne $p100.index.Trim() } | ForEach-Object { $_.index.Trim() })
}
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = $gpu_ids -join ','
$env:LLAMA_ARG_MAIN_GPU = [string] [array]::IndexOf($gpu_ids, $p100.index.Trim())
$env:LLAMA_ARG_SPLIT_MODE = 'layer'
$env:LLAMA_ARG_TENSOR_SPLIT = ($gpu_ids | ForEach-Object { if ($_ -eq $p100.index.Trim()) { '3' } else { '2' } }) -join ','
$env:LLAMA_ARG_THREADS = '6'
$env:LLAMA_ARG_THREADS_BATCH = '6'
$env:LLAMA_ARG_BATCH = [string] $BatchSize
$env:LLAMA_ARG_UBATCH = [string] $UBatchSize
$env:LLAMA_ARG_FLASH_ATTN = 'on'
$env:LLAMA_ARG_CACHE_TYPE_K = 'f16'
$env:LLAMA_ARG_CACHE_TYPE_V = 'f16'
$env:UNSLOTH_LLAMA_CPP_PATH = $RuntimeDir
$env:UNSLOTH_LLAMA_CPP_SCRIPTS_DIR = $RuntimeDir
$env:LLAMA_SERVER_PATH = Join-Path $RuntimeDir 'llama-server.exe'
$env:PATH = "$RuntimeDir;$env:PATH"
Write-Host "llama.cpp: $RuntimeDir"
Write-Host "CUDA_VISIBLE_DEVICES: $env:CUDA_VISIBLE_DEVICES"
Write-Host "Primary GPU: $($p100.name.Trim()); tensor split: $env:LLAMA_ARG_TENSOR_SPLIT"
$monitor_path = Join-Path $RuntimeDir 'thermal_unsloth.py'
$pythonw_path = Join-Path $env:USERPROFILE '.unsloth\studio\unsloth_studio\Scripts\pythonw.exe'
$thermal_enabled = -not $DisableThermalSwitch -and -not $P100Only -and (Test-Path -LiteralPath $ThermalConfig)
if ($thermal_enabled -and (-not (Test-Path -LiteralPath $monitor_path) -or -not (Test-Path -LiteralPath $pythonw_path))) {
    throw 'Thermal switching requires the packaged helper and Unsloth pythonw.exe.'
}
$desktop_process = Start-Process -FilePath $desktop_path -WindowStyle Hidden -PassThru
if ($thermal_enabled) {
    if ($monitor_path.Contains('"') -or $ThermalConfig.Contains('"')) { throw 'Invalid monitor path.' }
    $monitor_arguments = @('-X', 'utf8', ('"' + $monitor_path + '"'), '--config', ('"' + $ThermalConfig + '"'), '--desktop-pid', [string] $desktop_process.Id)
    Start-Process -FilePath $pythonw_path -ArgumentList $monitor_arguments -WindowStyle Hidden | Out-Null
    Write-Host "Thermal monitor: $ThermalConfig"
}
