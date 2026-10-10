param([switch] $OpenLmStudio, [switch] $Stop)

$ErrorActionPreference = 'Stop'
$memory_dir = Join-Path $env:USERPROFILE '.llama.cpp\desktop'
$script_path = Join-Path $memory_dir 'runtime\desktop_service.py'
$python_path = Join-Path $env:USERPROFILE '.unsloth\studio\unsloth_studio\Scripts\pythonw.exe'
if ($Stop) {
    New-Item -ItemType File -Path (Join-Path $memory_dir 'stop') -Force | Out-Null
    $running = @(Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($script_path) })
    foreach ($process in $running) {
        Wait-Process -Id $process.ProcessId -Timeout 30 -ErrorAction SilentlyContinue
        $remaining = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $process.ProcessId)
        if ($remaining -and $remaining.CommandLine -and $remaining.CommandLine.Contains($script_path)) { throw 'The service is still active. Wait for it to finish before updating.' }
    }
    return
}
Remove-Item -LiteralPath (Join-Path $memory_dir 'stop') -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $script_path) -or -not (Test-Path -LiteralPath $python_path)) { throw 'Install the common Desktop helpers first.' }
$running = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($script_path) }
if (-not $running) {
    Start-Process -FilePath $python_path -ArgumentList @('-X', 'utf8', ('"' + $script_path + '"')) -WindowStyle Hidden | Out-Null
}
if ($OpenLmStudio) {
    $app_path = Join-Path $env:LOCALAPPDATA 'Programs\LM Studio\LM Studio.exe'
    Start-Process -FilePath $app_path -WindowStyle Hidden | Out-Null
}
