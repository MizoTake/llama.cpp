param(
    [ValidateRange(1, 128)]
    [int] $Parallel = 8,
    [switch] $Tests,
    [string] $CudaRoot = (Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v12.9')
)

$ErrorActionPreference = 'Stop'
function invoke_logged {
    param([string] $Command, [string[]] $Arguments, [string] $LogPath)
    $command_path = (Get-Command $Command -ErrorAction Stop).Source
    $ErrorActionPreference = 'Continue'
    & $command_path @Arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $LogPath -ErrorAction Stop
    if ($LASTEXITCODE -ne 0) { throw "$Command failed with exit code $LASTEXITCODE." }
}

$source_dir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$vswhere_path = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere_path)) { throw 'Visual Studio Installer was not found.' }
$vs_path = & $vswhere_path -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs_path) { throw 'Visual Studio C++ tools were not found.' }
$vcvars_path = Join-Path $vs_path 'VC\Auxiliary\Build\vcvars64.bat'
$compiler_env = & cmd.exe /d /c "`"$vcvars_path`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'Visual Studio environment setup failed.' }
foreach ($entry in $compiler_env) {
    if ($entry -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:VSLANG = '1033'
$ninja_dir = Join-Path $vs_path 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
$env:PATH = "$ninja_dir;$env:PATH"
$cuda_root = (Resolve-Path -LiteralPath $CudaRoot).Path
$nvcc_path = Join-Path $cuda_root 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc_path)) { throw 'CUDA nvcc was not found.' }
$nvcc_version = & $nvcc_path --version
if ($LASTEXITCODE -ne 0 -or ($nvcc_version -join ' ') -notmatch 'release 12\.9,') {
    throw 'This build profile requires CUDA Toolkit 12.9 for Pascal support.'
}
$build_dir = Join-Path $source_dir 'build-p100-rtx4070'
$log_dir = Join-Path $build_dir 'logs'
New-Item -ItemType Directory -Path $log_dir -Force | Out-Null
$tool_targets = @('llama-server', 'llama-cli', 'llama-bench', 'llama-quantize', 'llama-imatrix', 'llama-gguf-split', 'llama-perplexity')
$test_targets = @('test-tokenizer-0', 'test-unicode', 'test-sampling', 'test-reasoning-budget', 'test-grammar-parser', 'test-grammar-integration', 'test-llama-grammar', 'test-chat', 'test-json-schema-to-grammar', 'test-chat-peg-parser', 'test-jinja', 'test-json-schema', 'test-chat-auto-parser', 'test-chat-template', 'test-log', 'test-peg-parser', 'test-arg-parser', 'test-model-resolution', 'test-quantize-fns', 'test-rope')
Push-Location $source_dir
try {
    invoke_logged cmake @('--preset', 'p100-rtx4070-release', "-DCMAKE_CUDA_COMPILER=$nvcc_path", "-DCUDAToolkit_ROOT=$cuda_root") (Join-Path $log_dir 'managed-configure.log')
    $targets = $tool_targets
    if ($Tests) { $targets += $test_targets + @('test-backend-ops') }
    invoke_logged cmake (@('--build', $build_dir, '--parallel', [string] $Parallel, '--target') + $targets) (Join-Path $log_dir 'managed-build.log')
    if ($Tests) {
        $test_regex = '^(' + ($test_targets -join '|') + ')(-|$)'
        invoke_logged ctest @('--test-dir', $build_dir, '--output-on-failure', '--parallel', '4', '-R', $test_regex) (Join-Path $log_dir 'managed-ctest.log')
    }
} finally {
    Pop-Location
}
Write-Host "Build output: $build_dir\bin"
