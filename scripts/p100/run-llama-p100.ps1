param(
    [ValidateSet('server', 'cli', 'bench', 'quantize', 'imatrix', 'gguf-split', 'perplexity')]
    [string] $Tool = 'server',
    [switch] $P100Only,
    [string] $BinaryDir,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $LlamaArguments
)

$ErrorActionPreference = 'Stop'
if (-not $BinaryDir) {
    $BinaryDir = if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "llama-$Tool.exe")) { $PSScriptRoot } else { Join-Path $PSScriptRoot '..\..\release\p100-first' }
}
$BinaryDir = (Resolve-Path -LiteralPath $BinaryDir).Path
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
$env:LLAMA_ARG_MAIN_GPU = '0'
$env:LLAMA_ARG_SPLIT_MODE = 'layer'
$split = @('3') + @($gpu_ids | Select-Object -Skip 1 | ForEach-Object { '2' })
$env:LLAMA_ARG_TENSOR_SPLIT = $split -join ','
$env:PATH = "$BinaryDir;$env:PATH"
$env:LLAMA_ARG_THREADS = '6'
$env:LLAMA_ARG_THREADS_BATCH = '6'
$binary_path = Join-Path $BinaryDir "llama-$Tool.exe"
if (-not (Test-Path -LiteralPath $binary_path)) { throw "Binary was not found: $binary_path" }
if ($Tool -eq 'bench') {
    $bench_split = $split -join '/'
    $LlamaArguments = @('--threads', '6', '--split-mode', 'layer', '--main-gpu', '0', '--tensor-split', $bench_split) + $LlamaArguments
}
& $binary_path @LlamaArguments
exit $LASTEXITCODE
