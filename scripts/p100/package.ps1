param(
    [string] $Destination,
    [string] $OpenSslRoot = (Join-Path $env:ProgramFiles 'OpenSSL-Win64'),
    [switch] $Zip
)

$ErrorActionPreference = 'Stop'
$source_dir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$build_dir = Join-Path $source_dir 'build-p100-rtx4070'
$cache_path = Join-Path $build_dir 'CMakeCache.txt'
if (-not (Test-Path -LiteralPath $cache_path)) { throw 'Run scripts/p100/build.ps1 first.' }
$cache = @{}
foreach ($line in Get-Content -LiteralPath $cache_path) {
    if ($line -match '^([^:#]+):[^=]+=(.*)$') { $cache[$matches[1]] = $matches[2] }
}
$required_cache = @{ CMAKE_BUILD_TYPE = 'Release'; CMAKE_CUDA_ARCHITECTURES = '60-real;89-real'; BUILD_SHARED_LIBS = 'OFF'; GGML_CUDA = 'ON'; GGML_NATIVE = 'ON'; GGML_LTO = 'ON'; GGML_CUDA_FA_QUANTS = 'all' }
foreach ($key in $required_cache.Keys) {
    if ($cache[$key] -ne $required_cache[$key]) { throw "Build cache mismatch for $key. Run scripts/p100/build.ps1 first." }
}
$cuda_root = $cache['CUDAToolkit_ROOT']
$tools = @('server', 'cli', 'bench', 'quantize', 'imatrix', 'gguf-split', 'perplexity')
$runtime_files = @{}
foreach ($tool in $tools) { $runtime_files["llama-$tool.exe"] = Join-Path $build_dir "bin\llama-$tool.exe" }
foreach ($name in @('cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll')) { $runtime_files[$name] = Join-Path $cuda_root "bin\$name" }
foreach ($name in @('libcrypto-3-x64.dll', 'libssl-3-x64.dll')) { $runtime_files[$name] = Join-Path $OpenSslRoot $name }
foreach ($path in $runtime_files.Values) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Package input was not found: $path" }
}
$source_commit = & git -C $source_dir rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot read the source commit.' }
$source_status = @(& git -C $source_dir status --porcelain --untracked-files=normal)
if ($LASTEXITCODE -ne 0) { throw 'Cannot read the source status.' }
if (-not $Destination) {
    $Destination = Join-Path $source_dir ('release\p100-first-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
}
$destination_path = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $destination_path) { throw "Destination already exists: $destination_path" }
if ($Zip -and (Test-Path -LiteralPath "$destination_path.zip")) { throw 'ZIP output already exists.' }
New-Item -ItemType Directory -Path $destination_path | Out-Null
foreach ($name in $runtime_files.Keys) { Copy-Item -LiteralPath $runtime_files[$name] -Destination (Join-Path $destination_path $name) }
foreach ($name in @('convert_hf_to_gguf.py', 'convert_lora_to_gguf.py', 'LICENSE')) { Copy-Item -LiteralPath (Join-Path $source_dir $name) -Destination $destination_path }
foreach ($directory in @('conversion', 'gguf-py', 'requirements')) {
    $input_dir = Join-Path $source_dir $directory
    foreach ($file in Get-ChildItem -LiteralPath $input_dir -File -Recurse | Where-Object { $_.FullName -notmatch '[\\/](__pycache__|\.pytest_cache)[\\/]' -and $_.Extension -ne '.pyc' }) {
        $relative_path = $file.FullName.Substring($source_dir.Length + 1)
        $output_path = Join-Path $destination_path $relative_path
        New-Item -ItemType Directory -Path (Split-Path -Parent $output_path) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $output_path
    }
}
foreach ($name in @('run-llama-p100.ps1', 'start-unsloth-p100.ps1', 'configure_unsloth.py', 'restore_unsloth.py')) { Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $destination_path }
Copy-Item -LiteralPath (Join-Path $source_dir 'docs\p100\README.md') -Destination $destination_path
Copy-Item -LiteralPath (Join-Path $source_dir 'docs\p100\validation-2026-10-04.json') -Destination $destination_path
$license_dir = Join-Path $destination_path 'licenses'
New-Item -ItemType Directory -Path $license_dir | Out-Null
Copy-Item -LiteralPath (Join-Path $cuda_root 'EULA.txt') -Destination (Join-Path $license_dir 'CUDA-EULA.txt')
Copy-Item -LiteralPath (Join-Path $OpenSslRoot 'license.txt') -Destination (Join-Path $license_dir 'OpenSSL-license.txt')
$utf8 = New-Object System.Text.UTF8Encoding($false)
$recipe_paths = @('CMakePresets.json') + @(Get-ChildItem -LiteralPath $PSScriptRoot -File | ForEach-Object { $_.FullName.Substring($source_dir.Length + 1).Replace('\', '/') })
$recipe_hashes = @($recipe_paths | Sort-Object | ForEach-Object { @{ path = $_; sha256 = (Get-FileHash -LiteralPath (Join-Path $source_dir $_) -Algorithm SHA256).Hash.ToLowerInvariant() } })
$manifest = [ordered]@{
    created_at = (Get-Date).ToString('o')
    source_commit = $source_commit
    source_dirty = ($source_status.Count -ne 0)
    source_status = $source_status
    preset = 'p100-rtx4070-release'
    cache = [ordered]@{}
    recipe_hashes = $recipe_hashes
    validation = 'Packaging only. Run the documented tests for this source revision.'
}
foreach ($key in @('CMAKE_BUILD_TYPE', 'CMAKE_CUDA_ARCHITECTURES', 'CMAKE_CUDA_COMPILER', 'CUDAToolkit_ROOT', 'GGML_NATIVE', 'GGML_LTO', 'GGML_CUDA_FA_QUANTS', 'BUILD_SHARED_LIBS', 'OPENSSL_INCLUDE_DIR')) { $manifest.cache[$key] = $cache[$key] }
[IO.File]::WriteAllText((Join-Path $destination_path 'build-manifest.json'), ($manifest | ConvertTo-Json -Depth 6) + "`n", $utf8)
$checksums = @(Get-ChildItem -LiteralPath $destination_path -File -Recurse | Sort-Object FullName | ForEach-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $_.FullName.Substring($destination_path.Length + 1).Replace('\', '/') })
[IO.File]::WriteAllText((Join-Path $destination_path 'SHA256SUMS.txt'), ($checksums -join "`n") + "`n", $utf8)
if ($Zip) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory($destination_path, "$destination_path.zip")
    $archive_hash = (Get-FileHash -LiteralPath "$destination_path.zip" -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText("$destination_path.zip.sha256", $archive_hash + '  ' + [IO.Path]::GetFileName("$destination_path.zip") + "`n", $utf8)
    Write-Host "ZIP: $destination_path.zip"
}
Write-Host "Package: $destination_path"
