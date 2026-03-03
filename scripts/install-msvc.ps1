param(
    [string]$InstallPath  # 默认不设置
)

$ErrorActionPreference = "Stop"

Write-Host "Checking for existing MSVC installation..."

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

if (Test-Path $vswhere) {
    $existing = & $vswhere `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath

    if ($existing) {
        Write-Host "MSVC already installed at: $existing"
        exit 0
    }
}

Write-Host "MSVC not found. Installing Build Tools..."

$tmpDir = Join-Path $env:TEMP "vsbt"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$installer = Join-Path $tmpDir "vs_BuildTools.exe"

Invoke-WebRequest `
    -Uri "https://aka.ms/vs/17/release/vs_BuildTools.exe" `
    -OutFile $installer

$args = @(
    "--quiet",
    "--wait",
    "--norestart",
    "--nocache",
    "--add", "Microsoft.VisualStudio.Workload.VCTools",
    "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041"
)

# 只有用户显式传入 InstallPath 才加
if ($InstallPath) {
    $args += @("--installPath", $InstallPath)
}

$process = Start-Process `
    -FilePath $installer `
    -ArgumentList $args `
    -PassThru `
    -Wait

if ($process.ExitCode -ne 0) {
    throw "Build Tools installation failed with exit code $($process.ExitCode)"
}

Write-Host "MSVC Build Tools installed successfully."