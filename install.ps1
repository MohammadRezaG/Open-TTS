$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$EnvironmentPath = Join-Path $ProjectRoot ".venv"
$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"

function Invoke-Python {
    param([string[]]$PythonArgs)

    & $EnvironmentPython @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function New-ProjectEnvironment {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $Launcher) {
        throw "Install Python 3.12 or 3.13 with the Python Launcher, then run this installer again."
    }

    foreach ($Version in @("3.13", "3.12")) {
        & $Launcher.Source "-$Version" -m venv $EnvironmentPath 2>$null
        if ($LASTEXITCODE -eq 0) {
            return
        }
    }

    throw "Could not create .venv. Install Python 3.12 or 3.13 and try again."
}

if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    New-ProjectEnvironment
}

$VersionText = & $EnvironmentPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $VersionText -notin @("3.12", "3.13")) {
    throw ".venv must use Python 3.12 or 3.13; found '$VersionText'. Remove or rename .venv and run this installer again."
}

Write-Host "Installing the Open TTS app into .venv (Python $VersionText)..."
Invoke-Python @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Python @(
    "-m", "pip", "install",
    "-e", (Join-Path $ProjectRoot "common"),
    "-e", (Join-Path $ProjectRoot "engine"),
    "-e", (Join-Path $ProjectRoot "api"),
    "-e", (Join-Path $ProjectRoot "webui")
)

$NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$DefaultChoice = if ($NvidiaSmi) { "2" } else { "1" }
Write-Host ""
Write-Host "Choose the supported PyTorch device backend:"
Write-Host "  1. CPU"
Write-Host "  2. NVIDIA GPU (CUDA 13.0)"
Write-Host "  3. NVIDIA GPU (CUDA 12.6, for a driver that cannot use CUDA 13.0)"
Write-Host ""
$Choice = Read-Host "Selection [default $DefaultChoice]"
if ([string]::IsNullOrWhiteSpace($Choice)) {
    $Choice = $DefaultChoice
}

switch ($Choice) {
    "1" {
        $IndexUrl = "https://download.pytorch.org/whl/cpu"
        $DeviceName = "CPU"
    }
    "2" {
        $IndexUrl = "https://download.pytorch.org/whl/cu130"
        $DeviceName = "NVIDIA GPU (CUDA 13.0)"
    }
    "3" {
        $IndexUrl = "https://download.pytorch.org/whl/cu126"
        $DeviceName = "NVIDIA GPU (CUDA 12.6)"
    }
    default {
        throw "Choose 1, 2, or 3. Run install.ps1 again to retry."
    }
}

Write-Host "Installing the $DeviceName PyTorch packages..."
Invoke-Python @("-m", "pip", "install", "--upgrade", "torch", "torchaudio", "--index-url", $IndexUrl)

$CudaAvailable = & $EnvironmentPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Could not verify the PyTorch installation."
}

if ($Choice -eq "1") {
    Write-Host "PyTorch CPU backend is ready."
} elseif ($CudaAvailable -eq "True") {
    $GpuName = & $EnvironmentPython -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "PyTorch GPU backend is ready: $GpuName"
} else {
    Write-Warning "The CUDA packages installed, but PyTorch cannot access the GPU. Check the NVIDIA driver, then rerun install.ps1 and try the other CUDA option."
}

$KittenDefault = if ($NvidiaSmi) { "2" } else { "1" }
Write-Host ""
Write-Host "Choose the Kitten English inference backend:" 
Write-Host "  1. CPU (ONNX Runtime)"
Write-Host "  2. NVIDIA GPU (ONNX Runtime GPU)"
Write-Host "  3. Skip Kitten for now"
$KittenChoice = Read-Host "Selection [default $KittenDefault]"
if ([string]::IsNullOrWhiteSpace($KittenChoice)) {
    $KittenChoice = $KittenDefault
}
if ($KittenChoice -notin @("1", "2", "3")) {
    throw "Choose 1, 2, or 3. Run install.ps1 again to retry."
}
if ($KittenChoice -ne "3") {
    Invoke-Python @("-m", "pip", "install", "-e", "$ProjectRoot\engine[kitten]")
    if ($KittenChoice -eq "2") {
        Invoke-Python @("-m", "pip", "uninstall", "-y", "onnxruntime")
        Invoke-Python @("-m", "pip", "install", "onnxruntime-gpu")
    }
}

Write-Host "Restart the API and UI to use the selected device."
