#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$Source,
    [switch]$InstallOnly
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$architecture = $env:PROCESSOR_ARCHITEW6432
if (-not $architecture) {
    $architecture = $env:PROCESSOR_ARCHITECTURE
}
if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or $architecture -ne 'AMD64') {
    throw 'This STT Lab installer requires Windows x64.'
}
if (-not $env:LOCALAPPDATA -and -not $env:STT_LAB_INSTALL_DIR) {
    throw 'LOCALAPPDATA is unavailable. Run the installer from your Windows user session.'
}

$release = 'v0.1.0'
$uvVersion = '0.13.7'
$uvArchiveName = 'uv-x86_64-pc-windows-msvc.zip'
$uvDownload = "https://releases.astral.sh/github/uv/releases/download/$uvVersion/$uvArchiveName"
# Published with the uv 0.13.7 assets on GitHub.
$uvArchiveSha256 = '0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021'
$sourceDownload = "https://github.com/christovnino-a11y/stt-lab/archive/refs/tags/$release.zip"
$appFiles = @('app.py', 'pyproject.toml', 'uv.lock')

if ($env:STT_LAB_INSTALL_DIR) {
    $root = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($env:STT_LAB_INSTALL_DIR)
} else {
    $root = Join-Path $env:LOCALAPPDATA 'STT Lab'
}
$uvDir = Join-Path $root "uv\$uvVersion"
$uv = Join-Path $uvDir 'uv.exe'
$releasesDir = Join-Path $root 'releases'
$launcher = Join-Path $root 'STT Lab.cmd'

function Get-ReleaseFile {
    param([string]$Uri, [string]$Destination)

    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination -TimeoutSec 180
}

function Invoke-Uv {
    & $uv @args
    if ($LASTEXITCODE -ne 0) {
        throw "uv $($args[0]) failed (exit code $LASTEXITCODE). Review the error above and rerun the installer."
    }
}

[IO.Directory]::CreateDirectory($root) | Out-Null
try {
    $installLock = [IO.File]::Open(
        (Join-Path $root 'install.lock'), 'OpenOrCreate', 'ReadWrite', 'None'
    )
} catch [IO.IOException] {
    throw "Could not lock the STT Lab installation. Close any other STT Lab installer and retry. $($_.Exception.Message)"
}

$staging = Join-Path $root ('.install-' + [Guid]::NewGuid().ToString('N'))
$originalTls = [Net.ServicePointManager]::SecurityProtocol
$originalPythonDir = $env:UV_PYTHON_INSTALL_DIR
$originalPythonRegistry = $env:UV_PYTHON_INSTALL_REGISTRY
$originalProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
$originalCacheDir = $env:UV_CACHE_DIR

try {
    [IO.Directory]::CreateDirectory($staging) | Out-Null
    if ($PSVersionTable.PSEdition -eq 'Desktop') {
        [Net.ServicePointManager]::SecurityProtocol = $originalTls -bor [Net.SecurityProtocolType]::Tls12
    }

    if ($PSBoundParameters.ContainsKey('Source')) {
        $sourceItem = Get-Item -LiteralPath $Source
        if ($sourceItem -isnot [IO.DirectoryInfo]) {
            throw 'Source must be a directory containing app.py, pyproject.toml and uv.lock.'
        }
        $sourceDir = $sourceItem.FullName
    } else {
        Write-Host "Downloading STT Lab $release..."
        $sourceZip = Join-Path $staging 'stt-lab.zip'
        $extractedSource = Join-Path $staging 'source'
        Get-ReleaseFile $sourceDownload $sourceZip
        Expand-Archive -LiteralPath $sourceZip -DestinationPath $extractedSource
        $archiveDirectories = @(Get-ChildItem -LiteralPath $extractedSource -Directory)
        if ($archiveDirectories.Count -ne 1) {
            throw 'The STT Lab source archive has an unexpected directory layout.'
        }
        $sourceDir = $archiveDirectories[0].FullName
    }

    $stagedFiles = Join-Path $staging 'app'
    [IO.Directory]::CreateDirectory($stagedFiles) | Out-Null
    foreach ($appFile in $appFiles) {
        $inputFile = Join-Path $sourceDir $appFile
        if (-not (Test-Path -LiteralPath $inputFile -PathType Leaf)) {
            throw "STT Lab source is missing $appFile in '$sourceDir'."
        }
        Copy-Item -LiteralPath $inputFile -Destination (Join-Path $stagedFiles $appFile)
    }

    # Keep the previous installation usable until the new source and environment are ready.
    $fileHashes = foreach ($appFile in $appFiles) {
        (Get-FileHash -LiteralPath (Join-Path $stagedFiles $appFile) -Algorithm SHA256).Hash
    }
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $hashBytes = $sha256.ComputeHash([Text.Encoding]::ASCII.GetBytes(($fileHashes -join ' ')))
    } finally {
        $sha256.Dispose()
    }
    $sourceHash = [BitConverter]::ToString($hashBytes, 0, 8).Replace('-', '').ToLowerInvariant()
    $releaseFolder = "$release-$sourceHash"
    $releaseDir = Join-Path $releasesDir $releaseFolder

    if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) {
        Write-Host "Downloading uv $uvVersion for Windows x64..."
        $uvZip = Join-Path $staging $uvArchiveName
        Get-ReleaseFile $uvDownload $uvZip
        if ((Get-FileHash -LiteralPath $uvZip -Algorithm SHA256).Hash -ne $uvArchiveSha256) {
            throw 'The downloaded uv archive failed SHA-256 verification.'
        }
        $extractedUv = Join-Path $staging 'uv'
        Expand-Archive -LiteralPath $uvZip -DestinationPath $extractedUv
        $uvExecutables = @(Get-ChildItem -LiteralPath $extractedUv -Filter 'uv.exe' -File -Recurse)
        if ($uvExecutables.Count -ne 1) {
            throw 'The uv archive did not contain the expected uv.exe.'
        }
        [IO.Directory]::CreateDirectory($uvDir) | Out-Null
        [IO.File]::Move($uvExecutables[0].FullName, $uv)
    }

    [IO.Directory]::CreateDirectory($releasesDir) | Out-Null
    if (-not (Test-Path -LiteralPath $releaseDir -PathType Container)) {
        [IO.Directory]::Move($stagedFiles, $releaseDir)
    }

    $env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
    $env:UV_PYTHON_INSTALL_REGISTRY = '0'
    $env:UV_PROJECT_ENVIRONMENT = '.venv'
    $env:UV_CACHE_DIR = Join-Path $root 'cache'

    Push-Location -LiteralPath $releaseDir
    try {
        Write-Host 'Installing Python 3.12 and STT Lab dependencies...'
        Invoke-Uv python install 3.12 --no-bin --no-registry
        Invoke-Uv sync --locked --python 3.12 --no-dev --managed-python
    } finally {
        Pop-Location
    }

    $launcherContents = @"
@echo off
setlocal DisableDelayedExpansion
set "UV_PYTHON_INSTALL_DIR=%~dp0python"
set "UV_PYTHON_INSTALL_REGISTRY=0"
set "UV_PROJECT_ENVIRONMENT=.venv"
set "UV_CACHE_DIR=%~dp0cache"
pushd "%~dp0releases\$releaseFolder"
if errorlevel 1 exit /b 1
"%~dp0uv\$uvVersion\uv.exe" run --locked --python 3.12 --no-dev --managed-python python app.py --open-browser %*
set "appExit=%errorlevel%"
popd
exit /b %appExit%
"@
    $stagedLauncher = Join-Path $staging 'STT Lab.cmd'
    Set-Content -LiteralPath $stagedLauncher -Value $launcherContents -Encoding Ascii
    if (Test-Path -LiteralPath $launcher -PathType Leaf) {
        [IO.File]::Replace($stagedLauncher, $launcher, (Join-Path $staging 'launcher.previous'))
    } else {
        [IO.File]::Move($stagedLauncher, $launcher)
    }
} finally {
    $env:UV_PYTHON_INSTALL_DIR = $originalPythonDir
    $env:UV_PYTHON_INSTALL_REGISTRY = $originalPythonRegistry
    $env:UV_PROJECT_ENVIRONMENT = $originalProjectEnvironment
    $env:UV_CACHE_DIR = $originalCacheDir
    [Net.ServicePointManager]::SecurityProtocol = $originalTls
    $installLock.Dispose()
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "STT Lab $release installed."
Write-Host 'To start it again, run this in PowerShell:'
if ($env:STT_LAB_INSTALL_DIR) {
    $quotedLauncher = $launcher.Replace("'", "''")
    Write-Host "& '$quotedLauncher'"
} else {
    Write-Host '& "$env:LOCALAPPDATA\STT Lab\STT Lab.cmd"'
}
if (-not $InstallOnly) {
    & $launcher
    if ($LASTEXITCODE -ne 0) {
        throw "STT Lab exited with code $LASTEXITCODE. Review the error above."
    }
}
