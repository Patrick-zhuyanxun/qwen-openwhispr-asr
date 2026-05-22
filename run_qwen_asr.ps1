$ErrorActionPreference = "Stop"

[CmdletBinding()]
param(
    [Alias("model-size")]
    [ValidateSet("0.6B", "1.7B")]
    [string]$ModelSize = $(if ($env:QWEN_ASR_MODEL_SIZE) { $env:QWEN_ASR_MODEL_SIZE } else { "0.6B" }),

    [Alias("host")]
    [string]$HostName = $(if ($env:QWEN_ASR_HOST) { $env:QWEN_ASR_HOST } else { "127.0.0.1" }),

    [Alias("port")]
    [int]$Port = $(if ($env:QWEN_ASR_PORT) { [int]$env:QWEN_ASR_PORT } else { 8179 })
)

$AppDir = Split-Path -Parent $PSCommandPath
$LogDir = Join-Path $AppDir ".logs"
$PidFile = Join-Path $LogDir "qwen-openwhispr-asr.pid"
$StdoutLog = Join-Path $LogDir "qwen-openwhispr-asr.out.log"
$StderrLog = Join-Path $LogDir "qwen-openwhispr-asr.err.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $AppDir

function Stop-ProcessTree {
    param([int]$TargetPid)

    if ($TargetPid -le 0 -or $TargetPid -eq $PID) {
        return
    }

    $existing = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        return
    }

    & taskkill.exe /PID $TargetPid /T /F | Out-Null
}

function Stop-OldQwenAsr {
    Write-Host "Stopping old Qwen ASR process if it exists..."

    if (Test-Path $PidFile) {
        $rawPid = (Get-Content $PidFile -Raw).Trim()
        $parsedPid = 0
        if ([int]::TryParse($rawPid, [ref]$parsedPid)) {
            Stop-ProcessTree -TargetPid $parsedPid
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    $normalizedAppDir = $AppDir.Replace("\", "/")
    $processes = Get-CimInstance Win32_Process | Where-Object {
        if (-not $_.CommandLine -or $_.ProcessId -eq $PID) {
            $false
        } else {
            $commandLine = $_.CommandLine.Replace("\", "/")
            $isQwenAsrMain = $commandLine -match "main\.py" -and (
                $commandLine.Contains($normalizedAppDir) -or
                $commandLine -match "--port\s+$Port\b"
            )
            $isQwenAsrMain
        }
    }

    foreach ($process in $processes) {
        Stop-ProcessTree -TargetPid ([int]$process.ProcessId)
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found in PATH. Install it first: winget install --id=astral-sh.uv -e"
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg was not found in PATH. Install it first: winget install --id=Gyan.FFmpeg -e"
}

Stop-OldQwenAsr

$arguments = @(
    "run",
    "python",
    "main.py",
    "--warmup",
    "--host",
    $HostName,
    "--port",
    [string]$Port,
    "--model-size",
    $ModelSize
)

Write-Host "Starting qwen-openwhispr-asr..."
$process = Start-Process `
    -FilePath "uv" `
    -ArgumentList $arguments `
    -WorkingDirectory $AppDir `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $PidFile -Value $process.Id

$healthUrl = "http://${HostName}:${Port}/health"
Write-Host "Waiting for $healthUrl ..."

for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 1

    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        $health | ConvertTo-Json -Compress
        Write-Host "OpenWhispr Server URL: http://${HostName}:${Port}/v1"
        Write-Host "Logs:"
        Write-Host "  $StdoutLog"
        Write-Host "  $StderrLog"
        exit 0
    } catch {
        $process.Refresh()
        if ($process.HasExited) {
            break
        }
    }
}

Write-Error "Server did not become healthy within 120 seconds. Check logs: $StderrLog"
exit 1
