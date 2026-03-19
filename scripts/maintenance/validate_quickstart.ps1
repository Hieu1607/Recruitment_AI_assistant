param(
    [string]$BackendEnvPath = "backend/.env",
    [string]$FrontendEnvPath = "frontend/.env"
)

$ErrorActionPreference = "Stop"

function Read-EnvFile {
    param([Parameter(Mandatory=$true)][string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Missing environment file: $Path"
    }

    $values = @{}
    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) { continue }
        if ($trimmed.StartsWith("#")) { continue }
        $pair = $trimmed -split "=", 2
        if ($pair.Count -ne 2) { continue }
        $key = $pair[0].Trim()
        $value = $pair[1].Trim().Trim('"').Trim("'")
        $values[$key] = $value
    }
    return $values
}

function Assert-RequiredVars {
    param(
        [Parameter(Mandatory=$true)][hashtable]$EnvValues,
        [Parameter(Mandatory=$true)][string[]]$RequiredKeys,
        [Parameter(Mandatory=$true)][string]$Context
    )

    $missing = @()
    foreach ($key in $RequiredKeys) {
        if (-not $EnvValues.ContainsKey($key) -or [string]::IsNullOrWhiteSpace($EnvValues[$key])) {
            $missing += $key
        }
    }

    if ($missing.Count -gt 0) {
        throw "$Context is missing required keys: $($missing -join ', ')"
    }
}

Write-Host "[1/4] Verifying docker compose services"
$composeStatus = docker compose ps --format json | ConvertFrom-Json
if (-not $composeStatus) {
    throw "No docker compose services detected. Run scripts/maintenance/dev.ps1 -Action up first."
}

$requiredServices = @("api", "worker", "frontend", "postgres", "minio")
$missingServices = @()
$stoppedServices = @()

foreach ($service in $requiredServices) {
    $entry = $composeStatus | Where-Object { $_.Service -eq $service }
    if (-not $entry) {
        $missingServices += $service
        continue
    }

    if ($entry.State -ne "running") {
        $stoppedServices += "$service($($entry.State))"
    }
}

if ($missingServices.Count -gt 0) {
    throw "Missing services in compose output: $($missingServices -join ', ')"
}

if ($stoppedServices.Count -gt 0) {
    throw "Services not running: $($stoppedServices -join ', ')"
}

Write-Host "[2/4] Validating environment files"
$backendEnv = Read-EnvFile -Path $BackendEnvPath
$frontendEnv = Read-EnvFile -Path $FrontendEnvPath

$requiredBackendKeys = @(
    "DATABASE_URL",
    "LLM_PROVIDER",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_BUCKET",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD"
)
Assert-RequiredVars -EnvValues $backendEnv -RequiredKeys $requiredBackendKeys -Context "backend/.env"
Assert-RequiredVars -EnvValues $frontendEnv -RequiredKeys @("VITE_API_BASE_URL") -Context "frontend/.env"

Write-Host "[3/4] Checking API health endpoint"
$health = Invoke-RestMethod -Method Get -Uri "http://localhost:8000/health"
if ($health.status -ne "ok") {
    throw "Health endpoint returned unexpected payload"
}

Write-Host "[4/4] Checking API metrics endpoint"
$metrics = Invoke-RestMethod -Method Get -Uri "http://localhost:8000/metrics"
if (-not $metrics.PSObject.Properties.Name.Contains("requestCount")) {
    throw "Metrics endpoint missing requestCount"
}

Write-Host "Quickstart validation passed."
