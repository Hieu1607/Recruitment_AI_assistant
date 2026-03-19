param(
    [ValidateSet("up", "down", "logs", "migrate")]
    [string]$Action = "up"
)

$composeCmd = "docker compose"

switch ($Action) {
    "up" {
        Write-Host "Starting stack with build..."
        Invoke-Expression "$composeCmd up -d --build"
    }
    "down" {
        Write-Host "Stopping stack..."
        Invoke-Expression "$composeCmd down"
    }
    "logs" {
        Write-Host "Streaming logs..."
        Invoke-Expression "$composeCmd logs -f"
    }
    "migrate" {
        Write-Host "Applying migrations in API container..."
        Invoke-Expression "$composeCmd exec api python -m alembic upgrade head"
    }
}
