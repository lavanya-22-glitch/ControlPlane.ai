# ControlPlane.ai - Automated PowerShell Benchmark Runner
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  ControlPlane.ai - Automated Test & Analytics Suite" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if proxy is running
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 2 -ErrorAction Stop
    if ($health.status -eq "healthy") {
        Write-Host "[OK] Connected to ControlPlane proxy at http://localhost:8000" -ForegroundColor Green
    }
} catch {
    Write-Host "[!] ControlPlane proxy is not currently running on port 8000." -ForegroundColor Yellow
    Write-Host "    Starting server in background..." -ForegroundColor Yellow
    Start-Process -NoNewWindow python -ArgumentList "-m controlplane.main"
    Start-Sleep -Seconds 2
}

# Run the benchmark engine
python scripts/run_benchmarks.py

# Open or display generated analytics
Write-Host "`nAnalytics report saved to: data/benchmark_analytics.md and data/benchmark_analytics.json" -ForegroundColor Green
