# run_all.ps1 -- launch N sweep shards in parallel on Windows.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Shards 4 -Stage 1
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Stage 2
#
# Each shard is its own python process in its own window: you can watch
# progress and kill one without killing the rest. They write to separate
# JSONL files which analyze/plot glob back together.

param(
    [int]$Shards = 4,
    [int]$Stage = 1,
    [string]$Config = "configs/sweep_grid.yaml",
    [string]$Winners = "configs/winners.yaml",
    [double]$MemoryFraction = 0
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path results, logs, figures, tables | Out-Null

switch ($Stage) {
    1 { $stageArgs = @("--stage1");            $prefix = "stage1" }
    2 {
        if (-not (Test-Path $Winners)) {
            throw "$Winners not found. Run stage 1, then 'analyze --emit-winners' first."
        }
        $stageArgs = @("--stage2", $Winners);  $prefix = "stage2"
    }
    default { $stageArgs = @();                $prefix = "full" }
}

if ($MemoryFraction -gt 0) {
    $stageArgs += @("--memory-fraction", "$MemoryFraction")
}

Write-Host "Launching $Shards shard(s), stage $Stage, config $Config" -ForegroundColor Cyan

$procs = @()
for ($i = 0; $i -lt $Shards; $i++) {
    $out = "results/${prefix}_$i.jsonl"
    $argList = @("-m", "sharpbench", "sweep", "--config", $Config,
                 "--shard", "$i/$Shards", "--out", $out) + $stageArgs
    Write-Host "  shard $i -> $out"
    $procs += Start-Process -FilePath "python" -ArgumentList $argList -PassThru
    Start-Sleep -Milliseconds 800
}

Write-Host "`nWaiting for $($procs.Count) shard(s)..." -ForegroundColor Cyan
$procs | Wait-Process

Write-Host "`nDone. Next:" -ForegroundColor Green
if ($Stage -eq 1) {
    Write-Host "  python -m sharpbench analyze --results `"results/stage1_*.jsonl`" --emit-winners $Winners"
} else {
    Write-Host "  python -m sharpbench analyze --results `"results/stage2_*.jsonl`" --latex tables/main.tex --csv tables/main.csv"
    Write-Host "  python -m sharpbench plot --results `"results/stage1_*.jsonl`" `"results/stage2_*.jsonl`""
}
