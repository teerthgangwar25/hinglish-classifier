# run_ablations.ps1  (v2)
# Runs the HingBERT ablation runs in time-boxed chunks, with a cool-down pause between chunks.
# Run from the repo root with the venv active.
#
# One-time per terminal window, so Windows allows the script (no admin needed, affects only this window):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# Examples
#   .\run_ablations.ps1                                   # stage 1: all 5 configs, seed 1
#   .\run_ablations.ps1 -ChunkMinutes 15 -CoolToC 55      # shorter chunks, cooler target
#   .\run_ablations.ps1 -Only roman_train,translit -Seeds 2,3
#
# Safe to stop and re-run: a run that already finished is skipped (its best\config.json exists).
# A run in progress is never interrupted, so a chunk can overrun by up to one run (max ~12 min).

param(
    [int]$ChunkMinutes = 40,        # stop starting new runs after this many minutes, then cool down
    [int]$CoolToC = 73,             # cool down until the GPU is at or below this temperature (Celsius)
    [int]$MaxCoolMinutes = 5,      # but never wait longer than this
    [string[]]$Seeds = @("1"),
    [string[]]$Only = @()           # optional filter: norm, translit, roman_train, len64, len256
)

# "powershell -File script.ps1 -Seeds 2,3" delivers "2,3" as ONE string, so split on commas ourselves.
$SeedList = @($Seeds | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
$OnlyList = @($Only | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })

$configs = @(
    @{ key = "norm";        args = "--norm on";            suffix = "len128_normon_scriptasis" },
    @{ key = "translit";    args = "--script translit";    suffix = "len128_normoff_scripttranslit" },
    @{ key = "roman_train"; args = "--script roman_train"; suffix = "len128_normoff_scriptroman_train" },
    @{ key = "len64";       args = "--max_len 64";         suffix = "len64_normoff_scriptasis" },
    @{ key = "len256";      args = "--max_len 256";        suffix = "len256_normoff_scriptasis" }
)
$allKeys = @($configs | ForEach-Object { $_.key })

$onlyText = "all"
if ($OnlyList.Count -gt 0) { $onlyText = $OnlyList -join ", " }
Write-Host "Seeds: $($SeedList -join ', ') | Configs: $onlyText"

if ($OnlyList.Count -gt 0) {
    $unknown = @($OnlyList | Where-Object { $allKeys -notcontains $_ })
    if ($unknown.Count -gt 0) {
        Write-Host "Unknown config name(s): $($unknown -join ', '). Valid names: $($allKeys -join ', ')"
        exit 1
    }
    $configs = @($configs | Where-Object { $OnlyList -contains $_.key })
}
if ($SeedList.Count -eq 0) { Write-Host "No seeds given."; exit 1 }

function Wait-GpuCool {
    Write-Host ""
    Write-Host "=== Cooling down: waiting for the GPU to reach $CoolToC C (max $MaxCoolMinutes min) ==="
    $t0 = Get-Date
    while ($true) {
        $temp = [int](& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
        $mins = ((Get-Date) - $t0).TotalMinutes
        Write-Host ("    GPU {0} C, waited {1:N1} min" -f $temp, $mins)
        if ($temp -le $CoolToC -or $mins -ge $MaxCoolMinutes) { break }
        Start-Sleep -Seconds 30
    }
    Write-Host "=== Resuming ==="
    Write-Host ""
}

$jobs = @()
foreach ($s in $SeedList) {
    foreach ($c in $configs) {
        $done = "models\hingbert_lr1e-05_seed${s}_$($c.suffix)\best\config.json"
        if (Test-Path $done) {
            Write-Host "skip (already finished): seed $s, $($c.key)"
        } else {
            $jobs += @{ seed = $s; cfg = $c }
        }
    }
}
Write-Host "$($jobs.Count) run(s) to do."
if ($jobs.Count -eq 0) {
    Write-Host "Nothing to do: every requested run has already finished."
    exit 0
}

$chunkStart = Get-Date
$completed = 0
foreach ($j in $jobs) {
    if (((Get-Date) - $chunkStart).TotalMinutes -ge $ChunkMinutes) {
        Wait-GpuCool
        $chunkStart = Get-Date
    }
    Write-Host ("[{0}] starting: seed {1}, {2}" -f (Get-Date -Format "HH:mm"), $j.seed, $j.cfg.key)
    $cmdArgs = @("src/train_transformer.py", "--model", "hingbert", "--seed", "$($j.seed)", "--lr", "1e-5") + ($j.cfg.args -split " ")
    & python @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: seed $($j.seed), $($j.cfg.key) (exit code $LASTEXITCODE). Stopping. Fix the error, then re-run this script."
        exit 1
    }
    $completed++
}

Write-Host ""
Write-Host "Done: $completed run(s) finished. Now run:  python src/summarize.py"
