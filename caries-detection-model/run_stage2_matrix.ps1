# Stage-2 training matrix (restart-resilient).
#
# Runs the locked reference configuration (pca) plus the off-control on the
# SAME folds/seed, then repeats both arms on seed 43 so the Nadeau & Bengio
# corrected t-test can include training variance.
#
# Every step is skipped when its out-of-fold report already exists, every fold
# is skipped when its checkpoint + metrics exist (--resume), a partially
# trained fold continues from {prefix}_last.pt, and each run waits until no
# other training process is running (one Graphical Processing Unit job at a
# time). A server restart therefore costs at most part of one fold; relaunching
# this file resumes where it stopped.
#
# WARNING: --resume trusts existing artifacts. Before rerunning after ANY
# label change, archive (rename away) outputs\best_<arm>_s<seed>_f*.pt,
# *_last.pt, *_metrics.json, *_history.json and oof_*.csv for that arm/seed,
# or the stale artifacts will be reused instead of retrained.
#
# Usage:  powershell -ExecutionPolicy Bypass -File run_stage2_matrix.ps1
# (for long detached runs use run_stage2_task.cmd, see its header)

$ErrorActionPreference = "Continue"

$py = "D:\Users\jaopi\anaconda3\envs\sp_project\python.exe"
$root = "D:\Mahidol University\SP\caries-detection-model"
$outs = Join-Path $root "outputs"
$peek = "@" + (Join-Path $outs "peeked_cases.json")
$log = Join-Path $outs "stage2_matrix.log"

function Log([string]$m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

# Single-driver guard: two drivers would race for the same Graphical
# Processing Unit and could start the same run twice.
$self = $PID
$peers = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.ProcessId -ne $self -and $_.CommandLine -match 'run_stage2_matrix' })
if ($peers.Count -gt 0) {
    Log ("another driver already running (pid {0}); exiting" -f (($peers | ForEach-Object { $_.ProcessId }) -join ','))
    exit 2
}

function Test-Oof([string]$arm, [int]$seed) {
    $new = Join-Path $outs ("oof_{0}_stratified-groupkfold_seed{1}_exc100.csv" -f $arm, $seed)
    $legacy = Join-Path $outs ("oof_{0}_stratified-groupkfold.csv" -f $arm)
    if ($seed -eq 42 -and (Test-Path $legacy)) { return $true }
    return (Test-Path $new)
}

function Wait-NoTraining {
    # Match on 'pipeline.py' only: the launched command line quotes the script
    # path, so a pattern like 'pipeline\.py\s+train' silently fails to match and
    # the driver would start a second training job on the same device.
    $deadline = (Get-Date).AddHours(6)
    while ((Get-Date) -lt $deadline) {
        $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match 'pipeline\.py' })
        if ($busy.Count -eq 0) { return $true }
        Log ("waiting for training process ids {0}" -f (($busy | ForEach-Object { $_.ProcessId }) -join ','))
        Start-Sleep -Seconds 60
    }
    return $false
}

$runs = @(
    [pscustomobject]@{ arm = "pca"; seed = 42 },
    [pscustomobject]@{ arm = "off"; seed = 42 },
    [pscustomobject]@{ arm = "pca"; seed = 43 },
    [pscustomobject]@{ arm = "off"; seed = 43 }
)

Log ("matrix start (runs: {0}, pid {1})" -f $runs.Count, $self)
foreach ($r in $runs) {
    $tag = "$($r.arm) seed$($r.seed)"
    if (Test-Oof $r.arm $r.seed) { Log "SKIP $tag (out-of-fold report exists)"; continue }
    Log "WAIT for the Graphical Processing Unit: $tag"
    if (-not (Wait-NoTraining)) { Log "TIMEOUT waiting for the unit; exiting"; exit 1 }
    if (Test-Oof $r.arm $r.seed) { Log "SKIP $tag (completed while waiting)"; continue }
    $runlog = Join-Path $outs ("stage2_{0}_seed{1}.log" -f $r.arm, $r.seed)
    Log "RUN $tag"
    & $py -u (Join-Path $root "pipeline.py") train `
        --ablation $r.arm --backbone efficientnet_b0 --loss focal `
        --img-size 384 --epochs 14 --cv stratified-groupkfold `
        --exclude $peek --save-oof --seed $r.seed --resume 2>&1 |
        Tee-Object -FilePath $runlog
    Log "DONE $tag exit=$LASTEXITCODE"
}
Log "matrix complete"
