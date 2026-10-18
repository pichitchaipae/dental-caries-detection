# Start the staging inference service (no changes to training artifacts).
#
# Example:
#   powershell -ExecutionPolicy Bypass -File deploy\run_staging.ps1 -Port 8011 -Folds "0,1,2,3"
#
# The service runs on CPU by default so it never competes with the training
# queue for the GPU. Set -Device cuda only when training is idle.

param(
    [int]$Port = 8011,
    [string]$Folds = "0,1,2,3,4",
    [string]$Device = "cpu",
    [string]$Calibration = ""
)

$py = "D:\Users\jaopi\anaconda3\envs\sp_project\python.exe"
$root = "D:\Mahidol University\SP\caries-detection-model"

$cps = @()
foreach ($f in ($Folds -split ",")) {
    $p = Join-Path $root ("outputs\best_pca_s42_f{0}.pt" -f $f.Trim())
    if (Test-Path $p) { $cps += $p }
}
if ($cps.Count -eq 0) { throw "no checkpoints found for folds: $Folds" }

$env:EFFNET_CHECKPOINTS = ($cps -join ";")
$env:EFFNET_DEVICE = $Device
$env:EFFNET_CALIBRATION = $Calibration
$env:EFFNET_TORCH_THREADS = "4"

New-Item -ItemType Directory -Force -Path (Join-Path $root "outputs\deploy") | Out-Null
Set-Location $root

Write-Host ("staging folds: " + (($cps | ForEach-Object { Split-Path $_ -Leaf }) -join ", "))
Write-Host ("device=$Device port=$Port calibration=$(if ($Calibration) { $Calibration } else { '<none>' })")
& $py -m uvicorn deploy_effnet_service:app --host 127.0.0.1 --port $Port --log-level info
