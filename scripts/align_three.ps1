# Three-audio CUDA parity helper.
# Usage (from repo root, release binary already built):
#   $env:MOSS_MODEL_DIR = "D:\MOSS-Transcribe-Diarize\pretrained\moss-transcribe-diarize"
#   .\scripts\align_three.ps1 -OutDir .\align_out
param(
    [string]$Model = $env:MOSS_MODEL_DIR,
    [string]$AudioDir = "D:\MOSS-Transcribe-Diarize",
    [string]$OutDir = ".\align_out",
    [string]$Backend = "cuda"
)

$ErrorActionPreference = "Stop"
if (-not $Model) { throw "Set -Model or MOSS_MODEL_DIR" }

$Prompt = "Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp."
$Bin = Join-Path (Split-Path $PSScriptRoot -Parent) "target\release\moss-transcribe.exe"
if (-not (Test-Path $Bin)) { throw "missing $Bin — run cargo build --release first" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$files = @(
    @{ n = "90s"; w = Join-Path $AudioDir "90s_16k.wav" },
    @{ n = "180s"; w = Join-Path $AudioDir "180s.wav" },
    @{ n = "ja"; w = Join-Path $AudioDir "ja.wav" }
)
foreach ($f in $files) {
    if (-not (Test-Path $f.w)) { throw "missing audio $($f.w)" }
    Write-Host "=== $($f.n) ==="
    $text = (& $Bin transcribe $f.w --model $Model --backend $Backend --prompt $Prompt 2> (Join-Path $OutDir "$($f.n)_stderr.log") | Out-String).Trim()
    [System.IO.File]::WriteAllText((Join-Path $OutDir "rust_$($f.n).txt"), $text)
    Write-Host "wrote rust_$($f.n).txt len=$($text.Length)"
}
Write-Host "done -> $OutDir"
