# ============================================================
#  compile_shaders.ps1
#  MX2LM D3D12 Fused Runtime
#  Compiles all HLSL shaders to .cso bytecode using dxc or fxc
# ============================================================

param(
    [string]$ShaderDir  = "$PSScriptRoot\..\shaders",
    [string]$OutputDir  = "$PSScriptRoot\..\build\shaders",
    [string]$ShaderModel = "cs_6_0",
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

# ── Find compiler ────────────────────────────────────────────
function Find-Compiler {
    # Prefer dxc (DXC / Shader Model 6+)
    $dxc = Get-Command dxc -ErrorAction SilentlyContinue
    if ($dxc) { return @{ exe = "dxc"; isDxc = $true } }

    # Common Windows SDK paths
    $sdkPaths = @(
        "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe",
        "C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe",
        "C:\Program Files (x86)\Windows Kits\10\bin\x64\dxc.exe"
    )
    foreach ($p in $sdkPaths) {
        if (Test-Path $p) { return @{ exe = $p; isDxc = $true } }
    }

    # Fallback: fxc
    $fxc = Get-Command fxc -ErrorAction SilentlyContinue
    if ($fxc) { return @{ exe = "fxc"; isDxc = $false } }

    throw "Neither dxc nor fxc found. Install Windows SDK."
}

# ── Shader list: (hlsl, entry, output_cso) ───────────────────
$shaders = @(
    @{ src = "fused_attention_force_moe.hlsl"; entry = "CSMain";    out = "fused_attention_force_moe.cso" },
    @{ src = "grid_build.hlsl";                entry = "CSCount";   out = "grid_build_count.cso"          },
    @{ src = "grid_build.hlsl";                entry = "CSScan";    out = "grid_build_scan.cso"           },
    @{ src = "grid_build.hlsl";                entry = "CSScatter"; out = "grid_build_scatter.cso"        },
    @{ src = "prefix_sum.hlsl";                entry = "CSScan";    out = "prefix_sum.cso"                },
    @{ src = "prefix_sum.hlsl";                entry = "CSAddBlockSums"; out = "prefix_sum_add.cso"       },
    @{ src = "debug_visualize.hlsl";           entry = "CSMain";    out = "debug_visualize.cso"           }
)

# ── Setup ─────────────────────────────────────────────────────
$compiler = Find-Compiler
Write-Host "  [compiler] $($compiler.exe)" -ForegroundColor Cyan

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$ok      = 0
$failed  = 0
$total   = $shaders.Count

foreach ($s in $shaders) {
    $srcPath = Join-Path $ShaderDir $s.src
    $outPath = Join-Path $OutputDir $s.out

    if (-not (Test-Path $srcPath)) {
        Write-Host "  [SKIP]  $($s.src) not found" -ForegroundColor Yellow
        $failed++
        continue
    }

    Write-Host "  [....] $($s.src)::$($s.entry) → $($s.out)" -ForegroundColor DarkCyan

    if ($compiler.isDxc) {
        $args = @(
            "-T", $ShaderModel,
            "-E", $s.entry,
            "-Fo", $outPath,
            "-I", $ShaderDir
        )
        if ($Debug) { $args += @("-Od", "-Zi", "-Qembed_debug") }
        else        { $args += @("-O3") }
        $args += $srcPath
    } else {
        # fxc fallback (SM 5.1 max)
        $sm51 = $ShaderModel -replace "6_0","5_1"
        $args = @(
            "/T", $sm51,
            "/E", $s.entry,
            "/Fo", $outPath,
            "/I", $ShaderDir
        )
        if ($Debug) { $args += "/Od" } else { $args += "/O3" }
        $args += $srcPath
    }

    $result = & $compiler.exe @args 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAIL]  $($s.src)::$($s.entry)" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        $failed++
    } else {
        Write-Host "  [OK]   $($s.out)" -ForegroundColor Green
        $ok++
    }
}

Write-Host ""
Write-Host "  Compiled: $ok / $total  |  Failed: $failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
if ($failed -gt 0) { exit 1 }
