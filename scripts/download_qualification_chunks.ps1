[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$BearerToken = "",

    [int]$Parallelism = 8,

    [string]$AssembledPath = "",

    [switch]$InsecureTls
)

$ErrorActionPreference = "Stop"
if ($Parallelism -lt 1) {
    throw "Parallelism must be positive"
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$chunks = @($manifest.chunks)
if ($chunks.Count -eq 0) {
    throw "Chunk manifest is empty"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Test-Chunk {
    param($Chunk, [string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    if ((Get-Item -LiteralPath $Path).Length -ne [int64]$Chunk.bytes) {
        return $false
    }
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    return $hash.ToLowerInvariant() -eq $Chunk.sha256.ToLowerInvariant()
}

for ($offset = 0; $offset -lt $chunks.Count; $offset += $Parallelism) {
    $last = [Math]::Min($offset + $Parallelism - 1, $chunks.Count - 1)
    $processes = @()
    foreach ($chunk in $chunks[$offset..$last]) {
        $target = Join-Path $OutputDirectory $chunk.name
        if (Test-Chunk -Chunk $chunk -Path $target) {
            Write-Host "[chunks] verified existing $($chunk.name)"
            continue
        }

        $url = "$($BaseUrl.TrimEnd('/'))/$($chunk.name)"
        $arguments = @(
            "--fail",
            "--location",
            "--retry", "4",
            "--retry-all-errors",
            "--connect-timeout", "15",
            "--max-time", "1800",
            "--silent",
            "--show-error"
        )
        if ($InsecureTls) {
            $arguments += "--insecure"
        }
        if ($BearerToken) {
            $header = '"Authorization: Bearer {0}"' -f $BearerToken
            $arguments += @("--header", $header)
        }
        $quotedTarget = '"{0}"' -f $target
        $arguments += @("--output", $quotedTarget, $url)
        $process = Start-Process `
            -FilePath "curl.exe" `
            -ArgumentList $arguments `
            -WindowStyle Hidden `
            -PassThru
        $processes += [pscustomobject]@{
            Chunk = $chunk
            Path = $target
            Process = $process
        }
    }

    foreach ($entry in $processes) {
        $entry.Process.WaitForExit()
        if ($entry.Process.ExitCode -ne 0) {
            throw "curl failed for $($entry.Chunk.name): $($entry.Process.ExitCode)"
        }
        if (-not (Test-Chunk -Chunk $entry.Chunk -Path $entry.Path)) {
            throw "Checksum or length mismatch for $($entry.Chunk.name)"
        }
        Write-Host "[chunks] downloaded and verified $($entry.Chunk.name)"
    }
}

Write-Host "[chunks] all $($chunks.Count) chunks verified"

if ($AssembledPath) {
    $assembledParent = Split-Path -Parent $AssembledPath
    if ($assembledParent) {
        New-Item -ItemType Directory -Force -Path $assembledParent | Out-Null
    }
    $output = [System.IO.File]::Open(
        $AssembledPath,
        [System.IO.FileMode]::Create,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        foreach ($chunk in $chunks) {
            $path = Join-Path $OutputDirectory $chunk.name
            $input = [System.IO.File]::OpenRead($path)
            try {
                $input.CopyTo($output)
            }
            finally {
                $input.Dispose()
            }
        }
    }
    finally {
        $output.Dispose()
    }
    $assembled = Get-Item -LiteralPath $AssembledPath
    if ($assembled.Length -ne [int64]$manifest.archive_bytes) {
        throw "Assembled archive length mismatch: $($assembled.Length)"
    }
    $assembledHash = (
        Get-FileHash -LiteralPath $AssembledPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($assembledHash -ne $manifest.archive_sha256.ToLowerInvariant()) {
        throw "Assembled archive SHA-256 mismatch"
    }
    Write-Host "[chunks] assembled archive verified: $AssembledPath"
}
