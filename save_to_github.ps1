param(
    [string]$Message = "auto backup",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($args -join ' ')"
    }
}

Write-Host "Checking AutoEditor Git status..."
Invoke-Git status --short | Out-Null
$status = git status --short

if ($status) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $commitMessage = "$Message $timestamp"

    Write-Host "Saving local changes..."
    Invoke-Git add -A
    Invoke-Git commit -m $commitMessage
} else {
    Write-Host "No local changes to commit."
}

if (-not $NoPush) {
    Write-Host "Checking GitHub for updates..."
    Invoke-Git pull --rebase origin master

    Write-Host "Uploading to GitHub..."
    Invoke-Git push origin master
} else {
    Write-Host "NoPush selected, skipping GitHub pull/upload."
}

Write-Host ""
Invoke-Git status --short --branch
Write-Host "Done."
