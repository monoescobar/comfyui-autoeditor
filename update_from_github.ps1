$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($Args -join ' ')"
    }
}

Write-Host "Getting latest AutoEditor from GitHub..."
Invoke-Git pull --ff-only origin master

Write-Host ""
Invoke-Git status --short --branch
Write-Host "Done."
