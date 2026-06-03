<#
.SYNOPSIS
Synchronizes masters_thesis/New_Thesis with a small Git remote for Overleaf.

.DESCRIPTION
This script wraps git subtree commands so the thesis source folder can stay tracked in this
repository while also being pushed to, and pulled from, a smaller GitHub repository that
Overleaf can sync.

The script must be run from inside the main repository. It resolves the repository root
automatically, so it can be called from any subfolder.

.PARAMETER Action
The sync action to run. Use Push to send committed New_Thesis changes to the thesis remote,
Pull to bring Overleaf/thesis-repo changes back into this repository, or Status to inspect
the configured path and remote.

.PARAMETER Remote
The Git remote name for the small thesis repository. Defaults to thesis.

.PARAMETER Branch
The branch in the small thesis repository. Defaults to main.

.PARAMETER Prefix
The folder in this repository that is mirrored to the small thesis repository. Defaults to
masters_thesis/New_Thesis.

.PARAMETER RemoteUrl
Optional GitHub URL for the small thesis repository. If the remote does not exist, the script
adds it before Push or Pull.

.PARAMETER NoSquash
Only applies to Pull. By default, subtree pulls use --squash to keep the large repository
history compact.

.PARAMETER NoRejoin
Only applies to Push. By default, subtree pushes use --rejoin so later squash pulls can
recognize the subtree relationship.

.PARAMETER AllowDirty
Allows Push or Pull to continue with uncommitted local changes. Use this carefully: subtree
pushes only committed content, and subtree pulls can conflict with local edits.

.EXAMPLE
.\scripts\sync_new_thesis_repo.ps1 Status

.EXAMPLE
.\scripts\sync_new_thesis_repo.ps1 Push -RemoteUrl https://github.com/YOUR_USER/YOUR_THESIS_REPO.git

.EXAMPLE
.\scripts\sync_new_thesis_repo.ps1 Pull
#>

[CmdletBinding()]
param(
    [ValidateSet("Push", "Pull", "Status")]
    [string]$Action = "Status",

    [string]$Remote = "thesis",

    [string]$Branch = "main",

    [string]$Prefix = "masters_thesis/New_Thesis",

    [string]$RemoteUrl = "https://github.com/bhnunes/thesis_overleaf_v2.git",

    [switch]$NoSquash,

    [switch]$NoRejoin,

    [switch]$AllowDirty,

    [switch]$Help
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($Help) {
    Write-Host @"
Usage:
  .\scripts\sync_new_thesis_repo.ps1 Status
  .\scripts\sync_new_thesis_repo.ps1 Push -RemoteUrl https://github.com/YOUR_USER/YOUR_THESIS_REPO.git
  .\scripts\sync_new_thesis_repo.ps1 Pull

Defaults:
  Remote: thesis
  Branch: main
  Prefix: masters_thesis/New_Thesis

Notes:
  Push sends committed thesis-folder changes to the small Overleaf/GitHub repo.
  Push uses --rejoin by default so future squash pulls can recognize this subtree.
  Pull brings small-repo changes back into masters_thesis/New_Thesis.
  Pull uses --squash by default to keep this large repo history compact.
  Use -AllowDirty only if you understand uncommitted files are not part of subtree push.
"@
    return
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Format-GitCommand {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    return "git $($Arguments -join ' ')"
}

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & git @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$(Format-GitCommand $Arguments) failed with exit code $exitCode."
    }
}

function Get-GitOutput {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $output = @(& git @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $message = ($output | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($message)) {
            $message = "No output from git."
        }
        throw "$(Format-GitCommand $Arguments) failed with exit code $exitCode. $message"
    }

    return $output
}

function Resolve-GitRoot {
    $rootOutput = @(Get-GitOutput @("rev-parse", "--show-toplevel"))
    if ($rootOutput.Count -eq 0 -or [string]::IsNullOrWhiteSpace($rootOutput[0])) {
        throw "This script must be run from inside a Git repository."
    }

    return (Resolve-Path -LiteralPath $rootOutput[0].Trim()).Path
}

function Normalize-SubtreePrefix {
    param([Parameter(Mandatory = $true)][string]$Value)

    $normalized = ($Value -replace "\\", "/").Trim("/")
    if ([string]::IsNullOrWhiteSpace($normalized)) {
        throw "Prefix cannot be empty."
    }

    return $normalized
}

function Get-PrefixPath {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$NormalizedPrefix
    )

    $path = $RepoRoot
    foreach ($part in $NormalizedPrefix -split "/") {
        $path = Join-Path $path $part
    }

    return $path
}

function Assert-PrefixExists {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$NormalizedPrefix
    )

    $prefixPath = Get-PrefixPath -RepoRoot $RepoRoot -NormalizedPrefix $NormalizedPrefix
    if (-not (Test-Path -LiteralPath $prefixPath -PathType Container)) {
        throw "Subtree prefix does not exist: $prefixPath"
    }
}

function Get-GitRemoteNames {
    return @(Get-GitOutput @("remote"))
}

function Ensure-GitRemote {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteName,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Url
    )

    $remotes = Get-GitRemoteNames
    $remoteExists = $remotes -contains $RemoteName

    if (-not $remoteExists) {
        if ([string]::IsNullOrWhiteSpace($Url)) {
            throw "Remote '$RemoteName' does not exist. Pass -RemoteUrl on the first run."
        }

        Write-Step "Adding Git remote '$RemoteName'"
        Invoke-Git @("remote", "add", $RemoteName, $Url)
        return
    }

    if (-not [string]::IsNullOrWhiteSpace($Url)) {
        $currentUrl = (@(Get-GitOutput @("remote", "get-url", $RemoteName))[0]).Trim()
        if ($currentUrl -ne $Url.Trim()) {
            throw @"
Remote '$RemoteName' already exists with a different URL.
Current: $currentUrl
Passed:  $Url
Update it manually with: git remote set-url $RemoteName $Url
"@
        }
    }
}

function Assert-CleanWorktree {
    param(
        [Parameter(Mandatory = $true)][string]$CurrentAction,
        [Parameter(Mandatory = $true)][bool]$DirtyAllowed
    )

    if ($DirtyAllowed) {
        Write-Host "Warning: continuing with uncommitted changes because -AllowDirty was set." `
            -ForegroundColor Yellow
        return
    }

    $statusLines = @(Get-GitOutput @("status", "--porcelain"))
    if ($statusLines.Count -eq 0) {
        return
    }

    Write-Host "Uncommitted changes detected:" -ForegroundColor Yellow
    $statusLines | Select-Object -First 20 | ForEach-Object {
        Write-Host "  $_"
    }
    if ($statusLines.Count -gt 20) {
        Write-Host "  ... and $($statusLines.Count - 20) more"
    }

    $lowerAction = $CurrentAction.ToLowerInvariant()
    throw "Commit or stash changes before subtree $lowerAction, or rerun with -AllowDirty."
}

function Show-SyncStatus {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$NormalizedPrefix,
        [Parameter(Mandatory = $true)][string]$RemoteName,
        [Parameter(Mandatory = $true)][string]$BranchName
    )

    Write-Step "Thesis subtree sync status"
    Write-Host "Repository root: $RepoRoot"
    Write-Host "Subtree prefix: $NormalizedPrefix"
    Write-Host "Remote name:     $RemoteName"
    Write-Host "Remote branch:   $BranchName"

    $remotes = Get-GitRemoteNames
    if ($remotes -contains $RemoteName) {
        $remoteUrlValue = (@(Get-GitOutput @("remote", "get-url", $RemoteName))[0]).Trim()
        Write-Host "Remote URL:      $remoteUrlValue"
    }
    else {
        Write-Host "Remote URL:      not configured" -ForegroundColor Yellow
        Write-Host "First push:      .\scripts\sync_new_thesis_repo.ps1 Push -RemoteUrl <github-url>"
    }

    $prefixStatus = @(Get-GitOutput @("status", "--porcelain", "--", $NormalizedPrefix))
    if ($prefixStatus.Count -eq 0) {
        Write-Host "Prefix status:   clean"
    }
    else {
        Write-Host "Prefix status:   $($prefixStatus.Count) uncommitted change(s)" -ForegroundColor Yellow
    }
}

try {
    $normalizedPrefix = Normalize-SubtreePrefix $Prefix
    $repoRoot = Resolve-GitRoot

    Push-Location $repoRoot
    try {
        Assert-PrefixExists -RepoRoot $repoRoot -NormalizedPrefix $normalizedPrefix

        switch ($Action) {
            "Status" {
                Show-SyncStatus `
                    -RepoRoot $repoRoot `
                    -NormalizedPrefix $normalizedPrefix `
                    -RemoteName $Remote `
                    -BranchName $Branch
            }
            "Push" {
                Assert-CleanWorktree -CurrentAction $Action -DirtyAllowed $AllowDirty.IsPresent
                Ensure-GitRemote -RemoteName $Remote -Url $RemoteUrl

                Write-Step "Pushing '$normalizedPrefix' to '$Remote/$Branch'"
                $pushArgs = @("subtree", "push", "--prefix=$normalizedPrefix")
                if (-not $NoRejoin) {
                    $pushArgs += "--rejoin"
                }
                $pushArgs += @($Remote, $Branch)

                Invoke-Git $pushArgs
                Write-Host "Push complete." -ForegroundColor Green
                if (-not $NoRejoin) {
                    Write-Host "Note: --rejoin may create a local commit; push this main repo afterward."
                }
            }
            "Pull" {
                Assert-CleanWorktree -CurrentAction $Action -DirtyAllowed $AllowDirty.IsPresent
                Ensure-GitRemote -RemoteName $Remote -Url $RemoteUrl

                $pullArgs = @("subtree", "pull", "--prefix=$normalizedPrefix", $Remote, $Branch)
                if (-not $NoSquash) {
                    $pullArgs += "--squash"
                }

                Write-Step "Pulling '$Remote/$Branch' into '$normalizedPrefix'"
                Invoke-Git $pullArgs
                Write-Host "Pull complete." -ForegroundColor Green
            }
        }
    }
    finally {
        Pop-Location
    }
}
catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
