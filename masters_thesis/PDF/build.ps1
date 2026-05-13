$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceDir = Resolve-Path (Join-Path $ScriptDir "..\New_Thesis")

$ArtifactPatterns = @(
    "*.aux",
    "*.bbl",
    "*.bcf",
    "*.blg",
    "*.fls",
    "*.fdb_latexmk",
    "*.ilg",
    "*.lof",
    "*.log",
    "*.lot",
    "*.nlo",
    "*.nls",
    "*.out",
    "*.pdf",
    "*.run.xml",
    "*.synctex.gz",
    "*.toc"
)

function Move-BuildArtifacts {
    foreach ($pattern in $ArtifactPatterns) {
        Get-ChildItem -LiteralPath $SourceDir -File -Filter $pattern -Force -ErrorAction SilentlyContinue |
            ForEach-Object {
                $Destination = Join-Path $ScriptDir $_.Name
                if (Test-Path -LiteralPath $Destination) {
                    Remove-Item -LiteralPath $Destination -Force
                }
                Move-Item -LiteralPath $_.FullName -Destination $Destination -Force
            }
    }
}

Push-Location $SourceDir
try {
    xelatex -interaction=nonstopmode -halt-on-error tese.tex
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    biber tese
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (Test-Path "tese.nlo") {
        makeindex tese.nlo -s nomencl.ist -o tese.nls
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    xelatex -interaction=nonstopmode -halt-on-error tese.tex
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    xelatex -interaction=nonstopmode -halt-on-error tese.tex
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
    Move-BuildArtifacts
}
