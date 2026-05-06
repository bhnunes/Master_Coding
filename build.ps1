$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir
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
}
