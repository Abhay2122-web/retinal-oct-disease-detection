# Run from the root folder of the existing retinal_oct_complete_project.
$ErrorActionPreference = "Stop"

Write-Host "1/3 Checking project files and dependencies..." -ForegroundColor Cyan
python verify_project.py

Write-Host "2/3 Demonstrating NumPy neural-network fundamentals..." -ForegroundColor Cyan
python numpy_fundamentals.py

Write-Host "3/3 Starting the existing Streamlit application..." -ForegroundColor Cyan
if (Test-Path "app.py") {
    streamlit run app.py
} elseif (Test-Path "app(3).py") {
    Write-Host "Found app(3).py. Copying it to app.py while preserving the original..." -ForegroundColor Yellow
    Copy-Item "app(3).py" "app.py" -ErrorAction Stop
    streamlit run app.py
} else {
    throw "Could not find app.py or app(3).py in the current folder."
}
