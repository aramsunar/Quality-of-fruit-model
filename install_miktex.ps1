# MiKTeX Installation Script
# Run this script as Administrator

Write-Host "Installing MiKTeX via Chocolatey..." -ForegroundColor Green

# Check if Chocolatey is installed
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey not found. Installing Chocolatey first..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

    Write-Host "Chocolatey installed. Please close this window and run the script again as Administrator." -ForegroundColor Yellow
    pause
    exit
}

Write-Host "Chocolatey found. Installing MiKTeX..." -ForegroundColor Green
choco install miktex -y

Write-Host "MiKTeX installation completed!" -ForegroundColor Green
Write-Host "Verifying installation..." -ForegroundColor Green

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Verify installation
if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    Write-Host "SUCCESS: pdflatex is available!" -ForegroundColor Green
    pdflatex --version
} else {
    Write-Host "Note: You may need to restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
}

pause
