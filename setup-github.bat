@echo off
REM Script to create GitHub repository and push code
REM Usage: setup-github.bat YOUR_GITHUB_TOKEN

if "%1"=="" (
    echo Usage: setup-github.bat YOUR_GITHUB_TOKEN
    echo Get token from: https://github.com/settings/tokens
    pause
    exit /b 1
)

setlocal enabledelayedexpansion

set TOKEN=%1
set REPO_NAME=GeoSpatial-Site-Readiness-Analysis
set USERNAME=dhyeyvala-10

echo Creating GitHub repository...
curl -X POST https://api.github.com/user/repos ^
  -H "Authorization: token %TOKEN%" ^
  -H "Accept: application/vnd.github+json" ^
  -d "{\"name\":\"%REPO_NAME%\",\"description\":\"AI-Powered Location Intelligence for EV Charging Stations and Warehouses\",\"private\":true}"

echo.
echo Waiting 3 seconds for repository to be created...
timeout /t 3 /nobreak

echo.
echo Pushing code to GitHub...
git push -u origin main

if errorlevel 0 (
    echo.
    echo SUCCESS! Repository created and code pushed to:
    echo https://github.com/%USERNAME%/%REPO_NAME%
) else (
    echo.
    echo Error pushing to repository
)

pause
