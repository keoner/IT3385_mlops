.\.venv\Scripts\Activate.ps1

Get-Content .env | Where-Object { $_ -and $_ -notlike '#*' } | ForEach-Object { $k, $v = $_.Split('=', 2); [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim().Trim("`"'"), "Process") }
dvc config --local remote.gcloud.gdrive_client_id "$env:GDRIVE_CLIENT_ID"
dvc config --local remote.gcloud.gdrive_client_secret "$env:GDRIVE_CLIENT_SECRET"