#!/bin/bash

Write-Host "=============================="
Write-Host " Launching MLflow Server"
Write-Host " URL: http://127.0.0.1:5000"
Write-Host "=============================="

mlflow server --host 127.0.0.1 --port 5000