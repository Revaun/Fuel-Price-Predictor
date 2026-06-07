# check-endpoint-status.ps1
$endpointName = "petrol-price-endpoint-v10"

while ($true) {
    $status = aws sagemaker describe-endpoint --endpoint-name $endpointName | ConvertFrom-Json
    $currentStatus = $status.EndpointStatus
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Endpoint status: $currentStatus"

    if ($currentStatus -eq "InService") {
        Write-Host "✅ Endpoint is now InService and ready for predictions."
        break
    }

    Start-Sleep -Seconds 10
}
