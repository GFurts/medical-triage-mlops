# Roda o mesmo benchmark contra cada backend, um container por vez.
# Antes: notebook na tomada, em modo desempenho, e `docker compose down`.

$experiments = @(
    @{ Label = "sklearn_original"; Env = @("MODEL_BACKEND=sklearn") },
    @{ Label = "sklearn_1job";     Env = @("MODEL_BACKEND=sklearn", "SKLEARN_N_JOBS=1") },
    @{ Label = "onnx";             Env = @("MODEL_BACKEND=onnx") }
)

foreach ($exp in $experiments) {
    Write-Host "`n=== $($exp.Label) ==="
    $envArgs = $exp.Env | ForEach-Object { "-e"; $_ }
    docker run --rm -d --name triage-exp -p 8000:8000 @envArgs medical-triage-api | Out-Null

    $healthy = $false
    for ($i = 0; $i -lt 45 -and -not $healthy; $i++) {
        Start-Sleep -Seconds 2
        $healthy = (docker inspect --format '{{.State.Health.Status}}' triage-exp) -eq "healthy"
    }
    if (-not $healthy) {
        docker logs triage-exp
        docker stop triage-exp | Out-Null
        throw "A API não ficou saudável no experimento $($exp.Label)"
    }

    poetry run python scripts/benchmark_latency.py --label $exp.Label --requests 500
    docker stats --no-stream --format 'Memória do container: {{.MemUsage}}' triage-exp
    docker stop triage-exp | Out-Null
}