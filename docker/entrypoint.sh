#!/bin/sh
# Dual-mode entrypoint. With no arguments (the Dockerfile's own default CMD) or
# `serve-dashboard` explicitly, seeds a small demo run (if `runs/` is empty) and
# serves the Streamlit dashboard on $PORT -- this is what a PaaS host runs by
# default, with no reliance on it correctly overriding the Dockerfile's command.
# `docker run aerovalid <fsa-args>` (e.g. `generate --config ...`) still runs the
# CLI instead.
set -e

serve_dashboard() {
    if [ -z "$(ls -A runs 2>/dev/null)" ]; then
        echo "entrypoint: runs/ is empty, seeding a small demo run..."
        fsa run-all --config configs/default.yaml --run-id demo --profile demo \
            --set generate.n_flights=2 \
            --set "generate.scenarios=[doublet_series]" \
            --set "variants=[V0, V1, V2]"
    fi
    exec streamlit run src/aerovalid/app/main.py \
        --server.port "${PORT:-8501}" \
        --server.address 0.0.0.0 \
        --server.headless true
}

if [ "$#" -eq 0 ] || [ "$1" = "serve-dashboard" ]; then
    serve_dashboard
else
    exec fsa "$@"
fi
