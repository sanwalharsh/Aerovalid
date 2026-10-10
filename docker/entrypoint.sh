#!/bin/sh
# Dual-mode entrypoint: `docker run aerovalid <fsa-args>` runs the CLI as before;
# `docker run aerovalid serve-dashboard` seeds a small demo run (if `runs/` is
# empty) and serves the Streamlit dashboard on $PORT -- the mode a PaaS host
# (Render/Railway/Fly) runs as its web service command.
set -e

if [ "$1" = "serve-dashboard" ]; then
    shift
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
        --server.headless true \
        "$@"
else
    exec fsa "$@"
fi
