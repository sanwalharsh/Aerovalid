# AeroValidate: flight-simulation validation and predictive analysis CLI.
# Builds a lean image with the core pipeline (sim/telemetry/prep/validate/predict/
# detect/identify/report); optional extras (torch/lightgbm/shap sequence models and
# scoring, pymavlink/pyulog real-log adapters) are not installed here -- every module
# that needs them guards the import and degrades gracefully without them (CLAUDE.md
# §2, DECISIONS.md R6). Install with `--build-arg EXTRAS=.[all]` to include them.
FROM python:3.11-slim AS base

ARG EXTRAS=.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir "${EXTRAS}"

COPY configs/ configs/

# `runs/`/`data/`/`reports/` are git-ignored and expected to be bind-mounted in:
#   docker run -v "$PWD/runs:/app/runs" aerovalid run-all --config configs/default.yaml --run-id demo
RUN mkdir -p runs data reports \
    && useradd --create-home --shell /bin/bash aerovalid \
    && chown -R aerovalid:aerovalid /app
USER aerovalid

# `fsa dashboard` serves Streamlit on its default port.
EXPOSE 8501

ENTRYPOINT ["fsa"]
CMD ["--help"]
