FROM apache/airflow:3.1.5-python3.12

# Ambiente isolado de treino, instalado a partir do MESMO poetry.lock da API
COPY --chown=airflow:root pyproject.toml poetry.lock /tmp/build/

RUN python -m venv /home/airflow/train-venv \
    && export PIP_USER=false VIRTUAL_ENV=/home/airflow/train-venv \
    && export PATH=/home/airflow/train-venv/bin:$PATH \
    && pip install --no-cache-dir poetry==2.4.1 \
    && cd /tmp/build \
    && poetry install --only main --no-root