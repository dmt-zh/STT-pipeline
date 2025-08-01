#!/bin/bash

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}")"
COMPOSE_YML="${BASE_DIR}/docker-compose.yml"
ENV_FILE="${BASE_DIR}/.env"

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_YML}" up -d
