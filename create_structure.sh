#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/projects/rag-notebook"

mkdir -pv "$PROJECT/notebooks"
mkdir -pv "$PROJECT/data/docs"

touch "$PROJECT/Dockerfile"
touch "$PROJECT/docker-compose.yml"
touch "$PROJECT/requirements.txt"
touch "$PROJECT/entrypoint.sh"
touch "$PROJECT/.env.example"
touch "$PROJECT/.gitignore"

echo "Структура создана:"
find "$PROJECT" | sort
