#!/usr/bin/env bash

runner_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_python="${PYTHON_BIN:-}"
if [[ -z "${config_python}" ]]; then
  for candidate in python3 python python.exe; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c "import json" >/dev/null 2>&1; then
      config_python="${candidate}"
      break
    fi
  done
fi
if [[ -z "${config_python}" ]]; then
  echo "Python is required to read experiments.json; set PYTHON_BIN to a Python interpreter." >&2
  exit 1
fi
config_values="$("${config_python}" "${runner_directory}/runner_config.py" "${CONFIG_FILE:-${runner_directory}/experiments.json}" "$1")" || exit 1
eval "${config_values}"
