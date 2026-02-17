#!/usr/bin/env bash
set -e

eval "$(mise activate bash)"

pip install -e ".[serve,dev]"

