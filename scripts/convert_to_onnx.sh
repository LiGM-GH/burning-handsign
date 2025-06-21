#!/usr/bin/env bash

cd "$(dirname $0)"
uv run main.py $1 $2
