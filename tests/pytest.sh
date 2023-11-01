#!/bin/bash

echo "Running Unit tests"

pytest --random-order --cov=uptrading --cov-config=.coveragerc tests/
