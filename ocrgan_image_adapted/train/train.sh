#!/bin/bash

# This script will execute all training scripts in the current directory, in order.

set -e  # Exit immediately if any command exits with a non-zero status.

SCRIPTS=(
    "train_dagm.sh"
    "train_everything.sh"
    "train_kolektorsdd.sh"
    "train_mvtec_all.sh"
    "train_mvtec.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        echo "=== Executing $script ==="
        ./"$script"
    else
        echo "=== Executing $script (not executable, using bash) ==="
        bash "$script"
    fi
    echo "=== Finished $script ==="
    echo
done

echo "=== All training scripts finished. ==="