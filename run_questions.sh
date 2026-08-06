#!/bin/bash

while IFS= read -r question
do
    echo "=========================================="
    echo "Question: $question"
    echo "=========================================="

    python index_utils.py "$question"

    echo
done < $1
