#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "Usage: $0 <seed>"
    exit 1
fi

# Replace the following with a call to your own program.

# Run the language model
python3 run_ngram_language_model.py $1
