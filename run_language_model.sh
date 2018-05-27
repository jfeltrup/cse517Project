#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "Usage: $0 <seed>"
    exit 1
fi

# Replace the following with a call to your own program.

# Activate the pytorch environment
source pytorch_env/bin/activate
# Run the language model
python3 run_language_model.py $1
# Deactivate the environment
deactivate
