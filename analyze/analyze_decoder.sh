#!/bin/bash

# Run this bash to compute the informations from the decoders.

echo "0" > input.txt
for ((i=1; i<=11; i++))
do
    # Launch the Python script and wait for it to finish
    python analyze_decoder.py
done
rm input.txt
