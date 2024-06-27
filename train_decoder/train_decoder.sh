#!/bin/bash

# Run this bash to train decoders.

echo "0" > input.txt
for ((i=1; i<=11; i++))
do
    # Launch the Python script and wait for it to finish
    python train_decoder.py
done
rm input.txt
