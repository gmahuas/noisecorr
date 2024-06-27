#!/bin/bash

# Run this bash to generate the training and testing sets.
# The two codes only differ by their random seed and the folder
# they store the files in.

# Simulate training set
echo "0" > input.txt
for ((i=1; i<=11; i++))
do
    # Launch the Python script and wait for it to finish
    python simulate_train.py
done
rm input.txt

# Simulate testing set
echo "0" > input.txt
for ((i=1; i<=11; i++))
do
    # Launch the Python script and wait for it to finish
    python simulate_test.py
done
rm input.txt
