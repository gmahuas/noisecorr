# Description

This repository accompanies the preprint "Mahuas, G., Buffet, T., Marre, O., Ferrari, U., & Mora, T. (2024). Strong, but not weak, noise correlations are beneficial for population coding." [Read the preprint on arXiv](https://arxiv.org/abs/2406.18439v1).

It contains the code necessary to reproduce the results shown in Fig. 4, illustrating the effect of noise correlations on stimulus encoding at different spatial scales.

# Manual

To reproduce panels 4B and 4C from precomputed data, run the notebook located at `./analyze/plot_panels.ipynb`.

To redo the analysis from scratch, follow these steps in order:
1. Run `simulate/simulate.sh` to generate the synthetic datasets.
2. Run `train_decoder/train_decoder.sh` to train the decoders.
3. Run `analyze/analyze_decoder.sh` to compute the mutual informations and the noise synergy from the decoders.
4. Plot the results using `analyze/plot_panels.ipynb`.

The Bash scripts need to be made executable by running: `chmod +x <script_name>.sh`.
