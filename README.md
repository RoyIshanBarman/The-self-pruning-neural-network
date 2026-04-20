# Self-Pruning Neural Network

This repository contains the implementation of a "Self-Pruning" Neural Network designed for image classification on the CIFAR-10 dataset, built as part of the AI Engineer Case Study for Tredence Analytics.

## Overview
Unlike standard post-training pruning techniques, this network learns to prune its own weights *dynamically during the training process*. It achieves this by:
1. Replacing standard linear layers with a custom `PrunableLinear` layer.
2. Associating every weight with a learnable gate parameter (passed through a Sigmoid activation).
3. Applying an **L1 Regularization Penalty** to the gates, pushing unnecessary connections to exactly zero.

## Repository Structure
- `main.py`: The complete source code, including the custom `PrunableLinear` layer, network architecture, and training/evaluation loop.
- `self_pruning_report.md`: The case study report detailing the mathematical intuition behind the L1 penalty, experiment results, and visualization of the gate distribution.
- `requirements.txt`: Project dependencies.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd <repo-directory>
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv myenv
   # On Windows:
   .\myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the experiment:**
   ```bash
   python main.py
   ```
   *Note: The script automatically downloads the CIFAR-10 dataset into a local `./data` folder and runs the training loop. It will output accuracy and sparsity metrics to the console and generate distribution plots.*

## Results
Please refer to the `self_pruning_report.md` for a detailed breakdown of the model's sparsity-vs-accuracy trade-off across different penalty hyperparameters ($\lambda$).
