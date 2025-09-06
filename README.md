# Continuous Thought Machine

A machine learning project for experimenting with continuous learning on the MNIST dataset.

## Installation

```sh
pip install -r requirements.txt
```

## Project Structure

```
ctm/                # Core package: data loading, models, training, utils, visualization
data/               # Dataset storage (MNIST)
experiments/        # Experiment scripts (e.g., mnist_train.py)
mnist_logs/         # Output logs and visualizations
notebooks/          # Jupyter notebooks for exploration
```

## How to Run the MNIST Experiment

```sh
python experiments/mnist_train.py
```

## Files and Directories

- [`ctm/data.py`](ctm/data.py): Data loading and preprocessing utilities.
- [`ctm/models.py`](ctm/models.py): Model definitions.
- [`ctm/train.py`](ctm/train.py): Training routines.
- [`ctm/utils.py`](ctm/utils.py): Helper functions.
- [`ctm/viz.py`](ctm/viz.py): Visualization utilities.
- [`experiments/mnist_train.py`](experiments/mnist_train.py): Main experiment script for MNIST.
- [`notebooks/ctm_mnist_final_implementation.ipynb`](notebooks/ctm_mnist_final_implementation.ipynb): Example notebook.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.