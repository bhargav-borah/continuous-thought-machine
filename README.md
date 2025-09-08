# Continuous Thought Machine

An implementation of the paper [Continuous Thought Machines (Darlow et al., 2025)](https://arxiv.org/abs/2505.05522) on MNIST dataset. I also wrote an [accompanying blog](https://open.substack.com/pub/bhargavborah/p/continuous-thought-machines-the-beginning?r=6ah8jl&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) sharing my thoughts and implementation details. The references I used besides the original paper are the [original blog](https://sakana.ai/ctm/) on the paper released by Sakana.ai, the [official GitHub repo](https://github.com/SakanaAI/continuous-thought-machines?tab=readme-ov-file), and the [interactive demonstration](https://pub.sakana.ai/ctm/).

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
