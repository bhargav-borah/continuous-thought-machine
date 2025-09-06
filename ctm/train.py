import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .utils import get_loss, calculate_accuracy
from IPython.display import display, clear_output


def update_training_curve_plot(fig, 
                               ax1, 
                               ax2, 
                               train_losses, 
                               test_losses, 
                               train_accuracies, 
                               test_accuracies, 
                               steps
                              ):
    """
    Plots training and testing curves (in terms of losses and accuracies), which are updated after every iteration

    Arguments:
        fig: The figure object containing the subplots
        ax1: The axis object used to plot losses
        ax2: The axis object used to plot accuracies
        train_losses: List of training losses recorded over iterations
        test_losses: List of test losses recorded at specific iterations
        train_accuracies: List of train accuracies recorded over iterations
        test_accuracies: List of test accuracies recorded at specific iterations
        steps: List of iterations at which test metrics were recorded
        
    """
    clear_output(wait=True)

    # Plot loss
    ax1.clear()
    ax1.plot(range(len(train_losses)), train_losses, 'b-', alpha=0.7, label=f'Train Loss: {train_losses[-1]:.3f}')
    ax1.plot(steps, test_losses, 'r-', marker='o', label=f'Test Loss: {test_losses[-1]:.3f}')
    ax1.set_title('Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.clear()
    ax2.plot(range(len(train_accuracies)), train_accuracies, 'b-', alpha=0.7, label=f'Train Accuracy: {train_accuracies[-1]:.3f}')
    ax2.plot(steps, test_accuracies, 'r-', marker='o', label=f'Test Accuracy: {test_accuracies[-1]:.3f}')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    display(fig)


def train(model,
          trainloader,
          testloader,
          iterations,
          test_every,
          device
          ):
  """
  Trains the model on batches generated from the trainloader and evaluates on the testloader every test_every iterations.

  Arguments:
      model: Instance of the ContinuousThoughtMachine model initialized with the desired arguments
      trainloader: torch.utils.data.DataLoader object which returns batches of images and corresponding labels from the training data
      testloader: torch.utils.data.DataLoader object which returns batches of images and corresponding labels from the test data
      iterations: Number of iterations the training loop is to run for
      test_every: Intervals of iterations starting from the first iteration, when the test set is to be evaluated
      device: Device to be used for training and inference
  Returns:
      model: The trained model
  """
  optimizer = torch.optim.AdamW(
      params=list(model.parameters()),
      lr=1e-4,
      eps=1e-8
  )
  iterator = iter(trainloader)
  model.train()

  train_losses = []
  test_losses = []
  train_accuracies = []
  test_accuracies = []
  steps = []

  plt.ion()
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

  with tqdm(total=iterations, initial=0, dynamic_ncols=True) as pbar:
    test_loss = None
    test_accuracy = None
    for stepi in range(iterations):

      try:
        inputs, targets = next(iterator)
      except StopIteration:
        iterator = iter(trainloader)
        inputs, targets = next(iterator)

      inputs, targets = inputs.to(device), targets.to(device)
      predictions, certainties, _ = model(inputs, track=False)
      train_loss, where_most_certain = get_loss(
          predictions,
          certainties,
          targets
      )
      train_accuracy = calculate_accuracy(
          predictions,
          targets,
          where_most_certain
      )

      train_losses.append(train_loss.item())
      train_accuracies.append(train_accuracy)

      train_loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if stepi % test_every == 0 or stepi == iterations - 1:
        model.eval()
        with torch.inference_mode():
          all_test_predictions = []
          all_test_targets = []
          all_test_where_most_certain = []
          all_test_losses = []

          for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, certainties, _ = model(inputs, track=False)
            test_loss, where_most_certain = get_loss(
                predictions,
                certainties,
                targets
            )

            all_test_losses.append(test_loss.item())
            all_test_predictions.append(predictions)
            all_test_targets.append(targets)
            all_test_where_most_certain.append(where_most_certain)

          all_test_predictions = torch.cat(all_test_predictions, dim=0)
          all_test_targets = torch.cat(all_test_targets, dim=0)
          all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)
          test_accuracy = calculate_accuracy(
              all_test_predictions,
              all_test_targets,
              all_test_where_most_certain
          )
          test_loss = sum(all_test_losses) / len(all_test_losses)

          test_losses.append(test_loss)
          test_accuracies.append(test_accuracy)
          steps.append(stepi)
      model.train()

      update_training_curve_plot(fig, ax1, ax2, train_losses, test_losses, train_accuracies, test_accuracies, steps)

    pbar.set_description(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}')
    pbar.update(1)

  plt.ioff()
  plt.close(fig)
  return model