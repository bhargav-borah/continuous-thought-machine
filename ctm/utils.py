import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_normalized_entropy(logits, reduction='mean'):
  """
  Computes normalized entropy, which is a measure of randomness, bounded in the range [0, 1].
  Greater normalized entropy implies greater randomness and lesser uncertainty.

  The normalized entropy is defined as:

        H_norm(p) = - (1 / log(k)) * Σ_i p_i * log(p_i)

    where:
        - p = (p_1, ..., p_k) is a probability distribution,
        - k is the number of classes,
        - H_norm(p) ∈ [0, 1].

    Interpretation:
        - H_norm(p) = 0 when the distribution is fully certain 
          (one class has probability 1, others 0).
        - H_norm(p) = 1 when the distribution is maximally uncertain 
          (all classes equally likely).

  Arguments:
      logits: Values representing preference for classes for each data point, can be converted into a probability (Boltzmann) distribution.

  Returns:
      normalized_entropy: Scalar representing normalized entropy.
  """
  preds = F.softmax(logits, dim=-1)
  log_preds = torch.log_softmax(logits, dim=-1)
  entropy = -torch.sum(preds * log_preds, dim=-1)
  num_classes = preds.shape[-1]
  max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
  normalized_entropy = entropy / max_entropy
  if len(logits.shape) > 2 and reduction == 'mean':
    normalized_entropy = normalized_entropy.flatten(1).mean(-1)
  return normalized_entropy


def get_loss(predictions, certainties, targets, use_most_certain=True):
  """
  Given predictions and uncertainties over internal ticks, 
  computes the mean of the losses at the tick indices with 
  lowest loss and highest certainty. 

  If use_most_certain set to False, then the loss at the last tick index instead of the loss at the most certain tick index.

  Arguments:
      predictions: Predictions made over all the internal ticks.
      certainties: Certainties of the predictions made over all the internal ticks.
      targets: Ground truth values.
      use_most_certain: Whether to use the most certain tick index or, the last tick index (default True).
  Returns:
      loss: Computed mean loss value
      loss_index_2: Index of the 'certainty' loss, an indicator of whether the use_most_certain was True or False
  """
  losses = nn.CrossEntropyLoss(reduction='none')(
      predictions,
      torch.repeat_interleave(
          targets.unsqueeze(-1),
          predictions.size(-1),
          -1
      )
  )

  loss_index_1 = losses.argmin(dim=1)
  loss_index_2 = certainties[:, 1].argmax(-1)
  if not use_most_certain:
    loss_index_2[:] = -1

  batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
  loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
  loss_selected = losses[batch_indexer, loss_index_2].mean()

  loss = (loss_minimum_ce + loss_selected) / 2
  return loss, loss_index_2

def calculate_accuracy(predictions, targets, where_most_certain):
  """
  Computes accuracy using predictions at the most certain internal tick, and the ground truth values.

  Arguments:
      predictions: History of predictions made by the CTM over internal ticks
      targets: Ground truth values
      where_most_certain: Tick indices corresponding to losses with most certain predictions
  Returns:
      accuracy: Scalar value representing computed accuracy
      
  """
  B = predictions.size(0)
  device = predictions.device

  predictions_at_most_certain_internal_tick = predictions.argmax(1)[torch.arange(B, device=device), where_most_certain].detach().cpu().numpy()
  accuracy = (targets.detach().cpu().numpy() == predictions_at_most_certain_internal_tick).mean()

  return accuracy