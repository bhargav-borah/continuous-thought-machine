import torch
import torch.nn as nn
import math
import numpy as np
from .utils import compute_normalized_entropy


class Identity(nn.Module):
  """Identity layer, outputs the same tensor as the output."""
  def __init__():
    super().__init__()

  def forward(self, x):
    return x
  

class NeuronLevelModels(nn.Module):
  """
  Neuron Level Models which transform the pre-activation history, which contains
  pre-activations spanning the maximum number of internal ticks during training,
  into post-activation.
  """
  def __init__(self,
               d_model,
               d_out,
               history_length
               ):
    """
    Arguments:
      d_model: Dimension of pre-activations
      d_out: Dimension of post-activation
      history_length: Maximum number of internal ticks during training
    """
    super().__init__()
    self.w1 = nn.Parameter(
        torch.empty((history_length, d_out, d_model)).uniform_(
            -1 / math.sqrt(history_length + d_out),
             1 / math.sqrt(history_length + d_out)
        ),
        requires_grad=True
    )
    self.b1 = nn.Parameter(
        torch.zeros((1, d_out, d_model)),
        requires_grad=True
    )
    self.w2 = nn.Parameter(
        torch.empty((d_out, d_model)).uniform_(
            -1 / math.sqrt(d_out + d_model),
             1 / math.sqrt(d_out + d_model)
        ),
        requires_grad=True
    )
    self.b2 = nn.Parameter(
        torch.zeros((1, d_out)),
        requires_grad=True
    )

  def forward(self, x):
    """
    Forward pass of NeuronLevelModels layer, each neuron has its own MLP

    Arguments: 
        x: Input tensor representating pre-activations history.
    Returns:
        out: Output tensor representing post-activation at current internal tick.
    """
    out = torch.einsum('bdm,mhd->bhd', x, self.w1) + self.b1
    out = torch.einsum('bhd,hd->bh', out, self.w2) + self.b2

    return out
  
class ContinuousThoughtMachine(nn.Module):
  """
  Class for transformations of input data throught the architecture of a Continuous Thought Machine.

  Arguments:
      d_input: Dimension of pre-activation vector for a single internal tick, and that of the attention output vector concatenated to the pre-activation vector.
      d_model: Number of neuron-level models in the architecture.
      dropout: A scalar value representing the magnitude of dropout to be applied at any point in the architecture. 
      history_length: Maximum number of internal ticks during training for which a pre-activation history is maintained.
      num_heads: Number of heads for the attention layer.
      out_dims: Dimension of the output vector for each data point at each internal tick; equal to the number of classes we are dealing with
      n_synch_action: Number of neurons to be used for synchronisation matrix, which is in turn going to be used for creating the attention output to be concatenated to pre-activation tensor.
      n_synch_out: Number of neurons to be used for synchronosation matrix, which in turn be transformed into output logits.
      synapse: nn.Sequential object representing the synapse layer (default None).
      backbone: nn.Sequential object representing the backbone which transforms input images to input features (default None).
  """
  def __init__(self,
               d_input,
               d_model,
               dropout,
               history_length,
               num_heads,
               out_dims,
               n_synch_action,
               n_synch_out,
               synapse=None,
               backbone=None
               ):
    super().__init__()
    self.d_input = d_input
    self.d_model = d_model
    self.dropout = dropout
    self.history_length = history_length
    self.num_heads = num_heads
    self.out_dims = out_dims
    self.n_synch_action = n_synch_action
    self.n_synch_out = n_synch_out
    if synapse is not None:
      self.synapse = synapse
    else:
      self.synapse = nn.Sequential(
        nn.Dropout(self.dropout),
        nn.LazyLinear(self.d_model * 2),
        nn.GLU(),
        nn.LayerNorm(self.d_model)
      )
    # self.post_activation has been initialized as start_activated_state in the official code repo
    self.post_activation = nn.Parameter(
        torch.zeros((self.d_model)).uniform_(
            -1 / math.sqrt(self.d_model),
             1 / math.sqrt(self.d_model)
        ),
        requires_grad=True
    )
    self.pre_activations_history = nn.Parameter(
        torch.empty((self.d_model, self.history_length)).uniform_(
            -1 / math.sqrt(self.d_model + self.history_length),
             1 / math.sqrt(self.d_model + self.history_length)
        ),
        requires_grad=True
    )
    self.attention = nn.MultiheadAttention(
        self.d_input,
        self.num_heads,
        self.dropout,
        batch_first=True
    )
    self.kv_projector = nn.Sequential(
        nn.LazyLinear(self.d_input),
        nn.LayerNorm(self.d_input)
    )
    self.q_projector = nn.Sequential(
        nn.LazyLinear(self.d_input),
        nn.LayerNorm(self.d_input)
    )
    self.output_projector = nn.LazyLinear(self.out_dims)
    self.nlms = NeuronLevelModels(
        d_model=self.d_model,
        d_out=self.d_model,
        history_length=self.history_length
    )
    if backbone is None:
        self.backbone = nn.Sequential(
            nn.LazyConv2d(
                d_input,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(
                d_input,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    else:
        self.backbone = backbone

    # Dimensions of the representations for synchronisation. For n_synch neurons, the pairwise-product would 
    # be a symmetric matrix, so we need only (n_synch * (n_sync + 1)) // 2 (including the diagonal elements) dimensions
    self.synch_representation_size_action = (self.n_synch_action * (self.n_synch_action + 1)) // 2
    self.synch_representation_size_out = (self.n_synch_out * (self.n_synch_out + 1)) // 2

    self.set_synchronisation_parameters('out')
    self.set_synchronisation_parameters('action')

  def set_synchronisation_parameters(self, synch_type):
    """
    Intializes the learning decay parameters, which give more weight to activations from recent ticks.

    Argument:
        synch_type: String value, either 'action' or 'out', representating whether the synchronisation is for 
                    generating a tensor to be concatenated to pre-activation tensor ('action') or for generating
                    a tensor to be transformed to output logits ('out').
    """
    synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
    self.register_parameter(
        f'decay_params_{synch_type}',
        nn.Parameter(
            torch.zeros(synch_representation_size),
            requires_grad=True
        )
    )

  def compute_synchronisation(self, post_activation, decay_alpha, decay_beta, r, synch_type):
    """
    Computes synchonisation amongst chosen neurons.

    This method doesn't use the entire synchronisation matrix computed from the post-activation history, nor does it use a sampling of
    the neuron-interactions in the synchronisation matrix. It only relies on the post-activation tensor at the current internal tick. 
    The choice of this implementation, to lower computational requirements, is also the basis of the decision to not maintain a post-activation 
    history.

    This works fine for image classification on the MNIST dataset. For more complex problems, it would be worth trying the idea of using the entire (or at least a sample) 
    of the neuron-intearctions encoded in the synchronisation matrix. This implementation, using a recursive formulation, has been proposed in section K at the end of the paper. 
    
    Arguments:
        post_activation: Post-activation tensor at current internal tick
        decay_alpha: The numerator of the formula for recursive computation of synchronisation matrix.
        decay_beta: The denominator of the formula for recursive computation of synchronisation matrix.
        r: Learnable parameter deciding the rate of decay of post-activations from past internal ticks.
        synch_type: String value representing whether the synchronisation is to be used for computation of pre-activation or for logits.
    """
    if synch_type == 'action':
      n_synch = self.n_synch_action
      selected_left = selected_right = post_activation[:, -n_synch:]
    elif synch_type == 'out':
      n_synch = self.n_synch_out
      selected_left = selected_right = post_activation[:, :n_synch]

    outer = selected_left.unsqueeze(dim=2) * selected_right.unsqueeze(dim=1)
    i, j = torch.triu_indices(n_synch, n_synch)
    pairwise_product = outer[:, i, j]

    if decay_alpha is None or decay_beta is None:
      decay_alpha = pairwise_product
      decay_beta = torch.ones_like(pairwise_product)
    else:
      decay_alpha = r * decay_alpha + pairwise_product
      decay_beta = r * decay_beta + 1

    synchronisatioon = decay_alpha / torch.sqrt(decay_beta)
    return synchronisatioon, decay_alpha, decay_beta

  def compute_features(self, x):
    """
    Computes features to be fed to the CTM.

    Argument:
        x: Input tensor representing images batch(es).
    Returns:
        kv: Output tensor representing the keys and values to be used during attention computation.
    """
    input_features = self.backbone(x).flatten(2).transpose(1, 2)
    kv = self.kv_projector(input_features)

    return kv

  def compute_certainty(self, current_prediction):
    """
    Computes the certainty (and uncertainty) in the prediction produced at current internal tick.

    Arguments:
        current_prediction: Prediction, in terms of logits, at the current time step.
    Returns:
        current_certainty: Output tensor with certainty and uncertainty values.
    """
    ne = compute_normalized_entropy(current_prediction)
    current_certainty = torch.stack((ne, 1 - ne), -1)

    return current_certainty

  def forward(self, x, track=False):
    """
    Forward pass of ContinuousThoughtMachineLayer.

    Arguments:
        x: Input tensor representing batch(es) of images
        track: Whether to track pre-activations, post-activations, action synchronisation, 
               output synchronisation, and attention, for the purpose of visualization (default False).
    Returns:
        predictions: Predictions over the span of internal ticks, in terms of logits.
        certainties: Certainties (and uncertainties) for predictions over internal ticks.
        synch_out: Output synchronisation.
    """
    B = x.size(0)
    device = x.device

    if track:
      pre_activations_tracking = []
      post_activations_tracking = []
      synch_out_tracking = []
      synch_action_tracking = []
      attention_tracking = []

    kv = self.compute_features(x)

    pre_activations_history = self.pre_activations_history.unsqueeze(dim=0).expand(B, -1, -1)
    post_activation = self.post_activation.unsqueeze(dim=0).expand(B, -1)

    predictions = torch.empty(B, self.out_dims, self.history_length, device=device, dtype=x.dtype)
    certainties = torch.empty(B, 2, self.history_length, device=device, dtype=x.dtype)

    decay_alpha_action, decay_beta_action = None, None
    r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1)
    r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)
    _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(post_activation, None, None, r_out, synch_type='out')

    for tick_idx in range(self.history_length):
      synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
          post_activation,
          decay_alpha_action,
          decay_beta_action,
          r_action,
          synch_type='action'
      )

      q = self.q_projector(synchronisation_action).unsqueeze(dim=1)
      attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
      attn_out = attn_out.squeeze(dim=1)
      pre_synapse_input = torch.concat([attn_out, post_activation], dim=-1)
      pre_activation = self.synapse(pre_synapse_input)

      pre_activations_history = torch.concat([pre_activations_history[:, :, :-1], pre_activation.unsqueeze(dim=-1)], dim=-1)
      post_activation = self.nlms(pre_activations_history)

      synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
          post_activation,
          decay_alpha_out,
          decay_beta_out,
          r_out,
          synch_type='out'
      )

      current_prediction = self.output_projector(synchronisation_out)
      current_certainty = self.compute_certainty(current_prediction)

      predictions[..., tick_idx] = current_prediction
      certainties[..., tick_idx] = current_certainty

      if track:
        pre_activations_tracking.append(pre_activations_history[:, :, -1].detach().cpu().numpy())
        post_activations_tracking.append(post_activation.detach().cpu().numpy())
        attention_tracking.append(attn_weights.detach().cpu().numpy())
        synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
        synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

    if track:
      return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
    return predictions, certainties, synchronisation_out
  

