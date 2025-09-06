import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from ctm.data import get_mnist
from ctm.models import ContinuousThoughtMachine
from ctm.train import train, update_training_curve_plot
from ctm.viz import make_gif

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_input = 128
    d_model = 128
    dropout = 0.0
    history_length = 15
    num_heads = 2
    out_dims = 10
    n_synch_action = 16
    n_synch_out = 16
    batch_size = 512
    iterations = 1000
    test_every = 50
    run_inference = True

    trainloader, testloader = get_mnist(batch_size=512)

    model = ContinuousThoughtMachine(
        d_input=d_input,
        d_model=d_model,
        dropout=dropout,
        history_length=history_length,
        num_heads=num_heads,
        out_dims=out_dims,
        n_synch_action=n_synch_action,
        n_synch_out=n_synch_out
    )

    model = train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        iterations=iterations,
        test_every=test_every,
        device=device
    )

    if run_inference:
        logdir = f'mnist_logs'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    model.eval()
    with torch.inference_mode():
        inputs, targets = next(iter(testloader))
        inputs, targets = inputs.to(device), targets.to(device)

        predictions, certainties, (synch_out_tracking, synch_action_tracking), \
        pre_activations_tracking, post_activations_tracking, attention = model(inputs, track=True)

        make_gif(
            predictions.detach().cpu().numpy(),
            certainties.detach().cpu().numpy(),
            targets.detach().cpu().numpy(),
            pre_activations_tracking,
            post_activations_tracking,
            attention,
            inputs.detach().cpu().numpy(),
            f'{logdir}/prediction.gif'
        )

if __name__ == '__main__':
    main()


