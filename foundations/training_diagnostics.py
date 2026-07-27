import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        
        stats: List[Dict[str, float]] = []
        hooks = []

        def hook_fn(module, inputs, output):
            mean = output.mean().item()
            std = output.std().item()

            # A neuron = one output feature (last dim). It's "dead" if it's
            # <= 0 across every sample in the batch.
            flat = output.reshape(-1, output.shape[-1])          # (batch, features)
            never_fires = (flat <= 0).all(dim=0)                  # (features,) bool
            dead_fraction = never_fires.float().mean().item()

            stats.append({
                'mean': round(mean, 4),
                'std': round(std, 4),
                'dead_fraction': round(dead_fraction, 4),
            })

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        return stats
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
       

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()

        criterion = nn.MSELoss()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        stats: List[Dict[str, float]] = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                if grad is None:
                    stats.append({'mean': 0.0, 'std': 0.0, 'norm': 0.0})
                    continue
                stats.append({
                    'mean': round(grad.mean().item(), 4),
                    'std': round(grad.std().item(), 4),
                    'norm': round(torch.norm(grad).item(), 4),
                })

        return stats
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # 1. Dead neurons
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'

        # 2. Exploding gradients (any layer)
        for stat in gradient_stats:
            if stat['norm'] > 1000:
                return 'exploding_gradients'

        # 3. Vanishing gradients (last layer only)
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        # 4. Activation std checks (all layers)
        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
        for stat in activation_stats:
            if stat['std'] > 10.0:
                return 'exploding_gradients'

        # 5. Healthy
        return 'healthy'
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
