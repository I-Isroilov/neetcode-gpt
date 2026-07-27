import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions: List[float] = []
        hooks = []

        def hook_fn(module, inputs, output):
            # Treat the last dimension as the "neuron" axis (matches
            # standard Linear -> ReLU stacks).
            flat = output.reshape(-1, output.shape[-1])   # (batch, features)
            never_fires = (flat == 0).all(dim=0)            # (features,) bool
            dead_fraction = never_fires.float().mean().item()
            dead_fractions.append(round(dead_fraction, 4))

        for layer in model.modules():
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        return dead_fractions
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
       

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if not dead_fractions:
            return 'healthy'

        # 1. Any layer badly dead
        if any(frac > 0.5 for frac in dead_fractions):
            return 'use_leaky_relu'

        # 2. First layer badly dead
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. Strictly increasing with depth AND last layer > 0.1
        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )
        if strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        # 4. All layers comfortably healthy
        if max(dead_fractions) < 0.1:
            return 'healthy'

        # 5. Catch-all
        return 'healthy'
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        
