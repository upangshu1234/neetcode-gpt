import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []

        with torch.no_grad():
            out = x

            for layer in model:
                out = layer(out)

                if isinstance(layer, nn.ReLU):
                    # A neuron is dead if it is zero for every sample in the batch.
                    if out.dim() == 1:
                        dead = (out == 0)
                    else:
                        dead = (out == 0).all(dim=0)

                    dead_fraction = dead.float().mean().item()
                    dead_fractions.append(round(dead_fraction, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # 1. Use LeakyReLU if any layer has too many dead neurons.
        if any(df > 0.5 for df in dead_fractions):
            return "use_leaky_relu"

        # 2. Reinitialize if the first layer is already problematic.
        if dead_fractions and dead_fractions[0] > 0.3:
            return "reinitialize"

        # 3. Reduce learning rate if dead fraction strictly increases with depth
        #    and the last layer has a significant dead fraction.
        if (
            len(dead_fractions) >= 2
            and all(dead_fractions[i] < dead_fractions[i + 1]
                    for i in range(len(dead_fractions) - 1))
            and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"

        # 4 & 5. Healthy
        return "healthy"