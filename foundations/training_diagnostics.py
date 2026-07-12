import torch
import torch.nn as nn
from typing import List, Dict

class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        activations = []
        handles = []

        # Define a hook to capture the outputs of nn.Linear layers
        def hook_fn(module, input, output):
            activations.append(output.detach())

        # Register the hook on all nn.Linear layers
        for module in model.modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(hook_fn))

        # Run the forward pass without tracking gradients
        with torch.no_grad():
            model(x)

        # Remove hooks to clean up memory
        for handle in handles:
            handle.remove()

        # Compute statistics for each captured activation
        for act in activations:
            mean = act.mean().item()
            std = act.std().item()
            
            # act shape is (batch_size, out_features). dim=0 checks across all samples for each neuron
            is_dead = (act <= 0).all(dim=0)
            dead_fraction = is_dead.float().mean().item()
            
            stats.append({
                "mean": round(mean, 4),
                "std": round(std, 4),
                "dead_fraction": round(dead_fraction, 4)
            })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        
        # Zero gradients, then run a forward and backward pass
        model.zero_grad()
        out = model(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()

        # Iterate through the modules and extract gradients from nn.Linear layers
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                
                mean = grad.mean().item()
                std = grad.std().item()
                norm = grad.norm().item()
                
                stats.append({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "norm": round(norm, 4)
                })
                
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # 1. Check for dead neurons
        if any(stat['dead_fraction'] > 0.5 for stat in activation_stats):
            return 'dead_neurons'
            
        # 2. Check for exploding gradients
        if any(stat['norm'] > 1000 for stat in gradient_stats):
            return 'exploding_gradients'
            
        # 3. Check for vanishing gradients in the final layer
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
            
        # 4. Check activation standard deviations across all layers
        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
            if stat['std'] > 10.0:
                return 'exploding_gradients'
                
        # 5. Default to healthy
        return 'healthy'