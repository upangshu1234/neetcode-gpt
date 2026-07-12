import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict
    ) -> str:

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        generated = ""

        for i in range(new_chars):

            # 1. Crop context if needed
            x = context[:, -context_length:]

            # 2. Forward pass -> last token logits -> probabilities
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Do not alter this line
            generator.set_state(initial_state)

            # 3. Sample next token
            next_token = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator
            )

            # 4. Append token to context
            context = torch.cat((context, next_token), dim=1)

            # 5. Decode token
            generated += int_to_char[next_token.item()]

        return generated