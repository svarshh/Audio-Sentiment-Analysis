import torch
from trial import AudioRNN
import os
import torch.serialization

ROOT = os.path.abspath(os.path.dirname(__file__))

full_model_path = os.path.join(ROOT, "model.pt")
weights_path = os.path.join(ROOT, "model_weights.pt")

# Allow AudioRNN to be used when unpickling
torch.serialization.add_safe_globals([AudioRNN])

# Load the full model (trusted source)
full_model = torch.load(full_model_path, map_location="cpu", weights_only=False)

# Save only the state_dict (weights)
torch.save(full_model.state_dict(), weights_path)
print(f"Saved weights to {weights_path}")