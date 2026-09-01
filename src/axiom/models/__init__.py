"""Deep sequence models for return forecasting.

Ported from the uploaded architecture's ``models/`` tree: the price
transformer, the temporal fusion transformer, the LSTM/GRU ensemble, the graph
network, and the PPO agent. PyTorch is an **optional** extra
(``pip install 'axiom[deep]'``); nothing in the core platform imports it, and
the whole package degrades to an honest error rather than a broken import when
it is absent.

Two contracts are imposed on everything here, both absent from the original and
both about the same failure.

*An unfitted model returns nothing.* The uploaded orchestrator called its RL
agent with ``state = torch.randn(50)`` — a literal random vector — and blended
the resulting "signal" into the combined prediction at weight 0.15. Random
input produces random output, and the ensemble had no way to tell that apart
from a view. Here, :meth:`DeepModel.predict` on an unfitted model raises, and
the adapter that feeds the alpha ensemble emits no signal at all rather than a
confident number derived from noise.

*A prediction is not a confidence.* A network's output magnitude is not a
probability, and treating it as one is how a 0.9 activation becomes 90%
conviction. The adapter derives confidence from out-of-sample validation error
recorded at fit time — the only thing that has any claim to measure how much
the model should be believed.

The models themselves are unchanged in architecture from the upload; the
corrections are at the boundary where their output becomes a trading decision.
"""

from axiom.models.base import DeepModel, ModelNotFittedError, TrainingReport, torch_available

__all__ = [
    "DeepModel",
    "ModelNotFittedError",
    "TrainingReport",
    "torch_available",
]
