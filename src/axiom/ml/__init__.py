"""Financial machine learning: labelling, validation, features, importance.

The models in :mod:`axiom.models` learn a mapping. This package is about the
three things that decide whether that mapping means anything, and which are
where financial ML almost always goes wrong:

*What is the label?* A fixed-horizon return ignores that the position would
have been stopped out on day two. :mod:`axiom.ml.labelling` labels by what
actually happens to a trade — profit target, stop, or time — not by an
arbitrary horizon.

*How is it validated?* Labels built from overlapping forward windows leak into
each other, so a random K-fold trains on the answer to its own test set.
:mod:`axiom.ml.crossval` purges and embargoes.

*What is a feature worth?* :mod:`axiom.ml.importance` measures it by damage
done when the feature is destroyed, out of sample, rather than by how often a
tree happened to split on it.

The methods are from López de Prado, *Advances in Financial Machine Learning*
(2018). They are implemented here from the published descriptions; no
third-party implementation was used or copied.
"""

from axiom.ml.classifier import LogisticClassifier, NotFittedError
from axiom.ml.crossval import (
    PurgedKFold,
    PurgedWalkForward,
    embargo_size,
    purged_train_indices,
)
from axiom.ml.features import (
    frac_diff_weights,
    fractional_difference,
    minimum_stationary_d,
    sample_uniqueness,
    time_decay_weights,
)
from axiom.ml.importance import FeatureImportance, mean_decrease_accuracy
from axiom.ml.labelling import (
    Barriers,
    Label,
    LabelSet,
    meta_label,
    triple_barrier_labels,
    volatility_target,
)
from axiom.ml.meta import (
    MetaLabelFilter,
    MetaTrainingReport,
    collect_signals,
    default_features,
    train_meta_filter,
)

__all__ = [
    "Barriers", "FeatureImportance", "Label", "LabelSet", "LogisticClassifier",
    "MetaLabelFilter", "MetaTrainingReport", "NotFittedError", "PurgedKFold",
    "PurgedWalkForward", "collect_signals", "default_features", "embargo_size",
    "frac_diff_weights", "fractional_difference", "mean_decrease_accuracy",
    "meta_label", "minimum_stationary_d", "purged_train_indices",
    "sample_uniqueness", "time_decay_weights", "train_meta_filter",
    "triple_barrier_labels", "volatility_target",
]
