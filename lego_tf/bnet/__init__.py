"""Go-forward substrate built on the `bricknet` package.

This is the actively-developed side of the repo (the legacy from-scratch tokenizer in
`lego_tf/tokenize/` is kept only as reference). We layer our contribution -- native LEGO
tokenization, interactive prefix-completion, hierarchy/caption conditioning, finish-calibration --
on top of bricknet's pose-free connector-graph representation and collision scorer. The model is
trained directly on the native token stream (no text serialization).
"""

from lego_tf.bnet.trees import (
    catalog,
    coerce_colors,
    sample_tree,
    truncate_tree,
    brick_count,
)

__all__ = [
    "catalog",
    "coerce_colors",
    "sample_tree",
    "truncate_tree",
    "brick_count",
]
