"""Supported infomation
Information about supported datasets, samplers, models, optimizers, criterions and metrics.
"""


SUPPORTED_DATASET = [
    "omniglot",
    "cifar10",
    "navict",
    "navict_cbow",
]

SUPPORTED_SAMPLER = [
    "shuffle_sampler",
    "balanced_batch_sampler",
]

SUPPORTED_MODEL = [
    "resnet18",
    "simple_cnn",
    "simple_lstm",
    "simple_rnn",
    "cbow_embedder",
    "simple_nn",
]

SUPPORTED_OPTIMIZER = [
    "adam",
]

SUPPORTED_CRITERION = [
    "cross_entropy",
    "nll_loss",
]

SUPPORTED_METRIC = [
    "classification",
]

SUPPORTED_TRAINER = [
    "default",
]

SUPPORTED_EMBEDDER = [
    "cbow",
]