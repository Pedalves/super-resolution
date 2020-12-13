from invoke import Collection

from . import tests, mlp, baseline, rdn

namespace = Collection(
    tests,
    mlp,
    rdn,
    baseline
)
