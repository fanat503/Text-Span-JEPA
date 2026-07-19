# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0

from .jepa import TextSpanJEPA, TextSpanJEPAConfig
from .encoder import TextSpanJEPLEncoder
from .predictor import TextSpanJEPApredictor
from .decoder import TiedTokenDecoder
from .collapse import (
    VarianceRegularization,
    CovarianceRegularization,
    TargetCentering,
    CollapseDiagnostics,
)
