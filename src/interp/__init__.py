# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Interpretability infrastructure for JEPA vs baseline comparison
# v0.9.0: + Interpretability Index, robustness, ground truth validation
#
# Module structure (20 modules):
#   sae.py                  — SparseAutoencoder (TopK, dead feature resampling, SAETrainer)
#   structural_probe.py     — StructuralProbe (Hewitt & Manning 2019)
#   causal_intervention.py  — direction_ablation, feature_steering, activation_patching
#   disentanglement.py      — DCI, MIG, SAP, Modularity scores
#   compare.py              — RepresentationComparator, full_comparison_report
#   probing_complexity.py   — ProbingComplexityCurve, LinguisticProbeTasks
#   polysemanticity.py      — PolysemanticityIndex, SuperpositionIndex, FeatureDeduplicationScore
#   causal_scrubbing.py     — CausalScrubber, FeatureHypothesis, InterventionPredictabilityScorer
#   representation_geometry.py — RepresentationGeometry, GeometryDegradationTest
#   feature_composition.py  — FeatureCompositionScore, FeatureInterferenceScore
#   statistical_tests.py    — BootstrapCI, PairedPermutationTest, MultipleComparisonCorrection,
#                              EffectSize, BayesianComparison, MetricComparisonReport
#   information_theory.py   — MINEEstimator, InfoNCEEstimator, ConditionalMIEstimator,
#                              RepresentationCompression, InformationPlane
#   layer_analysis.py       — LayerwiseProbe, LayerwiseCKA, LayerwiseGeometry, LayerRoutingAnalysis
#   stability.py            — TrainingStability, LossStability, EarlyStoppingAdvantage,
#                              CheckpointConsistency
#   probe_generalization.py — ProbeGeneralizationTest, ProbeSelectivityTest,
#                              StructuralProbeGeneralization

from .sae import SparseAutoencoder, SAETrainer
from .structural_probe import StructuralProbe
from .causal_intervention import (
    direction_ablation,
    feature_steering,
    activation_patching,
    intervention_predictability_score,
    CausalIntervention,
)
from .disentanglement import (
    DCIMetrics,
    SAPScore,
    MIGScore,
    ModularityScore,
    compute_all_disentanglement_metrics,
)
from .compare import RepresentationComparator, extract_linguistic_features
from .probing_complexity import ProbingComplexityCurve, LinguisticProbeTasks
from .polysemanticity import (
    PolysemanticityIndex,
    SuperpositionIndex,
    FeatureDeduplicationScore,
)
from .causal_scrubbing import (
    CausalScrubber,
    FeatureHypothesis,
    InterventionPredictabilityScorer,
)
from .representation_geometry import (
    RepresentationGeometry,
    GeometryDegradationTest,
)
from .feature_composition import (
    FeatureCompositionScore,
    FeatureInterferenceScore,
)
from .statistical_tests import (
    BootstrapCI,
    PairedPermutationTest,
    MultipleComparisonCorrection,
    EffectSize,
    BayesianComparison,
    MetricComparisonReport,
)
from .information_theory import (
    MINEEstimator,
    InfoNCEEstimator,
    ConditionalMIEstimator,
    RepresentationCompression,
    InformationPlane,
)
from .layer_analysis import (
    LayerwiseProbe,
    LayerwiseCKA,
    LayerwiseGeometry,
    LayerRoutingAnalysis,
)
from .stability import (
    TrainingStability,
    LossStability,
    EarlyStoppingAdvantage,
    CheckpointConsistency,
)
from .probe_generalization import (
    ProbeGeneralizationTest,
    ProbeSelectivityTest,
    StructuralProbeGeneralization,
)
from .visualization import (
    radar_chart,
    layer_heatmap,
    bar_chart_with_errors,
    probing_complexity_curve,
    convergence_plot,
    ablation_comparison_chart,
    scaling_law_plot,
    robustness_curve,
    information_plane,
)
from .ablation import (
    AblationConfig,
    ABLATION_CONFIGS,
    AblatedModel,
    AblationStudy,
)
from .scaling import (
    ScalingAnalysis,
    ComputeOptimalScale,
    InterpretabilityEfficiency,
)
from .interpretability_index import InterpretabilityIndex
from .robustness import (
    RepresentationRobustness,
    RobustnessBattery,
    token_dropout,
    token_substitution,
    token_permutation,
    span_corruption,
)
from .ground_truth import (
    SyntheticStructuredModel,
    GroundTruthValidation,
)

__all__ = [
    # SAE
    'SparseAutoencoder', 'SAETrainer',
    # Structural
    'StructuralProbe',
    # Causal intervention
    'direction_ablation', 'feature_steering', 'activation_patching',
    'intervention_predictability_score', 'CausalIntervention',
    # Disentanglement
    'DCIMetrics', 'SAPScore', 'MIGScore', 'ModularityScore',
    'compute_all_disentanglement_metrics',
    # Comparator
    'RepresentationComparator', 'extract_linguistic_features',
    # Probing complexity
    'ProbingComplexityCurve', 'LinguisticProbeTasks',
    # Polysemanticity
    'PolysemanticityIndex', 'SuperpositionIndex', 'FeatureDeduplicationScore',
    # Causal scrubbing
    'CausalScrubber', 'FeatureHypothesis', 'InterventionPredictabilityScorer',
    # Geometry
    'RepresentationGeometry', 'GeometryDegradationTest',
    # Composition
    'FeatureCompositionScore', 'FeatureInterferenceScore',
    # Statistical tests
    'BootstrapCI', 'PairedPermutationTest', 'MultipleComparisonCorrection',
    'EffectSize', 'BayesianComparison', 'MetricComparisonReport',
    # Information theory
    'MINEEstimator', 'InfoNCEEstimator', 'ConditionalMIEstimator',
    'RepresentationCompression', 'InformationPlane',
    # Layer analysis
    'LayerwiseProbe', 'LayerwiseCKA', 'LayerwiseGeometry', 'LayerRoutingAnalysis',
    # Stability
    'TrainingStability', 'LossStability', 'EarlyStoppingAdvantage',
    'CheckpointConsistency',
    # Probe generalization
    'ProbeGeneralizationTest', 'ProbeSelectivityTest', 'StructuralProbeGeneralization',
    # Visualization
    'radar_chart', 'layer_heatmap', 'bar_chart_with_errors',
    'probing_complexity_curve', 'convergence_plot',
    'ablation_comparison_chart', 'scaling_law_plot',
    'robustness_curve', 'information_plane',
    # Ablation
    'AblationConfig', 'ABLATION_CONFIGS', 'AblatedModel', 'AblationStudy',
    # Scaling
    'ScalingAnalysis', 'ComputeOptimalScale', 'InterpretabilityEfficiency',
    # Interpretability Index
    'InterpretabilityIndex',
    # Robustness
    'RepresentationRobustness', 'RobustnessBattery',
    'token_dropout', 'token_substitution', 'token_permutation', 'span_corruption',
    # Ground truth
    'SyntheticStructuredModel', 'GroundTruthValidation',
]
