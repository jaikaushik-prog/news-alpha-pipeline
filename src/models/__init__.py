# Models package
from .event_study import (
    calculate_abnormal_returns, calculate_car,
    event_study_single, event_study_batch,
    test_car_significance_by_sector, calculate_average_car_pattern,
    calculate_volume_weighted_attention, analyze_strategies
)
from .panel_regression import (
    prepare_regression_data, ols_regression,
    fixed_effects_regression, format_regression_table
)
from .classification import (
    create_binary_target, prepare_classification_features,
    train_classifier, get_feature_importance, predict_outperformance
)
from .hazard_model import (
    calculate_reaction_time, prepare_survival_data,
    fit_cox_model, kaplan_meier_by_group, compare_survival_log_rank
)

__all__ = [
    # Event study
    'calculate_abnormal_returns', 'calculate_car',
    'event_study_single', 'event_study_batch',
    'test_car_significance_by_sector', 'calculate_average_car_pattern',
    'calculate_volume_weighted_attention', 'analyze_strategies',
    # Panel regression
    'prepare_regression_data', 'ols_regression',
    'fixed_effects_regression', 'format_regression_table',
    # Classification
    'create_binary_target', 'prepare_classification_features',
    'train_classifier', 'get_feature_importance', 'predict_outperformance',
    # Hazard model
    'calculate_reaction_time', 'prepare_survival_data',
    'fit_cox_model', 'kaplan_meier_by_group', 'compare_survival_log_rank'
]
