# Validation package
from .placebo_tests import (
    generate_placebo_dates, run_placebo_event_study,
    compare_actual_vs_placebo, generate_random_sector_assignment,
    run_permutation_test
)
from .shuffling import (
    shuffle_timestamps, shuffle_sector_probabilities,
    bootstrap_panel, bootstrap_confidence_interval,
    leave_one_year_out, jackknife_estimate
)
from .negative_controls import (
    non_mentioned_sectors_test, compare_mentioned_vs_non_mentioned,
    pre_trend_test, cross_sector_spillover_test
)

__all__ = [
    # Placebo tests
    'generate_placebo_dates', 'run_placebo_event_study',
    'compare_actual_vs_placebo', 'generate_random_sector_assignment',
    'run_permutation_test',
    # Shuffling
    'shuffle_timestamps', 'shuffle_sector_probabilities',
    'bootstrap_panel', 'bootstrap_confidence_interval',
    'leave_one_year_out', 'jackknife_estimate',
    # Negative controls
    'non_mentioned_sectors_test', 'compare_mentioned_vs_non_mentioned',
    'pre_trend_test', 'cross_sector_spillover_test'
]
