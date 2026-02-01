# Events package
from .mention_detector import (
    calculate_cumulative_attention, detect_first_mention,
    detect_all_sector_mentions, detect_mention_clusters,
    build_mention_timeline
)
from .event_windows import (
    create_event_window, extract_window_returns,
    calculate_cumulative_returns, build_event_panel,
    add_pre_event_metrics, add_market_controls
)
from .align_text_price import (
    align_speech_to_market, create_sector_event_record,
    build_aligned_panel, calculate_abnormal_returns
)

__all__ = [
    # Mention detection
    'calculate_cumulative_attention', 'detect_first_mention',
    'detect_all_sector_mentions', 'detect_mention_clusters',
    'build_mention_timeline',
    # Event windows
    'create_event_window', 'extract_window_returns',
    'calculate_cumulative_returns', 'build_event_panel',
    'add_pre_event_metrics', 'add_market_controls',
    # Alignment
    'align_speech_to_market', 'create_sector_event_record',
    'build_aligned_panel', 'calculate_abnormal_returns'
]
