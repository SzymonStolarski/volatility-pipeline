from .garch_models import GARCHModel, make_garch
# ORDER MATTERS: xgb_models (xgboost) must be imported before lstm_models
# (torch). On macOS both wheels bundle their own libomp, and once torch's copy
# is loaded first every XGBoost fit in the process crashes (OMP error #179 /
# segfault). Loading xgboost first is safe in any later order of use. Scripts
# and tests should therefore import this package before importing torch.
from .xgb_models import XGBVolatilityModel, XGBHybridModel
from .lstm_models import LSTMVolatilityModel, LSTMHybridModel
from .realized_garch import RealizedGARCH, make_realized_garch
from .tuning import TuningCache, tuned_factory, TUNE_MODES
from .lstm_models import LSTM_SEARCH_SPACE
from .equations import (
    param_table,
    param_matrix,
    pvalue_matrix,
    asymmetry_summary,
    persistence,
    equation_lines,
    equations_markdown,
    scale_note,
)
