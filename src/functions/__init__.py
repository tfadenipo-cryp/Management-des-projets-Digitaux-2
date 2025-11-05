from .load_data import load_data
from .search_by_power import search_by_power
from .search_by_vehicle_type import search_by_vehicle_type
from .search_by_type_and_power import search_by_type_and_power
from .variable_analysis import variable_analysis
from .bivariate_analysis import bivariate_analysis
from .main_dashboard import main as main_dashboard

from . import premium_predictor as premium_predictor_module   # module
from .premium_predictor import premium_predictor              # fonction

__all__ = [
    "load_data",
    "search_by_power",
    "search_by_vehicle_type",
    "search_by_type_and_power",
    "variable_analysis",
    "bivariate_analysis",
    "main_dashboard",
    "premium_predictor_module",
    "premium_predictor",
]


