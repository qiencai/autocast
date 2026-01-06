from .ensemble import CRPS, AlphaFairCRPS, FairCRPS
from .spatiotemporal import MAE, MSE, NMAE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity

__all__ = [
    "CRPS",
    "MAE",
    "MSE",
    "NMAE",
    "NMSE",
    "NRMSE",
    "RMSE",
    "VMSE",
    "VRMSE",
    "AlphaFairCRPS",
    "FairCRPS",
    "LInfinity",
]

ALL_METRICS = (MSE, MAE, NMAE, NMSE, RMSE, NRMSE, VMSE, VRMSE, LInfinity)
ALL_ENSEMBLE_METRICS = (CRPS, AlphaFairCRPS, FairCRPS)
