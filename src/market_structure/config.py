"""
Configuration and optional dependency management for LCA Analysis package.

This module handles optional imports (PyMC, prince) and provides feature flags
to allow graceful degradation when certain libraries aren't available.
"""

# Optional PyMC import with error tracking
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    PYMC_AVAILABLE = True
    PYMC_ERROR = None
except Exception as e:
    PYMC_AVAILABLE = False
    PYMC_ERROR = str(e)
    pm = None
    pt = None
    az = None

# Optional prince import for MCA
try:
    import prince
    PRINCE_AVAILABLE = True
except ImportError:
    PRINCE_AVAILABLE = False
    prince = None


def get_available_models() -> list:
    """
    Return list of available model options based on installed dependencies.
    
    Returns a list of model name strings that can be used in the Streamlit selectbox.
    Models requiring optional dependencies are only included if those dependencies
    are available.
    """
    # Core models that are always available (use numpy/scipy/sklearn)
    model_options = [
        "Latent Class Analysis (LCA)",
        "Factor Analysis (Tetrachoric)",
        "Bayesian Factor Model (VI)",
        "Non-negative Matrix Factorization (NMF)"
    ]
    
    # Add MCA if prince is installed
    if PRINCE_AVAILABLE:
        model_options.append("Multiple Correspondence Analysis (MCA)")
    
    # Add PyMC-based models if available
    if PYMC_AVAILABLE:
        model_options.extend([
            "Bayesian Factor Model (PyMC)",
            "Discrete Choice Model (PyMC)"
        ])
    
    return model_options


def get_model_help_text() -> str:
    """Return the help text for the model selection dropdown."""
    return """
    **LCA**: Discrete latent segments  
    **Tetrachoric FA**: Continuous latent factors (proper for binary)  
    **Bayesian FA (VI)**: Fast variational inference  
    **NMF**: Parts-based decomposition  
    **MCA**: PCA for categorical data - ideal for binary purchase data  
    **Bayesian FA (PyMC)**: Full MCMC with uncertainty  
    **Discrete Choice (PyMC)**: Model with household/product features
    """