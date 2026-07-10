"""
Compatibility shim: expose `sklearn._loss` as top-level `_loss`.

This helps unpickling model files that reference the module name
`_loss` (older/newer sklearn installs sometimes change import paths).
"""
try:
    # Preferred: import the packaged module and export its names
    from sklearn._loss import *  # type: ignore
except Exception:
    # Fallback: import by full name and copy attributes
    import importlib
    try:
        _mod = importlib.import_module('sklearn._loss')
        for _k, _v in _mod.__dict__.items():
            if not _k.startswith('__'):
                globals()[_k] = _v
    except Exception:
        # If all fails, define a minimal placeholder to avoid ImportError during unpickle.
        # The unpickling process may still fail if real implementations are required.
        class _Placeholder:
            pass
    else:
        # Provide compatibility aliases for older pickles that referenced C-extension
        # classes named like `CyHalfSquaredError` (older sklearn versions).
        try:
            # Map a few expected 'Cy' prefixed names to the current implementations
            g = globals()
            if 'HalfSquaredError' in g and 'CyHalfSquaredError' not in g:
                g['CyHalfSquaredError'] = g['HalfSquaredError']
            if 'HalfGammaLoss' in g and 'CyHalfGammaLoss' not in g:
                g['CyHalfGammaLoss'] = g['HalfGammaLoss']
            if 'HalfPoissonLoss' in g and 'CyHalfPoissonLoss' not in g:
                g['CyHalfPoissonLoss'] = g['HalfPoissonLoss']
        except Exception:
            pass
# Ensure legacy 'Cy*' aliases exist even when top-level import succeeded
try:
    g = globals()
    if 'HalfSquaredError' in g and 'CyHalfSquaredError' not in g:
        g['CyHalfSquaredError'] = g['HalfSquaredError']
    if 'HalfGammaLoss' in g and 'CyHalfGammaLoss' not in g:
        g['CyHalfGammaLoss'] = g['HalfGammaLoss']
    if 'HalfPoissonLoss' in g and 'CyHalfPoissonLoss' not in g:
        g['CyHalfPoissonLoss'] = g['HalfPoissonLoss']
except Exception:
    pass
