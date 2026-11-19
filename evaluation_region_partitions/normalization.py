"""Shared luminosity scaling; exposures are in inverse femtobarns."""
import math


def exposure_scale(target, source):
    """Return target/source, rejecting absent, zero and non-finite exposures."""
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or x <= 0 for x in (target, source)):
        raise ValueError('Yield normalization requires positive finite exposures')
    scale = target / source
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Non-finite or underflowed exposure ratio')
    return scale
