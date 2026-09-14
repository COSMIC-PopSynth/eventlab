"""
Helpful functions to renormalize samples constructed with different IMFs.
"""

import numpy as np
from scipy.integrate import quad

__all__ = ['kroupa_01', 'salpeter_55', 'build_broken_pwr_law_imf', 'get_mean_mass']


def build_broken_pwr_law_imf(alphas, mcuts):
    """
    Build a normalized, continuous, piecewise power-law IMF.

    Each "piece" i is a power law ``m**alphas[i]`` valid between
    ``mcuts[i]`` and ``mcuts[i+1]``. The pieces are scaled relative to
    each other so the resulting function is continuous at every
    breakpoint, and the whole function is divided by its total integral
    so that it is a properly normalized probability density function
    (i.e. it integrates to 1 over ``[mcuts[0], mcuts[-1]]``).

    This follows the same construction used in COSMIC's
    ``Sample.sample_primary`` for ``kroupa93``, ``kroupa01``, and
    ``salpeter55``.

    Parameters
    ----------
    alphas : array-like
        Power law exponents for each piece, e.g. ``[-1.3, -2.3]``.
        There must be exactly one fewer exponent than breakpoints.
    mcuts : array-like, units of Msun
        Mass breakpoints separating each piece, e.g. ``[0.08, 0.5, 150.0]``.
        Must be sorted in increasing order and have one more element
        than `alphas`.

    Returns
    -------
    imf : callable
        A function ``imf(m)`` that returns the normalized probability
        density at mass `m` (in Msun). Accepts either a scalar or an
        array-like of masses. Returns 0 outside ``[mcuts[0], mcuts[-1]]``.

    Raises
    ------
    ValueError
        If `mcuts` does not have exactly one more element than `alphas`.

    Examples
    --------
    >>> imf = build_broken_pwr_law_imf([-1.3, -2.3], [0.08, 0.5, 150.0])
    >>> imf(1.0)
    """
    alphas = [float(a) for a in alphas]
    mcuts = [float(c) for c in mcuts]

    if len(mcuts) != len(alphas) + 1:
        raise ValueError(
            "mcuts must have exactly one more element than alphas "
            f"(got {len(alphas)} alphas and {len(mcuts)} mcuts)"
        )

    n_pieces = len(alphas)

    # --- Step 1: relative scale of each piece, so the function is ---
    # --- continuous across every breakpoint.                      ---
    coeffs = [1.0]
    for i in range(1, n_pieces):
        break_mass = mcuts[i]
        prev_alpha = alphas[i - 1]
        this_alpha = alphas[i]
        coeffs.append(coeffs[i - 1] * break_mass ** (prev_alpha - this_alpha))

    # --- Step 2: total integral of the un-normalized function, so ---
    # --- we can divide by it and make this a proper pdf.          ---
    total = 0.0
    for i in range(n_pieces):
        lo, hi = mcuts[i], mcuts[i + 1]
        alpha = alphas[i]
        coeff = coeffs[i]
        if alpha == -1.0:
            piece_integral = coeff * np.log(hi / lo)
        else:
            g = alpha + 1.0
            piece_integral = coeff / g * (hi ** g - lo ** g)
        total += piece_integral

    def _value_at_one_mass(mass):
        """Evaluate the normalized IMF at a single mass (a plain float)."""
        for i in range(n_pieces):
            lo, hi = mcuts[i], mcuts[i + 1]
            is_last_piece = (i == n_pieces - 1)
            if is_last_piece:
                in_this_piece = (mass >= lo) and (mass <= hi)
            else:
                in_this_piece = (mass >= lo) and (mass < hi)
            if in_this_piece:
                return coeffs[i] * mass ** alphas[i] / total
        return 0.0

    def imf(m):
        """Evaluate the normalized IMF at a scalar mass or array of masses."""
        if np.ndim(m) == 0:
            return _value_at_one_mass(float(m))
        # Loop explicitly, one mass at a time -- slower, but leaves no
        # room for a masking/indexing mistake to sneak in.
        values = [_value_at_one_mass(float(mass)) for mass in np.ravel(m)]
        return np.array(values).reshape(np.shape(m))

    return imf


# Built once at import time using the same breakpoints/exponents and
# default mass range as COSMIC's Sample class.
_kroupa_01_imf = build_broken_pwr_law_imf(alphas=[-1.3, -2.3], mcuts=[0.08, 0.5, 150.0])
_salpeter_55_imf = build_broken_pwr_law_imf(alphas=[-2.35], mcuts=[0.08, 150.0])


def kroupa_01(m):
    """
    Kroupa (2001) initial mass function, normalized between 0.08 and 150 Msun.

    See `Kroupa (2001) <https://arxiv.org/abs/astro-ph/0009005>`_. Matches
    the ``kroupa01`` option in COSMIC's ``Sample.sample_primary``.

    Parameters
    ----------
    m : float or array-like
        Stellar mass (or masses), in Msun.

    Returns
    -------
    float or ndarray
        Normalized probability density at mass `m`. Zero outside
        ``[0.08, 150.0]`` Msun.
    """
    return _kroupa_01_imf(m)


def salpeter_55(m):
    """
    Salpeter (1955) initial mass function, normalized between 0.08 and 150 Msun.

    See `Salpeter (1955) <http://adsabs.harvard.edu/abs/1955ApJ...121..161S>`_.
    Matches the ``salpeter55`` option in COSMIC's ``Sample.sample_primary``.

    Parameters
    ----------
    m : float or array-like
        Stellar mass (or masses), in Msun.

    Returns
    -------
    float or ndarray
        Normalized probability density at mass `m`. Zero outside
        ``[0.08, 150.0]`` Msun.
    """
    return _salpeter_55_imf(m)


def get_mean_mass(imf_fn, mlow, mhigh):
    """
    Compute the mean stellar mass for an IMF over a mass range.

    Parameters
    ----------
    imf_fn : callable
        Initial mass function, ``imf_fn(m)``, giving the number density
        per unit mass at mass `m`. Should be properly normalized from
        `mlow` to `mhigh` to give the true mean mass of the IMF.
    mlow : float
        Lower mass bound of the integration.
    mhigh : float
        Upper mass bound of the integration.

    Returns
    -------
    mean_mass_result : float
        Mean stellar mass, computed as the integral of
        m*`imf_fn`(m)*dm over ``[mlow, mhigh]``.
    """
    mean_mass_result, _ = quad(lambda m: m * imf_fn(m), mlow, mhigh)
    return mean_mass_result