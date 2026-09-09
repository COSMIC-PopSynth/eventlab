"""
Compute the delay time distribution (DTD) of events from a user-provided set of time bins and an event_df (see `events.get_events`) which stores the timing and duration of events for a stellar population.
"""

__all__ = ['get_dtd']

import numpy as np
import pandas as pd

def get_dtd(event_df, bins, sample_mass, event_is_instantaneous=False):
    """
    Compute the delay time distribution (DTD) power in user-specified bins,
    from the interval dataframe produced by get_events.

    Parameters
    ----------
    event_df : pandas.DataFrame
        Output of `get_events`. Minimally a dataframe which contains the columns `'tphys_start'`, `'tphys_end'`.
    bins : array-like
        Monotonically increasing bin edges, e.g. `[0, 6, 12, 20]` -> bins
        [0,6), [6,12), [12,20).
    sample_mass : float
        Total sampled population mass used to normalize power
        (power = duration / (bin_width * `sample_mass`)).

    Returns
    -------
    dtd_df : pandas.DataFrame 
        DataFrame with one row per time bin. Include columns `bin_start`, `bin_end`, `power`.
    """
    bins = np.asarray(bins, dtype=float)
    if len(bins) < 2:
        raise ValueError('bins must have at least 2 edges')
    if np.any(np.diff(bins) <= 0):
        raise ValueError('bins must be strictly increasing')

    bins_with_edges = list(zip(bins[:-1], bins[1:]))

    def _power_per_bin(events):
        starts = events['tphys_start'].to_numpy()
        ends = events['tphys_end'].to_numpy()

        powers = []
        for t_min, t_max in bins_with_edges:
            # clip each interval to the bin boundaries before summing duration
            overlap_start = np.maximum(starts, t_min)
            overlap_end = np.minimum(ends, t_max)
            clipped_durations = np.clip(overlap_end - overlap_start, 0, None)

            total_duration = clipped_durations.sum()
            bin_width = t_max - t_min
            if event_is_instantaneous:
                # the dtd is simply events per unit mass. each event contributes a single 1 / sampl_mass of power to the bin.
                power = 1 / sample_mass
            else:
                # here we compute the power as the total duration of events in the bin, normalized by the bin width and sample mass. 
                power = total_duration / (bin_width * sample_mass)
            powers.append(float(power))

        return powers

    dtd = _power_per_bin(event_df)

    return pd.DataFrame({
        'bin_start': bins[:-1],
        'bin_end': bins[1:],
        'power': dtd,
    })
