"""
Define an event of interest, and generate a catalog of the event from a COSMIC bcm/bpp including useful, user-specified, properties of the event-hosting-binary before and during the event.
"""

__all__ = ['build_mask', 'get_events']

import numpy as np
import pandas as pd

VALID_OPS = {'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal'}
VALID_STATS = {'max', 'min', 'median', 'mean', 'first', 'last'}


def build_mask(events, mask_list, mask_logic='and'):
    """
    Combine the list of (`column`, `op`, `val`) conditions in `mask_list` into a single boolean
    mask over `events`, using exact column names, according to the specified logic `mask_logic`.

    Parameters
    ----------
    events : pandas.DataFrame
        The dataframe to build the mask over (typically the COSMIC ``bcm`` with 
        sufficient time resolution, but any dataframe with the referenced columns
        works in principle).
    mask_list : list of tuple
        Each tuple is of the form ``(column, op, value)`` where ``op`` is
        one of ``'greater'``, ``'less'``, ``'greater_equal'``,
        ``'less_equal'``, ``'equal'``, ``'not_equal'``.
    mask_logic : str, optional
        ``'and'`` or ``'or'`` -- how to combine the individual conditions.
        Default is ``'and'``.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array the same length as ``events``.

    Examples
    --------
    >>> rsg_mask_1 = build_mask(bcm,
    ...                         [('lum_1', 'greater_equal', 1e4),
    ...                          ('lum_1', 'less_equal', 6e5),
    ...                          ('teff_1', 'greater_equal', 2900),
    ...                          ('teff_1', 'less_equal', 4500)],
    ...                         mask_logic='and')
    >>> rsg_mask_2 = build_mask(bcm,
    ...                         [('lum_2', 'greater_equal', 1e4),
    ...                          ('lum_2', 'less_equal', 6e5),
    ...                          ('teff_2', 'greater_equal', 2900),
    ...                          ('teff_2', 'less_equal', 4500)],
    ...                         mask_logic='and')
    >>> rsg_mask = rsg_mask_1 | rsg_mask_2
    >>> mass_transfer_mask = build_mask(bcm,
    ...                                 [('RRLO_1', 'greater', 1),
    ...                                  ('RRLO_2', 'greater', 1)],
    ...                                 mask_logic='or')
    """
    if mask_logic not in ('and', 'or'):
        raise ValueError("mask_logic must be 'and' or 'or'")

    if not mask_list:
        raise ValueError("mask_list must contain at least one (column, op, value) tuple")

    mask = np.ones(len(events), dtype=bool) if mask_logic == 'and' else np.zeros(len(events), dtype=bool)

    for mask_piece in mask_list:
        if len(mask_piece) != 3:
            raise ValueError(f"Each mask piece must be a tuple of (column, op, value), got {mask_piece}")
        
        col, op, val = mask_piece

        if col not in events.columns:
            raise KeyError(f"Column '{col}' not found in events dataframe")

        if op not in VALID_OPS:
            raise ValueError(f"Unsupported operation '{op}'. Must be one of {VALID_OPS}")

        if op == 'greater':
            cond = events[col] > val
        elif op == 'less':
            cond = events[col] < val
        elif op == 'greater_equal':
            cond = events[col] >= val
        elif op == 'less_equal':
            cond = events[col] <= val
        elif op == 'equal':
            cond = events[col] == val
        elif op == 'not_equal':
            cond = events[col] != val
        else:
            raise ValueError(f'Unsupported operation: {op}')

        mask = (mask & cond) if mask_logic == 'and' else (mask | cond)

    return np.asarray(mask)


def _validate_stats_dict(stats_dict, available_columns):
    """Check that a colname -> [stats] dict only references known columns/stats."""
    if not stats_dict:
        return {}

    missing_cols = set(stats_dict) - set(available_columns)
    if missing_cols:
        raise KeyError(f"Columns not found in bcm: {sorted(missing_cols)}")

    for col, stat_list in stats_dict.items():
        invalid = set(stat_list) - VALID_STATS
        if invalid:
            raise ValueError(f"Unsupported stats {invalid} for column '{col}'. Must be from {VALID_STATS}")

    return stats_dict


def _summarize_block(block, stats_dict):
    """Build one interval record: start/end time plus requested column stats."""
    record = {
        'tphys_start': float(block['tphys'].iloc[0]),
        'tphys_end': float(block['tphys'].iloc[-1]),
        'duration': float(block['tphys'].iloc[-1] - block['tphys'].iloc[0]),
    }
    for col, stat_list in stats_dict.items():
        values = block[col]
        for stat in stat_list:
            if stat == 'max':
                record[f'{col}_max'] = float(values.max())
            elif stat == 'min':
                record[f'{col}_min'] = float(values.min())
            elif stat == 'median':
                record[f'{col}_median'] = float(values.median())
            elif stat == 'mean':
                record[f'{col}_mean'] = float(values.mean())
            elif stat == 'first':
                record[f'{col}_first'] = float(values.iloc[0])
            elif stat == 'last':
                record[f'{col}_last'] = float(values.iloc[-1])
    return record


def _intervals_for_bin_num(group, stats_dict=None):
    """
    Find contiguous stretches (in sorted tphys order) where group['_mask']
    is True, for a single binary system. group must already contain a
    boolean '_mask' column. Returns a list of interval dicts.
    """
    if len(group) == 0:
        return []

    stats_dict = stats_dict or {}

    group = group.sort_values('tphys').reset_index(drop=True)
    selected_positions = np.flatnonzero(group['_mask'].to_numpy())

    if len(selected_positions) == 0:
        return []

    intervals = []
    start_pos = selected_positions[0]
    prev_pos = selected_positions[0]

    for pos in selected_positions[1:]:
        if pos != prev_pos + 1:
            block = group.iloc[start_pos:prev_pos + 1]
            intervals.append(_summarize_block(block, stats_dict))
            start_pos = pos
        prev_pos = pos

    block = group.iloc[start_pos:prev_pos + 1]
    intervals.append(_summarize_block(block, stats_dict))

    return intervals


def _collect_intervals(bcm, mask, stats_dict=None):
    """
    Compute intervals for one star (1 or 2) across every binary in bcm.

    mask       : boolean array/Series, same length as bcm, aligned by
                 position (e.g. the output of build_mask(bcm, ...), or any
                 boolean array/Series the user constructs themselves).
    stats_dict : dict of {column: [stat, ...]} -- exact bcm column names,
                 e.g. {'teff_1': ['min', 'max'], 'lum_1': ['mean']}.
    """

    mask = np.asarray(mask)
    if len(mask) != len(bcm):
        raise ValueError(f"mask length ({len(mask)}) does not match bcm length ({len(bcm)})")

    stats_dict = _validate_stats_dict(stats_dict, bcm.columns)

    required_cols = set(stats_dict) | {'tphys', 'bin_num'}
    missing = required_cols - set(bcm.columns)
    if missing:
        raise KeyError(f"Columns not found in bcm: {sorted(missing)}")

    events = bcm[list(required_cols)].copy()
    events['_mask'] = mask

    results = []
    for bin_num, group in events.groupby('bin_num', sort=False):
        intervals = _intervals_for_bin_num(group, stats_dict=stats_dict)
        for interval in intervals:
            results.append({'bin_num': bin_num, **interval})

    return results

def _add_interaction_info(events, bpp, bcm, no_merger_value='-001'):
    """
    Augment an events dataframe (bin_num, tphys_start, tphys_end, ...)
    with pre-/during-event interaction and merger history, computed per row,
    including information about each star's history.

    Adds
    ----
    pre_event_interactions       : 'None' | 'RLOF' | 'CEE'
    during_event_interactions    : 'None' | 'RLOF' | 'CEE'
    pre_event_merger             : bool
    during_event_merger          : bool  (False if pre_event_merger is True)
    pre_event_merger_type        : str -- the merger_type value if merged pre-event, else 'None'
    during_event_merger_type     : str -- the merger_type value if merged during event, else 'None'
                                    (always 'None' if pre_event_merger is True)
    pre_event_donor_kstars_*       : str (e.g. '1-3-5') or 'None'
    pre_event_accretor_kstars_*    : str or 'None'
    during_event_donor_kstars_*    : str or 'None'
    during_event_accretor_kstars_* : str or 'None'

    Notes
    -----
    - 'merger' is read off bcm's merger_type column: any value != no_merger_value
      ('-001' by default) counts as merged.
    - CEE takes precedence over RLOF if a window contains both (matches the
      original convention: any evol_type==7 -> 'CEE', else evol_type==3 -> 'RLOF').
    - columns marked with a * are per-star, e.g. pre_event_donor_kstars_1, pre_event_donor_kstars_2.
    """
    events = events.copy()

    bin_nums = events['bin_num'].unique()
    bpp = bpp[bpp['bin_num'].isin(bin_nums)]
    bcm = bcm[bcm['bin_num'].isin(bin_nums)]

    bpp_groups = {bin_num: group for bin_num, group in bpp.groupby('bin_num')}
    bcm_groups = {bin_num: group for bin_num, group in bcm.groupby('bin_num')}

    def interaction_label(sub_bpp):
        if sub_bpp.empty:
            return 'None'
        types = set(sub_bpp['evol_type'])
        if 7 in types:
            return 'CEE'
        if 3 in types:
            return 'RLOF'
        return 'None'

    def kstar_list(sub_bpp, kstar_col):
        if sub_bpp.empty:
            return 'None'
        vals = sorted(sub_bpp[kstar_col].unique())
        return '-'.join(map(str, vals)) if vals else 'None'

    def merger_type_value(sub_bcm):
        """First non-default merger_type in the window, or 'None' if none merged."""
        merged_rows = sub_bcm[sub_bcm['merger_type'] != no_merger_value]
        if merged_rows.empty:
            return 'None'
        return merged_rows['merger_type'].iloc[0]

    records = []
    for _, row in events.iterrows():
        bin_num = row['bin_num']

        bpp_bin = bpp_groups.get(bin_num, bpp.iloc[0:0])
        bcm_bin = bcm_groups.get(bin_num, bcm.iloc[0:0])

        pre_bpp = bpp_bin[bpp_bin['tphys'] < row['tphys_start']]
        during_bpp = bpp_bin[(bpp_bin['tphys'] >= row['tphys_start']) &
                              (bpp_bin['tphys'] <= row['tphys_end'])]

        pre_interactions = interaction_label(pre_bpp[pre_bpp['evol_type'].isin([3, 7])])
        during_interactions = interaction_label(during_bpp[during_bpp['evol_type'].isin([3, 7])])

        # get donation history for both stars
        pre_event_donor_kstars, pre_event_accretor_kstars = [], []
        during_event_donor_kstars, during_event_accretor_kstars = [], []

        for star in [1,2]:
            kstar_col = f'kstar_{star}'
            rrlo_col = f'RRLO_{star}'
            companion = 2 if star == 1 else 1
            rrlo_companion_col = f'RRLO_{companion}'

            pre_donor_kstars = kstar_list(pre_bpp[pre_bpp[rrlo_col] > 1], kstar_col) # star n is donating, get star n kstar
            pre_event_donor_kstars.append(pre_donor_kstars)

            pre_accretor_kstars = kstar_list(pre_bpp[pre_bpp[rrlo_companion_col] > 1], kstar_col) # star n is accreting (rrlo_m > 1), get star n kstar
            pre_event_accretor_kstars.append(pre_accretor_kstars)

            during_donor_kstars = kstar_list(during_bpp[during_bpp[rrlo_col] > 1], kstar_col) # star n is donating, get star n kstar
            during_event_donor_kstars.append(during_donor_kstars)

            during_accretor_kstars = kstar_list(during_bpp[during_bpp[rrlo_companion_col] > 1], kstar_col) # star n is accreting (rrlo_m > 1), get star n kstar
            during_event_accretor_kstars.append(during_accretor_kstars)

        pre_bcm = bcm_bin[bcm_bin['tphys'] < row['tphys_start']]
        during_bcm = bcm_bin[(bcm_bin['tphys'] >= row['tphys_start']) &
                              (bcm_bin['tphys'] <= row['tphys_end'])]

        pre_merger_type = merger_type_value(pre_bcm)
        pre_merger = pre_merger_type != 'None'

        if pre_merger:
            during_merger_type = 'None'
            during_merger = False
        else:
            during_merger_type = merger_type_value(during_bcm)
            during_merger = during_merger_type != 'None'

        records.append({
            'pre_event_interactions': pre_interactions,
            'during_event_interactions': during_interactions,
            'pre_event_merger': pre_merger,
            'during_event_merger': during_merger,
            'pre_event_merger_type': pre_merger_type,
            'during_event_merger_type': during_merger_type,
            'pre_event_donor_kstars_1': pre_event_donor_kstars[0],
            'pre_event_donor_kstars_2': pre_event_donor_kstars[1],
            'pre_event_accretor_kstars_1': pre_event_accretor_kstars[0],
            'pre_event_accretor_kstars_2': pre_event_accretor_kstars[1],
            'during_event_donor_kstars_1': during_event_donor_kstars[0],
            'during_event_donor_kstars_2': during_event_donor_kstars[1],
            'during_event_accretor_kstars_1': during_event_accretor_kstars[0],
            'during_event_accretor_kstars_2': during_event_accretor_kstars[1],
        })

    info_df = pd.DataFrame(records, index=events.index)
    return pd.concat([events, info_df], axis=1)

def get_events(bcm,
               mask,
               stats_dict=None,
               include_interaction_info=False,
               bpp=None):
    """
    Generate an event catalog: one row per contiguous interval where
    ``mask`` holds (event may be instantaneous like CCSNe without issue), for each binary in ``bcm``.

    Parameters
    ----------
    bcm : pandas.DataFrame
        The COSMIC ``bcm`` table (or a similar dataframe with ``bin_num`` and
        ``tphys`` columns) to extract intervals from.
    mask : numpy.ndarray
        Boolean array, same length as ``bcm``, defining which rows count
        as "in the event." Build with :func:`build_mask` or construct by
        hand for arbitrary custom logic.
    stats_dict : dict, optional
        Dictionary of ``{column: [stat, ...]}`` specifying summary
        statistics to compute over each interval, e.g.
        ``{'teff_1': ['min', 'max'], 'lum_2': ['mean'], 'rad_2': ['median'], 'sep': ['first', 'last']}``.
        Columns can be any column in ``bcm``, independent of what was used
        to build ``mask``. Stat options are ``'max'``, ``'min'``,
        ``'median'``, ``'mean'``, ``'first'``, ``'last'``.
    include_interaction_info : bool, optional
        If True, augment each interval with pre-/during-event interaction
        and merger history. Requires ``bpp`` to be provided. Default is
        False. Adds the following columns:

        - ``pre_event_interactions``: interaction type before the event
          -- 'None | 'RLOF' | 'CEE' (CEE takes precedence over RLOF if both occurred).
        - ``during_event_interactions``: interaction type during the event
          -- 'None' | 'RLOF' | 'CEE' (CEE takes precedence over RLOF if both occurred).
        - ``pre_event_merger``: True if a merger occurred before the event.
        - ``during_event_merger``: True if a merger occurred during the
          event (False if ``pre_event_merger`` is True).
        - ``pre_event_merger_type``: the ``merger_type`` value if merged
          pre-event, else 'None'.
        - ``during_event_merger_type``: the ``merger_type`` value if
          merged during the event, else 'None' (always 'None' if
          ``pre_event_merger`` is True).
        - ``pre_event_donor_kstars_*``: string of each kstar value at which the
          star (1 or 2) donated to its companion. Potential values include '' (star * did not donate),
          '1' (star * donated only on the MS) or longer '1-3-5' (mass transfer continued as star * was in all 3 phases).
        - ``pre_event_accretor_kstars_*``: same as ``pre_event_donor_kstars_*``, but the types of star * as it accreted from a companion.
        - ``during_event_donor_kstars_*``: same as ``pre_event_donor_kstars_*``, but for the event window.
        - ``during_event_accretor_kstars_*``: same as ``pre_event_accretor_kstars_*``, but for the event window.
    bpp : pandas.DataFrame, optional
        The COSMIC ``bpp`` dataframe. Required if ``include_interaction_info``
        is True (needed to evaluate MT stability); ignored otherwise.

    Returns
    -------
    event_df : pandas.DataFrame
        One row per contiguous interval, sorted by ``bin_num`` and
        ``tphys_start``, with columns:

        - ``bin_num``: the binary's identifier
        - ``tphys_start``: time the interval begins
        - ``tphys_end``: time the interval ends
        - ``duration``: ``tphys_end - tphys_start``
        - any ``<column>_<stat>`` columns requested via ``stats_dict``
        - if ``include_interaction_info=True``, the additional columns
          documented above.

    Examples
    --------
    >>> mask_1 = build_mask(bcm, [('kstar_1', 'equal', 7),
    ...                           ('mass_1', 'greater_equal', 2),
    ...                           ('mass_1', 'less_equal', 8)])
    >>> mask_2 = build_mask(bcm, [('kstar_2', 'equal', 7),
    ...                           ('mass_2', 'greater_equal', 2),
    ...                           ('mass_2', 'less_equal', 8)])
    >>> event_df_1 = get_events(bcm,
    ...                         mask_1,
    ...                         stats_dict={'teff_1': ['min', 'max'], 'lum_1': ['mean', 'median'], 'sep': ['first', 'last']},
    ...                         include_interaction_info=True,
    ...                         bpp=bpp)
    >>> event_df_2 = get_events(bcm,
    ...                         mask_2,
    ...                         stats_dict={'teff_2': ['min', 'max'], 'lum_2': ['mean', 'median'], 'sep': ['first', 'last']},
    ...                         include_interaction_info=True,
    ...                         bpp=bpp)
    >>> event_df_1['star'] = 1
    >>> event_df_2['star'] = 2
    >>> event_df = pd.merge(event_df_1, event_df_2, how='outer').sort_values(['bin_num', 'tphys_start']).reset_index(drop=True)
    """

    records = []
    if mask is not None:
        records.extend(_collect_intervals(
            bcm, mask, stats_dict=stats_dict,
        ))

    base_cols = ['bin_num', 'tphys_start', 'tphys_end', 'duration']
    if not records:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(records)
    other_cols = sorted(c for c in df.columns if c not in base_cols)
    df = df[base_cols + other_cols]
    df = df.sort_values(['bin_num', 'tphys_start']).reset_index(drop=True)

    if include_interaction_info:
        if bpp is None:
            raise ValueError("bpp must be provided if include_interaction_info is True")
        df = _add_interaction_info(df, bpp, bcm)

    return df