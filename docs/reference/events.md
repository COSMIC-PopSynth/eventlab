# events

- Define events over a `bcm` (or `bcm`-like) dataframe via a boolean mask created with the helper `build_mask` or your own custom mask. Extract contiguous time intervals per binary/star where the mask holds and select summary statistics to report on columns of interest during the events with `get_events`. Include key information about binary interaction (whether underwent RLOF/CEE/No interaction before/during event, what stages (kstars) accreted or donated at, whether merged and what merger_type was) using `include_interaction_info=True` (requires the `bpp`).

::: eventlab.events