"""
Helpful functions for generating mock photometry for COSMIC, including extinction, converting absolute <-> apparent magnitudes, etc. Relies heavily on
COGSWORTH's observables.photometry (https://cogsworth.readthedocs.io/en/latest/tutorials/observables/photometry.html) module, which interpolates over MIST
bolometric corrections. Repackaged here for utility. Future work may try to integrate MESA Colors.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tarfile
import shutil
import logging
import requests
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Basically all this stuff is all nabbed from Cogsworth.
# I will try to import it from them in the future as much as possible.
# ---------------------------------------------------------------------------

def get_log_g(mass, radius):
    """Computes log of the surface gravity in cgs

    Parameters
    ----------
    mass : :class:`~astropy.units.Quantity` [mass]
        Mass of the star
    radius : :class:`~astropy.units.Quantity` [radius]
        Radius of the star

    Returns
    -------
    log g : :class:`~numpy.ndarray`
        Log of the surface gravity in cgs
    """
    g = const.G * mass / radius**2

    # avoid division by zero errors (for massless remnants)
    with np.errstate(divide='ignore'):
        return np.log10(g.cgs.value)


def get_absolute_bol_mag(lum):
    """Computes the absolute bolometric magnitude following
    IAU Resolution B2 (https://www.iau.org/news/announcements/detail/ann15023/)

    Parameters
    ----------
    lum : :class:`~astropy.units.Quantity` [luminosity]
        Luminosity of the star

    Returns
    -------
    M_bol : :class:`~numpy.ndarray`
        Absolute bolometric magnitude
    """
    zero_point_lum = 3.0128e28 * u.watt
    return -2.5 * np.log10(lum / zero_point_lum)


def get_apparent_mag(M_abs, distance):
    """Convert absolute magnitude to apparent magnitude

    Parameters
    ----------
    M_abs : :class:`~numpy.ndarray`
        Absolute magnitude
    distance : :class:`~astropy.units.Quantity` [length]
        Distance

    Returns
    -------
    m_app : :class:`~numpy.ndarray`
        Apparent magnitude
    """
    finite_distance = np.isfinite(distance)
    m_app = np.repeat(np.inf, len(distance))
    m_app[finite_distance] = M_abs[finite_distance] + 5 * np.log10(distance[finite_distance] / (10 * u.pc))
    return m_app

def get_absolute_mag(m_app, distance):
    """Convert apparent magnitude to absolute magnitude

    Parameters
    ----------
    m_app : :class:`~numpy.ndarray`
        Apparent magnitude
    distance : :class:`~astropy.units.Quantity` [length]
        Distance

    Returns
    -------
    M_abs : :class:`~numpy.ndarray`
        Absolute magnitude
    """
    M_abs = m_app - 5 * np.log10(distance / (10 * u.pc))
    return M_abs


# ---------------------------------------------------------------------------
# MIST bolometric correction grid, inlined from cogsworth.obs.mist
# ---------------------------------------------------------------------------

MIST_FILTER_SETS = {
    "UBVRIplus": [
        "Bessell_U", "Bessell_B", "Bessell_V", "Bessell_R", "Bessell_I", "2MASS_J", "2MASS_H", "2MASS_Ks",
        "Kepler_Kp", "Kepler_D51", "Hipparcos_Hp", "Tycho_B", "Tycho_V",
        "Gaia_G_DR2Rev", "Gaia_BP_DR2Rev", "Gaia_RP_DR2Rev", "Gaia_G_MAW", "Gaia_BP_MAWf", "Gaia_BP_MAWb",
        "Gaia_RP_MAW", "TESS", "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"
    ],
}

@dataclass
class MISTBolometricCorrectionGrid:
    """
    Download, cache, and ingest MIST bolometric correction grids.

    Parameters
    ----------
    bands : tuple[str]
        tuple of photometric bands to include (e.g. ("2MASS_J", "2MASS_Ks"),
        or ("Bessell_B", "Bessell_V", "Bessel_I"))
    cache_dir
        directory where tarballs, extracted files, and HDF5s are stored
        (default: ~/.MIST_bc_grids)
    """
    bands: tuple[str] = ("2MASS_J", "2MASS_Ks")
    cache_dir: Path = Path("~/.MIST_bc_grids").expanduser()
    rebuild: bool = False

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        needed_filter_sets = set()
        for band in self.bands:
            found_it = False
            for filter_set in MIST_FILTER_SETS:
                if band in MIST_FILTER_SETS[filter_set]:
                    needed_filter_sets.add(filter_set)
                    found_it = True
                    break

            if not found_it:
                raise KeyError(f"band '{band}' not found in any MIST filter set")

        self.needed_filter_sets = needed_filter_sets

        dfs = [self.load_hdf5(filter_set) for filter_set in self.needed_filter_sets]
        df_cols = ["Rv", *self.bands]
        bc_grid = pd.concat(dfs, axis=1, copy=False)[df_cols]
        self.bc_grid = bc_grid.loc[:, ~bc_grid.columns.duplicated()]

        self._build_interpolators()

    def download_filter_set(self, filter_set: str) -> Path:
        tarball_path = self.cache_dir / f"{filter_set}.txz"

        if tarball_path.exists() and not self.rebuild:
            return tarball_path

        url = f"https://waps.cfa.harvard.edu/MIST/BC_tables/v2/{filter_set}.txz"

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tarball_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return tarball_path

    def extract_filter_set(self, filter_set: str) -> Path:
        extract_dir = self.cache_dir / filter_set

        if not self.rebuild and extract_dir.exists() and any(extract_dir.iterdir()):
            return extract_dir

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        tarball_path = self.download_filter_set(filter_set)

        with tarfile.open(tarball_path, mode="r:*") as tf:
            tf.extractall(path=extract_dir)

        return extract_dir

    def _iter_data_files(self, folder: Path):
        for p in folder.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                yield p

    def read_filter_set(self, filter_set: str) -> pd.DataFrame:
        extract_dir = self.extract_filter_set(filter_set)

        dfs: list[pd.DataFrame] = []

        for fp in sorted(self._iter_data_files(extract_dir)):
            with open(fp, "r") as f:
                for line in f:
                    stripped = line.strip().lstrip("#").strip()
                    if stripped.startswith("lgTef"):
                        header_line = stripped
                        break
            names = [s.replace("[Fe/H]", "feh").replace("Fe_H", "feh")
                     for s in header_line.split()]
            df = pd.read_csv(
                fp,
                sep="\\s+",
                comment="#",
                header=None,
                engine="python",
                names=names,
            )

            df["Teff"] = 10**df["lgTef"]
            df.drop(columns="lgTef", inplace=True)

            dfs.append(df)

        if not dfs:     # pragma: no cover
            raise FileNotFoundError(f"no BC files found in {extract_dir}")

        df = pd.concat(dfs, copy=False)

        if "a_Fe" in df.columns:
            df = df[df["a_Fe"] == 0.0].drop(columns="a_Fe")

        df.set_index(["Teff", "logg", "feh", "Av"], inplace=True)
        return df

    def build_hdf5(self, filter_set: str) -> Path:
        h5_path = self.cache_dir / f"{filter_set}.h5"

        if h5_path.exists() and not self.rebuild:
            return h5_path

        df = self.read_filter_set(filter_set)
        df.to_hdf(
            h5_path,
            key="bc",
            mode="w",
        )

        return h5_path

    def load_hdf5(self, filter_set: str) -> pd.DataFrame:
        h5_path = self.cache_dir / f"{filter_set}.h5"
        if not h5_path.exists() or self.rebuild:
            self.build_hdf5(filter_set)
        return pd.read_hdf(h5_path, key="bc")

    def _build_interpolators(self) -> None:
        df = self.bc_grid.sort_index()

        teff = np.asarray(df.index.get_level_values("Teff").unique(), dtype=float)
        logg = np.asarray(df.index.get_level_values("logg").unique(), dtype=float)
        feh = np.asarray(df.index.get_level_values("feh").unique(), dtype=float)
        av = np.asarray(df.index.get_level_values("Av").unique(), dtype=float)

        teff.sort()
        logg.sort()
        feh.sort()
        av.sort()

        full_index = pd.MultiIndex.from_product(
            [teff, logg, feh, av],
            names=["Teff", "logg", "feh", "Av"],
        )
        dense = df.reindex(full_index)

        self._grid_axes = (teff, logg, feh, av)
        self._interpolators: dict[str, RegularGridInterpolator] = {}

        n_teff, n_logg, n_feh, n_av = len(teff), len(logg), len(feh), len(av)

        for band in self.bands:
            values_1d = dense[band].to_numpy(dtype=float, copy=False)
            values_4d = values_1d.reshape(n_teff, n_logg, n_feh, n_av)

            self._interpolators[band] = RegularGridInterpolator(
                self._grid_axes,
                values_4d,
                method="linear",
            )

    def interp(
        self,
        teff: float | np.ndarray,
        logg: float | np.ndarray,
        feh: float | np.ndarray,
        av: float | np.ndarray,
        bands: tuple[str, ...] | None = None,
        silence_bounds_warning: bool = False,
    ) -> pd.Series | pd.DataFrame:
        use_bands = self.bands if bands is None else bands

        teff_a = np.asarray(teff, dtype=float)
        logg_a = np.asarray(logg, dtype=float)
        feh_a = np.asarray(feh, dtype=float)
        av_a = np.asarray(av, dtype=float)

        teff_b, logg_b, feh_b, av_b = np.broadcast_arrays(teff_a, logg_a, feh_a, av_a)
        n = teff_b.size

        teff_min, teff_max = self._grid_axes[0][0], self._grid_axes[0][-1]
        logg_min, logg_max = self._grid_axes[1][0], self._grid_axes[1][-1]
        feh_min, feh_max = self._grid_axes[2][0], self._grid_axes[2][-1]
        av_min, av_max = self._grid_axes[3][0], self._grid_axes[3][-1]

        if not silence_bounds_warning:
            for var, label, min_val, max_val in [
                (teff_b, "Teff", teff_min, teff_max),
                (logg_b, "logg", logg_min, logg_max),
                (feh_b, "feh", feh_min, feh_max),
                (av_b, "Av", av_min, av_max),
            ]:
                n_out_of_bounds = ((var < min_val) | (var > max_val)).sum()
                if n_out_of_bounds > 0:
                    logging.getLogger("cogsworth").warning(
                        f"cogsworth warning: {n_out_of_bounds} out of bounds points for {label} when "
                        f"interpolating MIST BCs (valid range: {min_val} to {max_val}). Clipping to bounds."
                    )

        teff_b = np.clip(teff_b, teff_min, teff_max)
        logg_b = np.clip(logg_b, logg_min, logg_max)
        feh_b = np.clip(feh_b, feh_min, feh_max)
        av_b = np.clip(av_b, av_min, av_max)

        pts = np.column_stack([
            teff_b.reshape(n),
            logg_b.reshape(n),
            feh_b.reshape(n),
            av_b.reshape(n),
        ])

        out = {b: self._interpolators[b](pts) for b in use_bands}

        if teff_b.shape == () and logg_b.shape == () and feh_b.shape == () and av_b.shape == ():
            return pd.Series({b: float(out[b][0]) for b in use_bands})

        return pd.DataFrame(out)


def build_photo_grid(bands):
    """
    Build a MIST bolometric correction grid for an arbitrary pair of bands.

    Parameters
    ----------
    bands : tuple[str]
        A tuple of photometric bands to build the grid for, e.g.
        ("Bessell_B", "Bessell_V", "Bessel_I") or ("2MASS_J", "2MASS_Ks"). Must both
        belong to the same MIST filter set (see `MIST_FILTER_SETS`).

    Returns
    -------
    bc_grid : MISTBolometricCorrectionGrid
        Grid set up for the given `bands`, ready to pass into
        `add_bands_photometry`.
    """
    return MISTBolometricCorrectionGrid(bands=bands)

def add_bands_photometry(df,
                         bc_grid,
                         metallicity=0.0142,
                         av=0.0,
                         teff_col='teff_1',
                         mass_col='mass_1',
                         rad_col='rad_1',
                         lum_col='lum_1',
                         suffix='_1',
                         distance=None,
                         silence_bounds_warning=False,
                         bands=None,
                         colors=None):
    """
    Add photometry in one or more bands (and any requested colors) to a
    DataFrame of stars.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `teff_col`, `mass_col`, `rad_col`, `lum_col`.
    bc_grid : MISTBolometricCorrectionGrid
        Bolometric correction grid, built with `build_photo_grid(bands)`.
        Required so the grid is only built once when calling this for
        multiple stars.
    metallicity : float
        Metallicity Z (mass fraction) applied to every star. Default (0.0142)
        is solar.
    av : float
        Visual extinction applied to every star. Default 0 (no extinction).
    teff_col : str
        Name of the effective temperature column in `df`. Defaults to
        'teff_1', COSMIC's default column name for the primary. Units of K.
    mass_col : str
        Name of the mass column in `df`. Defaults to 'mass_1', COSMIC's
        default column name for the primary. Units of Msun.
    rad_col : str
        Name of the radius column in `df`. Defaults to 'rad_1', COSMIC's
        default column name for the primary. Units of Rsun.
    lum_col : str
        Name of the luminosity column in `df`. Defaults to 'lum_1', COSMIC's
        default column name for the primary. Units of Lsun.
    suffix : str
        Suffix appended to every output column name (log_g, M_{band}, etc.),
        so results can be told apart when this is called more than once for
        different stars/objects. Defaults to '_1', matching COSMIC's
        convention for the primary; pass '_2' (or your own scheme) for the
        secondary, set suffix to the empty str if you plan to use once, etc.
    distance : float, optional
        A single distance in kiloparsecs applied to every star. If given,
        apparent mags are also added. If None, only absolute mags are
        computed.
    silence_bounds_warning : bool
        Whether to silence the out-of-bounds warning from the MIST BC grid.
    bands : tuple[str], optional
        The tuple of bands to compute photometry for, e.g. ("Bessell_B", "Bessell_V").
        Must match (or be a subset of) the bands `bc_grid` was built with.
        Defaults to `bc_grid.bands` if not given.
    colors : list[tuple[str, str]], optional
        Pairs of bands to difference into a color, e.g. [("Bessell_B", "Bessell_V")]
        computes B-V. Each band in a pair must be present in `bands`. If None,
        no colors are computed.

    Returns
    -------
    df : pandas.DataFrame
        Same DataFrame with added columns: log_g{suffix}, M_{band}{suffix} for
        each band (and m_{band}{suffix} if distance is given), plus
        {band0}-{band1}{suffix} for each requested color.

    Raises
    ------
    ValueError
        If any requested band is not available in `bc_grid`, or if a
        `colors` pair references a band not present in `bands`.

    Examples
    --------
    >>> bc_grid = build_photo_grid(bands=("Bessell_B", "Bessell_V", "Bessel_I"))
    >>> bcm = add_bands_photometry(bcm, bc_grid, bands=("Bessell_B", "Bessell_V", "Bessel_I"),
    ...                            colors=[("Bessell_B", "Bessell_V")],
    ...                            teff_col='teff_1', mass_col='mass_1', rad_col='rad_1',
    ...                            lum_col='lum_1', suffix='_1')
    >>> bcm = add_bands_photometry(bcm, bc_grid, bands=("Bessell_B", "Bessell_V", "Bessel_I"),
    ...                            colors=[("Bessell_B", "Bessell_V")],
    ...                            teff_col='teff_2', mass_col='mass_2', rad_col='rad_2',
    ...                            lum_col='lum_2', suffix='_2')
    """
    if bands is None:
        bands = bc_grid.bands

    missing = [b for b in bands if b not in bc_grid.bands]
    if missing:
        raise ValueError(
            f"band(s) {missing} not available in bc_grid (built with bands={bc_grid.bands}); "
            "rebuild bc_grid with build_photo_grid(bands=...) including these bands"
        )

    df = df.copy()

    logg_col = f"log_g{suffix}"

    if logg_col not in df:
        df[logg_col] = get_log_g(mass=df[mass_col].values * u.Msun,
                                  radius=df[rad_col].values * u.Rsun)

    feh_value = np.log10(metallicity / 0.0142)  # Bertelli+1994-style, Z_sun = 0.0142
    feh = np.full(len(df), feh_value)

    av_arr = np.full(len(df), av)

    bc = bc_grid.interp(teff=df[teff_col].values, logg=df[logg_col].values,
                         feh=feh, av=av_arr, bands=bands,
                         silence_bounds_warning=silence_bounds_warning)

    M_bol = get_absolute_bol_mag(lum=df[lum_col].values * u.Lsun)

    dist = np.full(len(df), distance) * u.kpc if distance is not None else None

    for b in bands:
        df[f"M_{b}{suffix}"] = M_bol - bc[b].values
        if dist is not None:
            df[f"m_{b}{suffix}"] = get_apparent_mag(df[f"M_{b}{suffix}"].values, dist)

    if colors is not None:
        for b0, b1 in colors:
            if b0 not in bands or b1 not in bands:
                raise ValueError(
                    f"color pair ({b0}, {b1}) requires both bands to be in `bands`={bands}"
                )
            df[f"{b0}-{b1}{suffix}"] = df[f"M_{b0}{suffix}"] - df[f"M_{b1}{suffix}"]

    return df