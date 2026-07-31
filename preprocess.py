"""Preprocessing fixes for MC and data DataFrames.

Each fix is idempotent: calling a fix that has already been applied on the
DataFrame is a no-op (a warning is printed but no error is raised).

Whether a fix has been applied is stored as a boolean column::

    ('_fix_<name>', '', '', ..., '')          # MultiIndex depth matched
    '_fix_<name>'                             # flat Index fallback

The flag column is cheap (one bool per row), survives ``pd.concat``, and
is visible when inspecting the DataFrame.  Use :func:`is_fix_applied` to
check programmatically and :func:`applied_fixes` to list all recorded fixes.

MC-only fixes
-------------
- :func:`fix_flash_pe_scale`  — scale flash PEs by a calibration factor

Data-only fixes
---------------
- :func:`fix_flash_time`      — correct flash time via per-event frame offset

MC + data fixes
---------------
- :func:`fix_sec_shw_energy` — scale secondary shower energy from collection-plane (I2) energy
- :func:`add_phi`            — derive shower and track azimuthal angles from direction

Pi0 fix (opt-in, call after preprocess_mc / preprocess_data)
-------------------------------------------------------------
- :func:`add_pi0`  — pi0 kinematics: opening angle, invariant mass, momentum
"""

import warnings

import numpy as np
import pandas as pd

__all__ = [
    'is_fix_applied',
    'applied_fixes',
    'preprocess_mc',
    'preprocess_data',
    'fix_flash_pe_scale',
    'fix_flash_time',
    'add_phi',
    'fix_prim_shw_energy',
    'fix_sec_shw_energy',
    'add_pi0',
]

_FIX_PREFIX = '_fix_'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag_col(df: pd.DataFrame, name: str):
    """Column key for fix *name*, padded to df's MultiIndex depth."""
    key = f'{_FIX_PREFIX}{name}'
    if not isinstance(df.columns, pd.MultiIndex):
        return key
    depth = df.columns.nlevels
    return tuple([key] + [''] * (depth - 1))


def is_fix_applied(df: pd.DataFrame, name: str) -> bool:
    """Return True if fix *name* has been recorded on *df*."""
    return _flag_col(df, name) in df.columns


def applied_fixes(df: pd.DataFrame) -> list[str]:
    """Return a list of fix names that have been recorded on *df*."""
    if isinstance(df.columns, pd.MultiIndex):
        top = [c[0] for c in df.columns]
    else:
        top = list(df.columns)
    return [c[len(_FIX_PREFIX):] for c in top if c.startswith(_FIX_PREFIX)]


def _mark_applied(df: pd.DataFrame, name: str) -> pd.DataFrame:
    flag_series = pd.Series(True, index=df.index, name=_flag_col(df, name))
    return pd.concat([df, flag_series], axis=1)


def _skip_if_applied(df: pd.DataFrame, name: str) -> bool:
    """Warn and return True if fix already applied; else return False."""
    if is_fix_applied(df, name):
        warnings.warn(
            f"Fix '{name}' is already applied to this DataFrame; skipping.",
            stacklevel=3,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# MC-only fixes
# ---------------------------------------------------------------------------

def fix_flash_pe_scale(df: pd.DataFrame, scale: float = 0.66) -> pd.DataFrame:
    """Scale flash PEs by *scale* (MC only).

    Corrects for the flash PE calibration difference between MC and data.
    Default scale factor is 0.66.

    Parameters
    ----------
    df : pd.DataFrame
        MC DataFrame containing ``slc.barycenterFM.flashPEs``.
    scale : float
        Multiplicative factor applied to flash PE values.
    """
    name = 'flash_pe_scale'
    if _skip_if_applied(df, name):
        return df
    col = ('slc', 'barycenterFM', 'flashPEs', '', '', '')
    df[col] = df[col] * scale
    return _mark_applied(df, name)


# ---------------------------------------------------------------------------
# Data-only fixes
# ---------------------------------------------------------------------------

def fix_flash_time(df: pd.DataFrame, offset: float = 0.19) -> pd.DataFrame:
    """Apply flash time correction using the per-event frame offset (data only).

    Corrects the recorded flash time as::

        flashTime_corrected = flashTime + frameApplyAtCaf / 1e3 - offset

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame containing ``slc.barycenterFM.flashTime`` and
        ``frameApplyAtCaf``.
    offset : float
        Data–MC timing offset in µs (default 0.19 µs).
    """
    name = 'flash_time'
    if _skip_if_applied(df, name):
        return df
    col = ('slc', 'barycenterFM', 'flashTime', '', '', '')
    df[col] = df.slc.barycenterFM.flashTime + df.frameApplyAtCaf / 1e3 - offset
    return _mark_applied(df, name)


# ---------------------------------------------------------------------------
# Bundled entry points
# ---------------------------------------------------------------------------

def preprocess_mc(df: pd.DataFrame, *, flash_pe_scale: float = 0.66) -> pd.DataFrame:
    """Apply all standard MC preprocessing fixes in order.

    Applies:

    1. :func:`fix_flash_pe_scale`  — flash PE calibration correction
    2. :func:`fix_prim_shw_energy` — primary shower reco_energy from collection-plane (I2) energy
    3. :func:`fix_sec_shw_energy`  — secondary shower energy from collection-plane (I2) energy
    4. :func:`add_phi`             — shower and track azimuthal angles

    All fixes are idempotent; calling this on an already-preprocessed
    DataFrame is safe (each already-applied fix warns and skips).

    Parameters
    ----------
    df : pd.DataFrame
        MC DataFrame.
    flash_pe_scale : float
        Scale factor forwarded to :func:`fix_flash_pe_scale` (default 0.66).
    """
    df = fix_flash_pe_scale(df, scale=flash_pe_scale)
    df = fix_prim_shw_energy(df)
    df = fix_sec_shw_energy(df)
    df = add_phi(df)
    return df


def preprocess_data(df: pd.DataFrame, *, flash_time_offset: float = 0.19) -> pd.DataFrame:
    """Apply all standard data preprocessing fixes in order.

    Applies:

    1. :func:`fix_flash_time`      — flash time frame-offset correction
    2. :func:`fix_prim_shw_energy` — primary shower reco_energy from collection-plane (I2) energy
    3. :func:`fix_sec_shw_energy`  — secondary shower energy from collection-plane (I2) energy
    4. :func:`add_phi`             — shower and track azimuthal angles

    All fixes are idempotent; calling this on an already-preprocessed
    DataFrame is safe (each already-applied fix warns and skips).

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame (on-beam or off-beam).
    flash_time_offset : float
        Timing offset in µs forwarded to :func:`fix_flash_time` (default 0.19).
    """
    df = fix_flash_time(df, offset=flash_time_offset)
    df = fix_prim_shw_energy(df)
    df = fix_sec_shw_energy(df)
    df = add_phi(df)
    return df


def add_phi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute azimuthal angle φ (degrees) for the primary shower and track.

    Stores results in ``primshw.shw.dir.phi`` and ``primtrk.trk.dir.phi``.
    Track phi will be NaN for events with no reconstructed track.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``primshw.shw.dir.{x,y}`` and
        ``primtrk.trk.dir.{x,y}``.
    """
    name = 'phi'
    if _skip_if_applied(df, name):
        return df
    depth = df.columns.nlevels if isinstance(df.columns, pd.MultiIndex) else 1
    pad = [''] * max(0, depth - 4)
    shw_col = tuple(['primshw', 'shw', 'dir', 'phi'] + pad)
    trk_col = tuple(['primtrk', 'trk', 'dir', 'phi'] + pad)

    # Compute phi values
    shw_phi = np.arctan2(df.primshw.shw.dir.x, df.primshw.shw.dir.y) * 180 / np.pi
    trk_phi = np.arctan2(df.primtrk.trk.dir.x, df.primtrk.trk.dir.y) * 180 / np.pi

    # Batch add columns to avoid fragmentation
    new_cols = pd.DataFrame({shw_col: shw_phi, trk_col: trk_phi})
    df = pd.concat([df, new_cols], axis=1)
    return _mark_applied(df, name)


# ---------------------------------------------------------------------------
# Shower energy fixes
# ---------------------------------------------------------------------------

def fix_prim_shw_energy(
    df: pd.DataFrame,
    scale: float = 1.17,
) -> pd.DataFrame:
    """Set primary shower reco_energy from collection-plane (I2) energy * scale.

    Mirrors ``_add_reco_energy`` in ``make_nueccdf.py``.

    If ``primshw.shw.reco_energy`` already exists, checks that its values match
    the expected scaled energy and warns if not.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``primshw.shw.plane.I2.energy``.
    scale : float
        Global energy scale factor (default 1.17).
    """
    name = 'prim_shw_energy'
    if _skip_if_applied(df, name):
        return df
    col = ('primshw', 'shw', 'reco_energy', '', '', '')
    if col in df.columns:
        expected = df.primshw.shw.plane.I2.energy * scale
        ratio = (df[col] / expected).dropna()
        if not np.allclose(ratio, 1.0, rtol=0.01):
            warnings.warn(
                f"primshw.shw.reco_energy already exists but differs from "
                f"plane.I2.energy * {scale} "
                f"(mean ratio: {ratio.mean():.3f}). "
                "Overwriting.",
                stacklevel=2,
            )
    df[col] = df.primshw.shw.plane.I2.energy * scale
    return _mark_applied(df, name)


def fix_sec_shw_energy(
    df: pd.DataFrame,
    scale: float = 1.17,
) -> pd.DataFrame:
    """Set secondary shower reco_energy from collection-plane (I2) energy * scale.

    Matches the primary shower treatment in :func:`fix_prim_shw_energy`.
    Must be called before :func:`add_pi0` so the pi0 invariant mass uses the
    scaled energy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``secshw.shw.plane.I2.energy``.
    scale : float
        Global energy scale factor (default 1.17).
    """
    name = 'sec_shw_energy'
    if _skip_if_applied(df, name):
        return df
    col = ('secshw', 'shw', 'reco_energy', '', '', '')
    df[col] = df.secshw.shw.plane.I2.energy * scale
    return _mark_applied(df, name)


def add_pi0(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pi0 kinematics from the primary and secondary shower.

    Assumes ``primshw.shw.reco_energy`` and ``secshw.shw.reco_energy`` have
    already been set (call :func:`fix_sec_shw_energy` first if needed).

    Derived columns
    ---------------
    pi0.cos2angle             — cos of opening angle between the two showers
    pi0.openangle             — opening angle in degrees
    primshw.shw.p.{x,y,z}    — primary shower momentum vector
    secshw.shw.p.{x,y,z}     — secondary shower momentum vector
    pi0.mass                  — pi0 invariant mass [GeV]
    pi0.p.{x,y,z}             — pi0 momentum vector
    pi0.p.totp                — pi0 momentum magnitude
    pi0.dir.{x,y,z}           — pi0 unit direction

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing primary and secondary shower direction and energy
        columns.
    """
    name = 'pi0'
    if _skip_if_applied(df, name):
        return df

    def _valid(s, sentinel=-999):
        return s.where(s != sentinel)

    prim_E  = df.primshw.shw.reco_energy
    sec_E   = df.secshw.shw.reco_energy
    prim_dx = _valid(df.primshw.shw.dir.x)
    prim_dy = _valid(df.primshw.shw.dir.y)
    prim_dz = _valid(df.primshw.shw.dir.z)
    sec_dx  = _valid(df.secshw.shw.dir.x)
    sec_dy  = _valid(df.secshw.shw.dir.y)
    sec_dz  = _valid(df.secshw.shw.dir.z)

    cos2angle  = prim_dx*sec_dx + prim_dy*sec_dy + prim_dz*sec_dz
    open_angle = np.degrees(np.arccos(cos2angle.clip(-1, 1)))

    prim_px = prim_E * prim_dx
    prim_py = prim_E * prim_dy
    prim_pz = prim_E * prim_dz
    sec_px  = sec_E  * sec_dx
    sec_py  = sec_E  * sec_dy
    sec_pz  = sec_E  * sec_dz

    pi0_px  = prim_px + sec_px
    pi0_py  = prim_py + sec_py
    pi0_pz  = prim_pz + sec_pz
    pi0_mag = np.sqrt(pi0_px**2 + pi0_py**2 + pi0_pz**2)

    pi0_mag_safe = pi0_mag.where(pi0_mag > 0, other=np.nan)
    pi0_dx = pi0_px / pi0_mag_safe
    pi0_dy = pi0_py / pi0_mag_safe
    pi0_dz = pi0_pz / pi0_mag_safe

    pi0_mass = np.sqrt((2 * prim_E * sec_E * (1 - cos2angle)).clip(0))
    
    alpha = (prim_E - sec_E) / (prim_E + sec_E)
    _denom = (1 - alpha**2) * (1 - cos2angle)
    _denom_safe = np.where(_denom > 0, _denom, np.nan)
    _arg = (2 / _denom_safe) - 1
    pi0_mag_alt = 0.135 * np.sqrt(np.where(_arg >= 0, _arg, np.nan))

    new_cols = pd.DataFrame({
        ('pi0',    'cos2angle', '',    '', '', ''): cos2angle,
        ('pi0',    'openangle', '',    '', '', ''): open_angle,
        ('primshw', 'shw',     'p', 'x', '', ''): prim_px,
        ('primshw', 'shw',     'p', 'y', '', ''): prim_py,
        ('primshw', 'shw',     'p', 'z', '', ''): prim_pz,
        ('secshw',  'shw',     'p', 'x', '', ''): sec_px,
        ('secshw',  'shw',     'p', 'y', '', ''): sec_py,
        ('secshw',  'shw',     'p', 'z', '', ''): sec_pz,
        ('pi0',    'mass',      '',    '', '', ''): pi0_mass,
        ('pi0',    'p',        'x',    '', '', ''): pi0_px,
        ('pi0',    'p',        'y',    '', '', ''): pi0_py,
        ('pi0',    'p',        'z',    '', '', ''): pi0_pz,
        ('pi0',    'p',        'totp', '', '', ''): pi0_mag,
        ('pi0',    'p',        'totp_alt', '', '', ''): pi0_mag_alt,
        ('pi0',    'alpha',     '',    '', '', ''): alpha,
        ('pi0',    'dir',      'x',    '', '', ''): pi0_dx,
        ('pi0',    'dir',      'y',    '', '', ''): pi0_dy,
        ('pi0',    'dir',      'z',    '', '', ''): pi0_dz,
    }, index=df.index)

    df = pd.concat([df, new_cols], axis=1)
    return _mark_applied(df, name)
