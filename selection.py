import numpy as np
import pandas as pd
from . import config
from makedf.util import *
from pyanalib.pandas_helpers import *

from .utils import ensure_lexsorted
from .constants import signal_dict, generic_dict
from .geometry import whereTPC


# ---------------------------------------------------------------------------
# Individual cut functions — each returns a boolean mask over df rows.
# Call these directly when you need fine-grained control (e.g. in detvar loops).
# ---------------------------------------------------------------------------

def cut_preselection(df, pe_cut=2e3, nuscore_cut=0.5, realisticFV=True):
    """Preselection: flash PE, nu score, clear-cosmic veto, and fiducial volume."""
    mask = (
        (df.slc.barycenterFM.flashPEs > pe_cut)
        & (df.slc.nu_score > nuscore_cut)
        & (df.slc.is_clear_cosmic == 0)
    )
    if realisticFV:
        mask &= InFV(df.slc.vertex, det="SBND_nohighyz", inzback=0)
    return mask


def cut_flash_matching(df, spill_start=0.335, spill_end=0.335 + 1.6, score_cut=0.02):
    """Flash matching: beam spill timing and flash match score."""
    return InSpill(df, spill_start, spill_end) & InScore(df, score_cut)


def cut_shower_energy(df, min_shower_energy=0.5):
    """Shower energy: minimum reco energy and positive track score."""
    return (df.primshw.shw.reco_energy > min_shower_energy) & (df.primshw.trackScore > 0)


def cut_muon_rejection(df, max_track_length=200):
    """Muon rejection: no reconstructed track longer than max_track_length cm."""
    return np.isnan(df.primtrk.trk.len) | (df.primtrk.trk.len < max_track_length)


def cut_conversion_gap(df, min_conversion_gap=0.001, max_conversion_gap=2):
    """Conversion gap: primary shower vertex-to-start distance in cm."""
    return (
        (df.primshw.shw.conversion_gap > min_conversion_gap)
        & (df.primshw.shw.conversion_gap < max_conversion_gap)
    )


def cut_dedx(df, min_dedx=1.25, max_dedx=2.5):
    """dE/dx: best-plane dE/dx of the primary shower."""
    return (
        (df.primshw.shw.bestplane_dEdx > min_dedx)
        & (df.primshw.shw.bestplane_dEdx < max_dedx)
    )


def cut_opening_angle(df, min_opening_angle=0.03, max_opening_angle=0.15):
    """Opening angle: primary shower opening angle in radians."""
    return (
        (df.primshw.shw.open_angle > min_opening_angle)
        & (df.primshw.shw.open_angle < max_opening_angle)
    )


def cut_shower_length(df, min_shower_length=0.1, max_shower_length=200):
    """Shower length: primary shower length in cm."""
    return (
        (df.primshw.shw.len > min_shower_length)
        & (df.primshw.shw.len < max_shower_length)
    )


def cut_direction(df, min_direction=-1, max_direction=1):
    """Direction: z-component of the primary shower direction (cos theta)."""
    return (
        (df.primshw.shw.dir.z > min_direction)
        & (df.primshw.shw.dir.z < max_direction)
    )


def InSpill(df, spill_start=0.335, spill_end=0.335 + 1.6):
    return (df.slc.barycenterFM.flashTime > spill_start) & (df.slc.barycenterFM.flashTime < spill_end)


def InScore(df, score_cut=0.02):
    return (df.slc.barycenterFM.score > score_cut)


# ---------------------------------------------------------------------------
# Full selection pipeline
# ---------------------------------------------------------------------------

def select(indf,
           stage=None,
           savedict=False,
           spring=True,
           realisticFV=True,
           spill_start=0.335,
           spill_end=0.335 + 1.6,
           score_cut=0.02,
           nuscore_cut=0.5,
           pe_cut=2e3,
           shower_scale=1.17,
           min_shower_energy=0.5,
           max_track_length=200,
           min_conversion_gap=0.001,
           max_conversion_gap=2,
           min_dedx=1.25,
           max_dedx=2.5,
           min_opening_angle=0.03,
           max_opening_angle=0.15,
           min_shower_length=0.1,
           max_shower_length=200,
           min_direction=-1,
           max_direction=1,
           extra_cuts=None,
           skip_cuts=None):
    """Apply the full nue CC selection to a DataFrame.

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame.
    stage : str, optional
        Stop and return after this named stage. One of: ``'preselection'``,
        ``'flash matching'``, ``'shower energy'``, ``'muon rejection'``,
        ``'conversion gap'``, ``'dEdx'``, ``'opening angle'``,
        ``'shower length'``. If None, all cuts are applied.
    savedict : bool, default False
        If True, return a dict of DataFrames keyed by stage name instead of
        the final DataFrame.
    spring : bool, default True
        Use max-plane shower energy for reco energy (True) or best-plane (False).
    realisticFV : bool, default True
        Apply the realistic active-volume fiducial cut.
    spill_start, spill_end : float
        Beam-spill flash-time window in µs.
    score_cut : float, default 0.02
        Minimum flash-match score.
    nuscore_cut : float, default 0.5
        Minimum neutrino score.
    pe_cut : float, default 2000
        Minimum flash photoelectrons.
    shower_scale : float, default 1.17
        Reco-to-true shower energy scale factor.
    min_shower_energy : float, default 0.5
        Minimum primary shower energy in GeV.
    max_track_length : float, default 200
        Maximum primary track length in cm.
    min_conversion_gap, max_conversion_gap : float
        Shower conversion gap range in cm.
    min_dedx, max_dedx : float
        Best-plane dE/dx range in MeV/cm.
    min_opening_angle, max_opening_angle : float
        Shower opening angle range in radians.
    min_shower_length, max_shower_length : float
        Shower length range in cm.
    min_direction, max_direction : float
        Shower direction (cos theta) range.
    extra_cuts : callable or list of callable, optional
        Additional cut function(s) applied after all named stages. Each must
        accept a DataFrame and return a boolean mask, e.g.
        ``lambda df: abs(df.x) > 10``.
    skip_cuts : list of str, optional
        Named stages to skip entirely. Valid names match the ``stage``
        parameter: ``'preselection'``, ``'flash matching'``, ``'shower energy'``,
        ``'muon rejection'``, ``'conversion gap'``, ``'dEdx'``,
        ``'opening angle'``, ``'shower length'``.

    Returns
    -------
    pandas.DataFrame or dict
        Final selected DataFrame, or a dict of per-stage DataFrames when
        ``savedict=True`` or ``stage`` is set.
    """
    valid_stages = [
        'preselection', 'flash matching', 'shower energy', 'muon rejection',
        'conversion gap', 'dEdx', 'opening angle', 'shower length',
    ]
    if stage is not None and stage not in valid_stages:
        raise ValueError(f"Unknown stage '{stage}'. Valid options: {valid_stages}")

    skip = set(skip_cuts or [])
    invalid_skips = skip - set(valid_stages)
    if invalid_skips:
        raise ValueError(f"skip_cuts contains unknown stages: {sorted(invalid_skips)}. Valid options: {valid_stages}")

    _extra = extra_cuts if isinstance(extra_cuts, list) else [extra_cuts] if extra_cuts is not None else []

    df_dict = {}
    df = indf.copy()

    # Reco energy definition (preprocessing, not a cut).
    energy_col = ("primshw", "shw", "reco_energy", '', '', '')
    if spring:
        df[energy_col] = df.primshw.shw.maxplane_energy * shower_scale
    else:
        df[energy_col] = df.primshw.shw.bestplane_energy * shower_scale

    def save_stage(stage_name, current_df):
        if savedict:
            df_dict[stage_name] = current_df
        if stage == stage_name:
            return df_dict if savedict else current_df
        return None

    if 'preselection' not in skip:
        df = df[cut_preselection(df, pe_cut=pe_cut, nuscore_cut=nuscore_cut, realisticFV=realisticFV)]
    result = save_stage('preselection', df)
    if result is not None: return result

    if 'flash matching' not in skip:
        df = df[cut_flash_matching(df, spill_start=spill_start, spill_end=spill_end, score_cut=score_cut)]
    result = save_stage('flash matching', df)
    if result is not None: return result

    if 'shower energy' not in skip:
        df = df[cut_shower_energy(df, min_shower_energy=min_shower_energy)]
    result = save_stage('shower energy', df)
    if result is not None: return result

    if 'muon rejection' not in skip:
        df = df[cut_muon_rejection(df, max_track_length=max_track_length)]
    result = save_stage('muon rejection', df)
    if result is not None: return result

    if 'conversion gap' not in skip:
        df = df[cut_conversion_gap(df, min_conversion_gap=min_conversion_gap, max_conversion_gap=max_conversion_gap)]
    result = save_stage('conversion gap', df)
    if result is not None: return result

    if 'dEdx' not in skip:
        df = df[cut_dedx(df, min_dedx=min_dedx, max_dedx=max_dedx)]
    result = save_stage('dEdx', df)
    if result is not None: return result

    if 'opening angle' not in skip:
        df = df[cut_opening_angle(df, min_opening_angle=min_opening_angle, max_opening_angle=max_opening_angle)]
    result = save_stage('opening angle', df)
    if result is not None: return result

    if 'shower length' not in skip:
        df = df[cut_shower_length(df, min_shower_length=min_shower_length, max_shower_length=max_shower_length)]
    result = save_stage('shower length', df)
    if result is not None: return result

    df = df[cut_direction(df, min_direction=min_direction, max_direction=max_direction)]

    for cut_fn in _extra:
        df = df[cut_fn(df)]

    return df_dict if savedict else df


def select_sideband(indf,
                    savedict=False,
                    min_conversion_gap=2,
                    max_conversion_gap=1e3,
                    max_track_length=1e3,
                    min_dedx=3,
                    max_dedx=6,
                    min_opening_angle=0.15,
                    max_opening_angle=1.0,
                    **kwargs):
    """Apply sideband selection cuts with modified default parameters.

    Calls :func:`select` with sideband-specific defaults. All parameters can
    be overridden via ``kwargs``.
    """
    return select(indf,
                  savedict=savedict,
                  max_track_length=max_track_length,
                  min_conversion_gap=min_conversion_gap,
                  max_conversion_gap=max_conversion_gap,
                  min_dedx=min_dedx,
                  max_dedx=max_dedx,
                  min_opening_angle=min_opening_angle,
                  max_opening_angle=max_opening_angle,
                  **kwargs)


def define_signal(indf: pd.DataFrame, prefix=None):
    """Define signal/background categories for neutrino interactions.

    Categorizes events into signal (CC nue) and background categories
    based on truth information and fiducial volume.

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with MultiIndex columns containing truth information.
    prefix : str or tuple, optional
        Column prefix to access truth information. If None, uses top-level columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added ``signal`` column (values from ``signal_dict``).
    """
    nudf = ensure_lexsorted(ensure_lexsorted(indf, 0), 1)

    if prefix is None:
        mcdf = nudf
    else:
        mcdf = nudf[prefix]

    whereFV = InFV(mcdf.position, det="SBND_nohighyz", inzback=0)
    whereAV = InAV(df=mcdf.position)
    whereCCnue = (
        (mcdf.iscc == 1)
        & (abs(mcdf.pdg) == 12)
        & (abs(mcdf.e.pdg) == 11)
        & (mcdf.e.genE > 0.5)
    )

    if "signal" in nudf.columns:
        signal = nudf["signal"].to_numpy(copy=True)
    else:
        signal = np.full(len(nudf), -1, dtype=np.int16)

    signal[whereFV & (mcdf.iscc == 1) & (abs(mcdf.pdg) == 14) & (mcdf.npi0 > 0)]  = signal_dict["numuCCpi0"]
    signal[whereFV & (mcdf.iscc == 0) & (mcdf.npi0 > 0)]                           = signal_dict["NCpi0"]
    signal[whereFV & (mcdf.iscc == 1) & (abs(mcdf.pdg) == 12)]                     = signal_dict["othernueCC"]
    signal[whereFV & (mcdf.iscc == 1) & (abs(mcdf.pdg) == 14) & (mcdf.npi0 == 0)] = signal_dict["othernumuCC"]
    signal[whereFV & (mcdf.iscc == 0) & (mcdf.npi0 == 0)]                          = signal_dict["otherNC"]
    signal[whereAV & (signal < 0)]                                                  = signal_dict["nonFV"]
    signal[whereAV == False]                                                        = signal_dict["dirt"]
    signal[np.isnan(mcdf.E)]                                                        = signal_dict['cosmic']
    signal[whereFV & whereCCnue]                                                    = signal_dict["nueCC"]

    nudf["signal"] = signal
    if ((nudf.signal < 0) | (nudf.signal >= len(signal_dict))).any():
        print("Warning: unidentified signal/background channels present.")
    return nudf


def define_generic(indf: pd.DataFrame, prefix=None):
    """Define broad signal/background categories (CC nu, NC nu, non-FV, dirt, cosmic).

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with MultiIndex columns containing truth information.
    prefix : str or tuple, optional
        Column prefix to access truth information. If None, uses top-level columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added ``signal`` column (values from ``generic_dict``).
    """
    indf = ensure_lexsorted(indf, 0)
    nudf = ensure_lexsorted(indf.copy(), 1)

    if prefix is None:
        mcdf = nudf
    else:
        mcdf = nudf[prefix]

    whereFV = InFV(df=mcdf.position, inzback=0, det="SBND")
    whereAV = InAV(df=mcdf.position)

    if "signal" not in nudf.columns:
        nudf["signal"] = -1

    nudf["signal"] = np.where(whereAV == False,              generic_dict["dirt"],  nudf["signal"])
    nudf["signal"] = np.where(whereAV,                       generic_dict["nonFV"], nudf["signal"])
    nudf["signal"] = np.where(whereFV & (mcdf.iscc == 0),    generic_dict["NCnu"],  nudf["signal"])
    nudf["signal"] = np.where(whereFV & (mcdf.iscc == 1),    generic_dict["CCnu"],  nudf["signal"])
    nudf["signal"] = np.where(np.isnan(mcdf.E),              generic_dict["cosmic"],nudf["signal"])

    if ((nudf.signal < 0) | (nudf.signal >= len(generic_dict))).any():
        print("Warning: unidentified signal/background channels present.")
    indf["signal"] = nudf["signal"]
    return indf
