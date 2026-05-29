"""Fake-data test (FDT) helpers: response matrix construction and fake-data histogram building.

These utilities handle the cafpyana-free layer of an FDT workflow. The WienerSVD
unfolding call and any generator-level plotting remain in the notebook, since they
cross the cafpyana boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections.abc import Callable

from .utils import get_hist1d, get_hist2d, ensure_lexsorted
from .classes import VariableConfig, SystematicsOutput
from .syst import _classify_category

__all__ = [
    'UnfoldInput',
    'get_response_matrix',
    'make_fake_data_hists',
]


# ---------------------------------------------------------------------------


@dataclass
class UnfoldInput:
    """Pre-built inputs to a WienerSVD unfolding call.

    Stores the fixed components (response matrix, CV signal prediction, and
    per-key systematic covariance matrices) that are expensive to compute and
    do not change between fake-data test iterations. The measurement and
    covariance key selection are deferred to ``unfold()``, which is called
    once per FDT iteration.

    Parameters
    ----------
    response : np.ndarray, shape (n_reco, n_true)
        Response matrix from get_response_matrix.
    cv_signal : np.ndarray, shape (n_true,)
        Central-value signal prediction scaled by rate_scale.
    syst_covs : dict of str -> np.ndarray
        All systematic covariance matrices from syst_output.xsec_syst_dict,
        keyed by their original string identifiers. Not yet scaled by
        flux_scale — that is applied inside unfold().
    rate_scale : float
        Factor already baked into cv_signal. Default 1.0.
    flux_scale : float
        Factor applied to the summed covariance inside unfold(). Default 1.0.
    """
    response:   np.ndarray
    cv_signal:  np.ndarray
    syst_covs:  dict
    rate_scale: float = 1.0
    flux_scale: float = 1.0

    def unfold(
        self,
        wienersvd_fn: Callable,
        measure: np.ndarray,
        allowed_keys: tuple[str, ...] | None = None,
        extra_cov: np.ndarray | None = None,
        meas_scale: float | None = None,
        c_type: int = 2,
        norm_type: float = 0.5,
    ) -> dict:
        """Run WienerSVD unfolding.

        Parameters
        ----------
        wienersvd_fn : callable
            The WienerSVD function imported from cafpyana in the notebook.
        measure : np.ndarray, shape (n_reco,)
            Background-subtracted measurement in event-rate units. Scaled by
            meas_scale before being passed to WienerSVD.
        allowed_keys : tuple of str or None, optional
            Categories to include, matched via the same classification logic as
            syst.py (e.g. ``('GENIE', 'MCstat')``). Each key in syst_covs is
            classified into a category ('GENIE', 'Flux', 'MCstat', 'DetVar',
            'Geant4') and included only if its category appears in allowed_keys.
            GENIE aliases (SBNNuSyst, SuSAv2) are handled correctly. None
            (default) includes all keys.
        extra_cov : np.ndarray or None, optional
            Additional covariance matrix added after summing syst_covs and
            applying flux_scale. Use for fake-data statistical uncertainty.
        meas_scale : float or None, optional
            Multiplicative scale applied to measure before passing to WienerSVD.
            Use when the measurement (e.g. real data or an alternate fake-data
            sample) requires a different unit conversion than cv_signal.
            Defaults to rate_scale.
        c_type : int, optional
            WienerSVD smoothness matrix type (0=unit, 1=1st deriv, 2=2nd deriv,
            3=3rd deriv). Default 2.
        norm_type : float, optional
            WienerSVD signal normalisation exponent. Default 0.5.

        Returns
        -------
        dict
            WienerSVD output dict with keys 'unfold', 'AddSmear', 'WF',
            'UnfoldCov', 'CovRotation'.
        """
        if meas_scale is None:
            meas_scale = self.rate_scale
        n_bins = self.cv_signal.shape[0]
        cov = np.zeros((n_bins, n_bins))
        for k, c in self.syst_covs.items():
            if allowed_keys is None or _classify_category(k) in allowed_keys:
                cov += c
        cov = cov * self.flux_scale
        if extra_cov is not None:
            cov = cov + extra_cov
        return wienersvd_fn(
            Response=self.response,
            Signal=self.cv_signal,
            Measure=measure * meas_scale,
            Covariance=cov,
            C_type=c_type,
            Norm_type=norm_type,
        )

    @classmethod
    def build(
        cls,
        var: VariableConfig,
        reco_df: pd.DataFrame,
        true_df: pd.DataFrame,
        syst_output: SystematicsOutput,
        rate_scale: float = 1.0,
        flux_scale: float = 1.0,
    ) -> UnfoldInput:
        """Construct an UnfoldInput from a VariableConfig and SystematicsOutput.

        Builds the expensive fixed components once (response matrix, CV signal,
        full per-key covariance dict). Key selection and measurement are passed
        to unfold() at call time.

        Parameters
        ----------
        var : VariableConfig
            Specifies which variable to unfold. Supplies column keys and bins
            for the response matrix and cv_signal histogram.
        reco_df : pd.DataFrame
            Full selected sample with weights_mc and signal columns.
            Signal events (signal == 0) are used for the response matrix.
        true_df : pd.DataFrame
            Truth-level signal-only sample with weights_mc column.
        syst_output : SystematicsOutput
            Output of get_total_cov for this variable. All entries in
            xsec_syst_dict are stored; key filtering happens in unfold().
        rate_scale : float, optional
            Multiplied into cv_signal. Use to convert from event-rate to
            cross-section units (e.g. integrated_flux * (mcbnb_pot / 1e6)).
            Default 1.0.
        flux_scale : float, optional
            Multiplied into the summed covariance inside unfold(). Use to
            convert from flux-averaged event-rate units to event-rate squared
            (e.g. (integrated_flux * (mcbnb_pot / 1e6))**2). Default 1.0.

        Returns
        -------
        UnfoldInput
        """
        true_df = ensure_lexsorted(true_df, axis=1)

        response = get_response_matrix(reco_df, true_df, var)

        cv_signal = get_hist1d(
            data=true_df[var.var_nu_col],
            bins=var.bins,
            weights=true_df.weights_mc,
        ) * rate_scale

        syst_covs = {
            k: entry['cov']
            for k, entry in syst_output.xsec_syst_dict.items()
        }

        return cls(
            response=response,
            cv_signal=cv_signal,
            syst_covs=syst_covs,
            rate_scale=rate_scale,
            flux_scale=flux_scale,
        )

def get_response_matrix(
    reco_df: pd.DataFrame,
    true_df: pd.DataFrame,
    var: VariableConfig,
) -> np.ndarray:
    """Build a response (smearing) matrix normalized column-wise by the truth histogram.

    Parameters
    ----------
    reco_df : pd.DataFrame
        Selected signal events with a weights_mc column. The reco and truth
        columns are read from var.var_evt_reco_col and var.var_evt_truth_col.
    true_df : pd.DataFrame
        Truth-level signal-only sample (e.g. mcsig_df). The truth column is
        read from var.var_nu_col. weights_mc is used if present, otherwise
        uniform weights of 1.0.
    var : VariableConfig
        Variable configuration supplying column keys and bin edges.
        Only variables with a VariableConfig (i.e. cross-section extraction
        variables) should be passed here.

    Returns
    -------
    np.ndarray, shape (n_bins, n_bins)
        R[i, j] = fraction of true-bin-j events reconstructed in reco-bin-i.
    """
    # filter to signal events first — signal is a flat top-level column so
    # no lexsort is needed to access it, and sorting the smaller result is cheaper
    reco_df  = reco_df[reco_df.signal == 0]
    reco_df  = ensure_lexsorted(reco_df, axis=1)
    true_df  = ensure_lexsorted(true_df, axis=1)

    truth_hist = get_hist1d(
        data=true_df[var.var_nu_col],
        bins=var.bins,
        weights=true_df.weights_mc,
    )
    smearing = get_hist2d(
        weights=reco_df.weights_mc,
        x=reco_df[var.var_evt_reco_col],
        y=reco_df[var.var_evt_truth_col],
        bins=var.bins,
    )
    response = np.divide(
        smearing,
        truth_hist,
        out=np.zeros_like(smearing),
        where=truth_hist != 0,
    )
    return response


# ---------------------------------------------------------------------------


def make_fake_data_hists(
    reco_df: pd.DataFrame,
    true_df: pd.DataFrame,
    var: VariableConfig,
    reco_mask: np.ndarray,
    true_mask: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build background-subtracted fake-data and modified signal histograms for an FDT.

    Applies a multiplicative weight to masked events in the reco and truth samples,
    producing the two histograms needed as inputs to WienerSVD unfolding.
    The CV background (signal != 0) is computed internally from the unmodified
    reco_df weights and subtracted from the fake-data histogram.

    Parameters
    ----------
    reco_df : pd.DataFrame
        Full selected sample (signal + background) with a weights_mc column
        and a signal column (output of define_signal).
    true_df : pd.DataFrame
        True-signal-only sample (e.g. mcsig_df), indexed at truth level,
        with a weights_mc column.
    var : VariableConfig
        Variable configuration supplying column keys and bin edges.
    reco_mask : np.ndarray of bool
        Boolean mask selecting events in reco_df to reweight.
    true_mask : np.ndarray of bool
        Boolean mask selecting events in true_df to reweight.
    weight : float
        Multiplicative scale applied to masked events.

    Returns
    -------
    fake_data_hist : np.ndarray
        Background-subtracted fake measurement histogram (in weights_mc units).
    fake_signal_hist : np.ndarray
        Modified truth-level signal histogram (in weights_mc units).
    """
    reco_df = ensure_lexsorted(reco_df, axis=1)
    true_df = ensure_lexsorted(true_df, axis=1)

    backgr_df = reco_df[reco_df.signal != 0]
    cv_backgr_hist = get_hist1d(
        data=backgr_df[var.var_evt_reco_col], bins=var.bins,
        weights=backgr_df.weights_mc,
    )

    reco_weights = reco_df.weights_mc.values.copy()
    reco_weights[reco_mask] *= weight

    true_weights = true_df.weights_mc.values.copy()
    true_weights[true_mask] *= weight

    fake_data_hist = (
        get_hist1d(data=reco_df[var.var_evt_reco_col], bins=var.bins,
                   weights=reco_weights)
        - cv_backgr_hist
    )
    fake_signal_hist = get_hist1d(
        data=true_df[var.var_nu_col], bins=var.bins,
        weights=true_weights,
    )
    return fake_data_hist, fake_signal_hist
