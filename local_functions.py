import yaml
import numpy as np
import matplotlib.pyplot as plt
import aopy
import seaborn as sn
import os
import copy

def load_yaml_params(filename, figure_type='paper_settings', return_vis_params=True):
    with open(filename, 'r') as file:
        params = yaml.safe_load(file)

    # Redefine often used keys for easy access
    ds_params = params['data_selection_parameters']
    data_paths = params['data_paths']
    save_paths = params['save_paths']
    filenames = params['filenames']
    postproc_params = params['postproc_data_parameters']
    uq_params = params['unit_quality_metric_parameters']
    analysis_params = params['analysis_parameters']
    vis_params = params['visualization_parameters']

    # Set global plotting params
    for plot_param_key in list(params['visualization_parameters'][figure_type].keys()):
        plt.rcParams[plot_param_key]=params['visualization_parameters'][figure_type][plot_param_key]
        
    # Go through and make all save directories if they don't exist
    for path in list(save_paths.keys()):
        if isinstance(save_paths[path], str):
            os.makedirs(save_paths[path], exist_ok=True)    
        elif isinstance(save_paths[path], dict):
            for key2 in list(save_paths[path].keys()):
                os.makedirs(save_paths[path][key2], exist_ok=True)
    
    target_colors = sn.color_palette(n_colors=9)
    if return_vis_params:
        return ds_params, data_paths, save_paths, filenames, postproc_params, uq_params, analysis_params, vis_params, target_colors
    else:
        return ds_params, data_paths, save_paths, filenames, postproc_params, uq_params, analysis_params, target_colors

def compile_raster(data, trigger_idx, tbefore, tafter, samplerate, unit_data=False, smooth=False):
    '''
    args:
        data(ntrial-list of data dict): Each element of the list is a dictionary with each entry being a (nt) array of binned spikes
        trigger_idx (ntrial):
        tbefore (float):
        tafter (float):
        unit_data (bool): If data is labelled units
        samplerate (int): neural data samplerate
    '''
    
    idx_before = int(np.ceil(tbefore*samplerate))
    idx_after = int(np.ceil(tafter*samplerate))    
    ntrials = len(data)
    ntime = int(idx_before+idx_after)
    
    if unit_data:
        unique_unit_labels = np.sort(np.unique([list(itrial_data.keys()) for itrial_data in data]).astype(int))
        nunits = len(unique_unit_labels)
        raster_array = np.zeros((ntime, ntrials, nunits))*np.nan
    else:
        unique_unit_labels = None
        raster_array = np.zeros((ntime, ntrials, data[0].shape[1]))*np.nan
    
    for itrial in range(ntrials):
        start_idx = int(trigger_idx[itrial]-idx_before)
        end_idx = int(trigger_idx[itrial]+idx_after)
        raster_idx_start = int(0)
        raster_idx_end = ntime
        
        if unit_data:
            trial_len = len(data[itrial][str(unique_unit_labels[0])])
        else:
            trial_len = data[itrial].shape[0]
        
        # Contingency if start_idx < 0 or end_idx is longer than the data segment
        if start_idx < 0:
            print('start_idx < 0')
            raster_idx_start = -start_idx
            start_idx = 0
            
        if end_idx > trial_len:
            print('end_idx < trial_len')
            if unit_data:
                trial_len = len(data[itrial][str(unique_unit_labels[itrial])])
            else:
                trial_len = len(data[itrial][itrial])
            raster_idx_end = int(trial_len - trigger_idx[itrial] + idx_before)
            end_idx = trial_len

        if unit_data:
            for iunit, unit_label in enumerate(unique_unit_labels):
                if smooth:
                    # [smooth_timeseries_gaus(data_segs[itr], 1/spike_bin_width_mc, width=smooth_width, nstd=smooth_nstd) for itr in range(len(data_segs))]
                    raster_array[raster_idx_start:raster_idx_end,itrial,iunit] = smooth_timeseries_gaus(data[itrial][str(unit_label)], 1/spike_bin_width_mc, width=smooth_width, nstd=smooth_nstd)[start_idx:end_idx]
                else:
                    raster_array[raster_idx_start:raster_idx_end,itrial,iunit] = data[itrial][str(unit_label)][start_idx:end_idx]
                    
        else:
            raster_array[raster_idx_start:raster_idx_end,itrial,:] = data[itrial][start_idx:end_idx,:]
            
    return raster_array, unique_unit_labels

def get_pseudopopulation_raster(unit_df, raster_list, target_idx_list, nunits, units_to_use=None, return_units_used=False):
    '''
    Args:
        unit_df [dataframe]: must have info on rec number, unitidx
        raster_list [list of (ntime, nch, ntrials) arrays]:
        target_idx_list [list of (ntrial) arrays]
        nunits [int]

    Returns:
        (ntime, ntrial, nunits) pseudopopulation raster array
        (ntrial) target labels
    '''
    # Randomly select nunits from unit_df
    if units_to_use is None:
        units_to_use = np.sort(np.random.choice(np.arange(len(unit_df)), size=nunits, replace=False))
        
    # Get the raster info from each of these units and compile
    ntime, _, ntrials = raster_list[0].shape
    raster_output = np.zeros((ntime, len(units_to_use), ntrials))*np.nan
    for iunit, unit in enumerate(units_to_use):
        rec_number = unit_df['rec_number'][unit]
        unit_idx = unit_df['stable_unit_idx'][unit]
        sorted_target_label_idx = np.argsort(target_idx_list[rec_number]) # Sort target labels for each unit
        raster_output[:,iunit,:] = raster_list[rec_number][:,unit_idx,sorted_target_label_idx]
    
    if return_units_used:
        return raster_output, target_idx_list[rec_number][sorted_target_label_idx], units_to_use
    else:
        return raster_output, target_idx_list[rec_number][sorted_target_label_idx]
    

def compute_CCA(La, Lb):
    '''
    Implemented as described in Gallego 2020 (10.1038/s41593-019-0555-4)
    Args:
        La (ntime, nfeature): Latent dynamics from data A
        Lb (ntime, nfeature): Latent dynamics from data B
    '''
    # Compute Qr decomp to extract the orthonormal basis for the column vectors in the latent dynamics
    Qa, Ra = np.linalg.qr(La)
    Qb, Rb = np.linalg.qr(Lb)

    # Take inner product and perform SVD to get new manifold directions and correlations (S provdes the correlations)
    U, S, V = np.linalg.svd(Qa.T @ Qb)

    # Compute new manifold directions (Ma, Mb)
    Ma = np.linalg.pinv(Ra) @ U
    Mb = np.linalg.pinv(Rb) @ V
    
    # Project neural activity onto shared space:
    La_shared = Ma @ La
    Lb_shared = Mb @ Lb

    return Ma, Mb, S, La_shared, Lb_shared

from scipy.linalg import qr, svd, inv
def canoncorr_gallego(X:np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    
    returns: A,B,r,U,V 
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X,0)
    Y = Y - np.mean(Y,0)

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1,T11,perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0,0])))).eps*max([n,p1]))

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:,:rankX]
        T11 = T11[:rankX,:rankX]

    Q2,T22,perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0,0])))).eps*max([n,p2]))

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:,:rankY]
        T22 = T22[:rankY,:rankY]

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX,rankY)
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:,:d] * np.sqrt(n-1)
    B = inv(T22) @ M[:,:d] * np.sqrt(n-1)
    r = D[:d]
    # remove roundoff errs
    r[r>=1] = 1
    r[r<=0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1,:] = np.vstack((A, np.zeros((p1-rankX,d))))
    B[perm2,:] = np.vstack((B, np.zeros((p2-rankY,d))))
    
    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V
    
def weiner_filter(neural_data, kin_data, regularization=None, alpha=1, lags=None,fit_intercept=True):
    '''
    Calculates the task relevant dimensions by regressing neural activity against kinematic data using least squares.
    If the input neural data is 3D, all trials will be concatenated to calculate the subspace. 
    Calculation is based on the approach used in Sun et al. 2022 https://doi.org/10.1038/s41586-021-04329-x
    
    .. math::
    
        R \\in \\mathbb{R}^{nt \\times nch}
        M \\in \\mathbb{R}^{nt \\times nkin}
        \\beta \\in \\mathbb{R}^{nch \\times nkin}
        R = M\\beta^T
        [\\beta_0 \beta_x \beta_y]^T = (M^T M)^{-1} M^T R

    Args:
        neural_data ((nt, nch)): Input neural data (:math:`R`) to regress against kinematic activity.
        kin_data ((nt, ndim)): Kinematic variables (:math:`M`), commonly position or instantaneous velocity. 'ndims' refers to the number of physical dimensions that define the kinematic data (i.e. X and Y)
        conc_proj_data (bool): If the projected neural data should be concatenated.

    Returns:
        tuple: Tuple containing:
            | **(nch, ndim):** Subspace (:math:`\beta`) that best predicts kinematic variables. Note the first column represents the intercept, then the next dimensions represent the behvaioral variables
            | **((nt, nch) or list of (nt, ndim)):** Neural data projected onto task relevant subspace

    '''        
    # Set input neural data as a float
    neural_data_centered = copy.deepcopy(neural_data.astype(float))

    # Center neural data:
    neural_data_centered -= np.nanmean(neural_data_centered, axis=0)
    ntime = neural_data_centered.shape[0]
    if fit_intercept:
        conc_kin_data = np.ones((ntime, kin_data.shape[1]+1))*np.nan
        conc_kin_data[:,0] = 1
        conc_kin_data[:,1:] = kin_data
    else:
        conc_kin_data = kin_data


    if lags is not None and lags!=0:
        for ilag in range(lags):
            if ilag == 0:
                lagged_ndc = neural_data_centered[lags:,:]
            else:
                lagged_ndc = np.concatenate((lagged_ndc, neural_data_centered[lags-ilag:-ilag,:]), axis=1)

        neural_data_centered = lagged_ndc

    if regularization is None:
        task_subspace = np.linalg.pinv(neural_data_centered.T @ neural_data_centered) @ neural_data_centered.T @ conc_kin_data[lags:,:]
    elif regularization == 'lasso':
        lasso_model = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000).fit(neural_data_centered, conc_kin_data[lags:,:])
        task_subspace = (lasso_model.coef_).T
    elif regularization == 'ridge':
        ridge_model = sklearn.linear_model.Ridge(alpha=alpha, max_iter=10000).fit(neural_data_centered, conc_kin_data[lags:,:])
        task_subspace = (ridge_model.coef_).T
    elif regularization == 'elastic net':
        elasticnet_model = sklearn.linear_model.ElasticNet(alpha=alpha, max_iter=10000).fit(neural_data_centered, conc_kin_data[lags:,:])
        task_subspace = (elasticnet_model.coef_).T
        
    projected_data = neural_data_centered @ task_subspace
    
    return task_subspace, projected_data
def make_lagged_matrix(data, lags):
    if lags == 0:
        return data
    else:
        for ilag in range(lags):
            if ilag == 0:
                lagged_data = data[lags:,:]
            else:
                lagged_data = np.concatenate((lagged_data, data[lags-ilag:-ilag,:]), axis=1)

        return lagged_data
    
def get_neighboring_ccs(ccs, nneighbors=1):
    neighboring_ccs = np.zeros(ccs.shape[0])*np.nan
    for ivalue in range(ccs.shape[0]):
        start_idx = ivalue-nneighbors
        end_idx = ivalue+nneighbors+1
        if start_idx < 0:
            start_idx = 0
        ccs_subset = copy.deepcopy(ccs[:,start_idx:end_idx][start_idx:end_idx,:])
        diag_indices = np.diag_indices(ccs_subset.shape[0])
        ccs_subset[diag_indices[0],diag_indices[1]] = np.nan
        neighboring_ccs[ivalue] = np.nanmean(ccs_subset)
        
    return neighboring_ccs