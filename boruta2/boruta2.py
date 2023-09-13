#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>
        Andrey Lange  <lange_am@mail.ru>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""

from __future__ import print_function, division
import numpy as np
import scipy as sp
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import r2_score
from sklearn.tree import BaseDecisionTree

#from sklearn.tree._tree import DOUBLE, DTYPE
#from sklearn.utils.validation import _check_sample_weight

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices, \
     _get_n_samples_bootstrap

#from sklearn.ensemble._forest import _parallel_build_trees

from joblib import Parallel, delayed, dump, load
import pickle as pkl
import os
import threading
import warnings

#from scipy.sparse import issparse


class Boruta2(BaseEstimator, TransformerMixin):
    """
    Improved Python implementation of the Boruta R package.

    The improvements of this implementation include:
    - Faster run times:
        Thanks to scikit-learn's fast implementation of the ensemble methods.
    - Scikit-learn like interface:
        Use BorutaPy just like any other scikit learner: fit, fit_transform and
        transform are all implemented in a similar fashion.
    - Modularity:
        Any ensemble method could be used: random forest, extra trees
        classifier, even gradient boosted trees.
    - Two step correction:
        The original Boruta code corrects for multiple testing in an overly
        conservative way. In this implementation, the Benjamini Hochberg FDR is
        used to correct in each iteration across active features. This means
        only those features are included in the correction which are still in
        the selection process. Following this, each that passed goes through a
        regular Bonferroni correction to check for the repeated testing over
        the iterations.
    - Percentile:
        Instead of using the max values of the shadow features the user can
        specify which percentile to use. This gives a finer control over this
        crucial parameter. For more info, please read about the perc parameter.
    - Automatic tree number:
        Setting the n_estimator to 'auto' will calculate the number of trees
        in each itartion based on the number of features under investigation.
        This way more trees are used when the training data has many feautres
        and less when most of the features have been rejected.
    - Ranking of features:
        After fitting BorutaPy it provides the user with ranking of features.
        Confirmed ones are 1, Tentatives are 2, and the rejected are ranked
        starting from 3, based on their feautre importance history through
        the iterations.

    # --- Added by A. Lange---:
    - Automatic dumping RandomForestRegressor() parameters and feature_importance_ 
      on every iteration. When retraining a detection if such a Random Forest model was already 
      trained is performed and the feature_importance_ is loaded. 
      If to set higher number of trees, then the ensemble is extended by training 
      and adding the missing trees to it and redumped.

    We highly recommend using pruned trees with a depth between 3-7.

    For more, see the docs of these functions, and the examples below.

    Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

    Boruta is an "all relevant" feature selection method, while most other are
    "minimal optimal"; this means it tries to find all features carrying
    information usable for prediction, rather than finding a possibly compact
    subset of features on which some classifier has a minimal error.

    Why bother with all relevant feature selection?
    When you try to understand the phenomenon that made your data, you should
    care about all factors that contribute to it, not just the bluntest signs
    of it in context of your methodology (yes, minimal optimal set of features
    by definition depends on your classifier choice).

    Parameters
    ----------

    estimator : object
        A supervised learning estimator, with a 'fit' method that returns the
        feature_importances_ attribute. Important features must correspond to
        high absolute values in the feature_importances_.

    n_estimators : int or string, default = 1000
        If int sets the number of estimators in the chosen ensemble method.
        If 'auto' this is determined automatically based on the size of the
        dataset. The other parameters of the used estimators need to be set
        with initialisation.

    perc : int, default = 100
        Instead of the max we use the percentile defined by the user, to pick
        our threshold for comparison between shadow and real features. The max
        tend to be too stringent. This provides a finer control over this. The
        lower perc is the more false positives will be picked as relevant but
        also the less relevant features will be left out. The usual trade-off.
        The default is essentially the vanilla Boruta corresponding to the max.

    alpha : float, default = 0.05
        Level at which the corrected p-values will get rejected in both
        correction steps.

    two_step : Boolean, default = True
        If you want to use the original implementation of Boruta with Bonferroni
        correction only set this to False.

    max_iter : int, default = 100
        The number of maximum iterations to perform.

    random_state : int, RandomState instance or None; default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays iteration number
        - 2: which features have been selected already

    early_stopping : bool, default = False
        Whether to use early stopping to terminate the selection process
        before reaching `max_iter` iterations if the algorithm cannot
        confirm a tentative feature for `n_iter_no_change` iterations.
        Will speed up the process at a cost of a possibility of a
        worse result.
        
    n_iter_no_change : int, default = 20
        Ignored if `early_stopping` is False. The maximum amount of
        iterations without confirming a tentative feature. 

    # --- Added by A.Lange ---

    dump_rf_path: str or None, default None
        Dump the results of RandomForestRegressor after each iteration to the specified folder.
        When restarting BorutaPy, the folder is checked for the dump of every iteration.
        If the dump of feature_importance is found for the current iteration, 
        then it is loaded and the training of the RF is skipped. 
        If the number of trees in the dumped RF model is less
        than the number calculated depending on n_estimators, 
        then the missing random trees are added to the ensemble and the total importance is updated and saved.
        If None, then no dump/load is performed.

    mean_feature_repr: int, default = 100
        Average number of the occurrences of every feature in the whole ensemble,
        which is used to detect the number of trees when n_estimators == 'auto'.
        n_estimators is found using the following expression:
        mean_feature_repr = n * p (a.k.a binomial mean), where n = n_estimators * n_splits, 
        n_splits is the average (over all trees) number of splits in one tree 
        (less or equal to 2**max_depth-1, which is the number of intermediate nodes in the perfect tree), 
        p is the probability of the feature occurrence in one split in a tree,
        for ex., for RandomForest p = sqrt(2*n_features)/(2*n_features), 
        and the multiplier 2 is due to shaddow permuted features are added in Boruta. 

    importance_normalize: bool, default = True
        Method for the calculation of feature_importance if the estimator is RandomForestRegressor. 
        By default, the importance of every tree is normalized: 
        the importances of all features sum up to 1.0, as sklearn does in tree_.compute_feature_importances(normalize=True)),
        and the .feature importance_ is the average over all the trees. 
        When importance_normalize==False we sum over unnormalized importance vectors of all the trees and normalize the resulted sum.

    Attributes
    ----------

    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features - only confirmed ones are True.

    support_weak_ : array of shape [n_features]
        The mask of selected tentative features, which haven't gained enough
        support during the max_iter number of iterations..

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1 and tentative features are assigned
        rank 2.

    importance_history_ : array-like, shape [n_features, n_iters]
        The calculated importance values for each feature across all iterations.  

    Examples
    --------
    
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    
    # load X and y
    # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    y = y.ravel()
    
    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    
    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
    
    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)
    
    # check selected features - first 5 features are selected
    feat_selector.support_
    
    # check ranking of features
    feat_selector.ranking_
    
    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    References
    ----------

    [1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
        Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
    """

    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 two_step=True, max_iter=100, random_state=None, verbose=0,
                 early_stopping=False, n_iter_no_change=20, 
                 dump_rf_path=None, mean_feature_repr=100, 
                 importance_normalize=True):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.dump_rf_path = dump_rf_path
        self.mean_feature_repr = mean_feature_repr
        self.importance_normalize = importance_normalize
        self.__version__ = '0.3.1'
        self._is_lightgbm = 'lightgbm' in str(type(self.estimator))

    def fit(self, X, y):
        """
        Fits the Boruta feature selection with the provided estimator.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """

        return self._fit(X, y)

    def transform(self, X, weak=False, return_df=False):
        """
        Reduces the input X to the features selected by Boruta.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.
        
        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        return self._transform(X, weak, return_df)

    def fit_transform(self, X, y, weak=False, return_df=False):
        """
        Fits Boruta, then reduces the input X to the selected features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.

        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        self._fit(X, y)
        return self._transform(X, weak, return_df)

    def _validate_pandas_input(self, arg):
        try:
            return arg.values
        except AttributeError:
            raise ValueError(
                "input needs to be a numpy array or pandas data frame."
            )

    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X) 
        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        self.random_state = check_random_state(self.random_state)
        
        early_stopping = False
        if self.early_stopping:
            if self.n_iter_no_change >= self.max_iter:
                if self.verbose > 0:
                    print(
                        f"n_iter_no_change is bigger or equal to max_iter"
                        f"({self.n_iter_no_change} >= {self.max_iter}), "
                        f"early stopping will not be used."
                    )
            else:
                early_stopping = True
        
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # average number of splits in a tree
        _n_splits = None
        # early stopping vars
        _same_iters = 1
        _last_dec_reg = None
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        if hasattr(self.estimator, 'warm_start'):
            self.estimator.set_params(warm_start=False)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected, _n_splits)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            if self._is_lightgbm:
                self.estimator.set_params(random_state=self.random_state.randint(0, 10000))
            else:
                self.estimator.set_params(random_state=self.random_state)  # already not int 

            # add shadow attributes, shuffle them and train estimator, get importance 
            # and the average number of splits over the trees in the ensemble if possible (None otherwise) 
            cur_imp, _n_splits = self._add_shadows_get_imps(X, y, dec_reg, _iter)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            print('imp_sha_max p_value 1.0', (cur_imp[0] < np.percentile(cur_imp[1], 100)).sum()/len(cur_imp[0]))
            print('imp_real_min p_value 0.0', (cur_imp[1] > np.percentile(cur_imp[0], 0)).sum()/len(cur_imp[1]))

            print('imp_sha_max p_value 0.99', (cur_imp[0] < np.percentile(cur_imp[1], 99)).sum()/len(cur_imp[0]))
            print('imp_real_min p_value 0.01', (cur_imp[1] > np.percentile(cur_imp[0], 1)).sum()/len(cur_imp[1]))

            print('imp_sha_max p_value 0.95', (cur_imp[0] < np.percentile(cur_imp[1], 95)).sum()/len(cur_imp[0]))
            print('imp_real_min p_value 0.05', (cur_imp[1] > np.percentile(cur_imp[0], 5)).sum()/len(cur_imp[1]))

            print('imp_sha_max p_value 0.9', (cur_imp[0] > np.percentile(cur_imp[1], 90)).sum()/len(cur_imp[0]))
            print('imp_real_min p_value 0.1', (cur_imp[1] > np.percentile(cur_imp[0], 10)).sum()/len(cur_imp[1]))

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1
                
            # early stopping
            if early_stopping:
                if _last_dec_reg is not None and (_last_dec_reg == dec_reg).all():
                    _same_iters += 1
                    if self.verbose > 0:
                        print(
                            f"Early stopping: {_same_iters} out "
                            f"of {self.n_iter_no_change}"
                        )
                else:
                    _same_iters = 1
                    _last_dec_reg = dec_reg.copy()
                if _same_iters > self.n_iter_no_change:
                    break

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # calculate ranks in each iteration, then median of ranks across feats
            iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
            rank_medians = np.nanmedian(iter_ranks, axis=0)
            ranks = self._nanrankdata(rank_medians, axis=0)
            
            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=bool)

        self.importance_history_ = imp_history

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        return self

    def _transform(self, X, weak=False, return_df=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')

        if weak:
            indices = self.support_ + self.support_weak_
        else:
            indices = self.support_

        if return_df:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]
        return X

    def _get_tree_num(self, n_feat, n_splits):
        depth = None
        try:
            params = self.estimator.get_params() 
            depth = params['max_depth']
        except KeyError:
            warnings.warn(
                "The estimator does not have a max_depth property or the get_params() method, as a result "
                " the number of trees to use cannot be estimated automatically."
            )

        # the probality that a feature is selected during one split
        if 'max_features' in params:
            max_feat = params['max_features']
            if max_feat == 'sqrt':
                # n_feat * 2 because the training matrix is extended with n shadow features
                p = np.sqrt(n_feat * 2) / (n_feat * 2)
            elif max_feat == 'log2':
                p = np.log2(n_feat * 2) / (n_feat * 2)
            elif isinstance(max_feat, int):
                p = max_feat / (n_feat * 2)
            elif isinstance(max_feat, float):
                p = max_feat
            elif max_feat is None:
                p = 1.0
            else:
                p = 1.0
                warnings.warn("Incorrect max_features, set 1.0 by default.")
        else:
            p = 1.0
            warnings.warn(
                "The estimator does not have max_features property, by default "
                " it is considered as if it is 1.0."
            )

        if depth is None:
            depth = 1

        # the maximal n_splits is 2**depth-1, which is the number of intermediate nodes in the perfect tree.
        # to avoid the underestimating the n_estimators we reduce n_splits downto 2**max((depth - 1), 1) - 1  
        if n_splits is None:
            n_splits = 2**max((depth - 1), 1) - 1  

        n_estimators = int(self.mean_feature_repr / (p * n_splits))
        print('number of trees estimated:', n_estimators)
        return n_estimators

    def _get_imp(self, X, y, dec_reg, _iter):
        """
        Calculate feature_importance, 
        with (by A. Lange) dump/load, and estimate the average n_splits in a tree.
        """
        model = self.estimator
        
        if isinstance(model, RandomForestRegressor):
            if self.dump_rf_path is None:  # no dump/load
                try:
                    model.oob_score = False
                    model.fit(X, y)
                    n_splits = get_avg_n_splits(model)
                    print('oob_score:', get_oob_score(model, X, y))
                except Exception as e:
                    raise ValueError('Please check your X and y variable.'
                                     'RandomForestRegressor cannot be fitted to your data.\n' + str(e))
                if self.importance_normalize:
                    return model.feature_importances_, n_splits
                    # equivalent to 
                    #   _, fi1 = compute_feature_importances(model) 
                    #   return finalize_normalization(fi1)
                else:
                    fi0, _ = compute_feature_importances(model)
                    return finalize_normalization(fi0), n_splits
            else:  # try to load, analyze params and n_estimators, and dump/redump
                params = model.get_params()
                fname = os.path.join(self.dump_rf_path, str(_iter)+'.pkl')

                add_missing = False  # retrain and redump RF 
                if os.path.isfile(fname):
                    loaded = load(open(fname, 'rb'))
                    if (loaded['dec_reg'] == dec_reg).all():
                        # changing of these estimator params does not lead to RF re-training
                        ignored_params = ['n_estimators', 'n_jobs', 'random_state', 'oob_score', 'verbose']
                        if {k: v for k, v in params.items() if k not in ignored_params} == \
                           {k: v for k, v in loaded['params'].items() if k not in ignored_params}:
                            if params['n_estimators'] > loaded['params']['n_estimators']:
                                add_missing = True  # add missing trees and redump
                            else:  # just load importances without retrain RF
                                print('RF importance loaded from', fname)
                                if self.importance_normalize:
                                    return finalize_normalization(loaded['fi1']), loaded['n_splits']
                                else:
                                    return finalize_normalization(loaded['fi0']), loaded['n_splits']
                        else:
                            print(fname, 'found, but params has changed - retrain...')
                    else:
                        print(fname, 'found, but dec_reg is different - retrain...')
                
                if add_missing:  # add missing trees, random_state should be changed to avoid similar trees
                    n_missing_trees = params['n_estimators'] - loaded['params']['n_estimators']
                    model.set_params(n_estimators=n_missing_trees)
                    print(fname, 'found, adding missing', n_missing_trees, 'trees')

                try:
                    model.fit(X, y)
                except Exception as e:
                    raise ValueError('Please check your X and y variable.'
                                     'RandomForestRegressor cannot be fitted to your data.\n' + str(e))
                
                if add_missing:
                    if n_missing_trees > loaded['params']['n_estimators']:
                        n_splits = get_avg_n_splits(model)
                    else:
                        n_splits = loaded['n_splits']
                else:
                    n_splits = get_avg_n_splits(model)

                fi0, fi1 = compute_feature_importances(model)
                
                if add_missing == 1:  # update importances
                    fi0 += loaded['fi0']
                    fi1 += loaded['fi1']
                
                to_dump = {'iter': _iter, 'dec_reg': dec_reg, 
                           'params': params, 'fi0': fi0, 'fi1': fi1, 'n_splits': n_splits}
                dump(to_dump, open(fname, "wb"), protocol=pkl.HIGHEST_PROTOCOL)
                
                if self.importance_normalize:
                    return finalize_normalization(fi1), n_splits
                else:
                    return finalize_normalization(fi0), n_splits
                
        else:  # other estimators
            try:
                model.fit(X, y)
            except Exception as e:
                raise ValueError('Please check your X and y variable, and the estimator passed.'
                                 'It cannot be fitted to your data.\n' + str(e))
            
            try:
                n_splits = get_avg_n_splits(model)
            except Exception:
                n_splits = None

            try:
                return model.feature_importances_, n_splits
            except Exception:
                raise ValueError('Only methods with feature_importance_ attribute '
                                 'are currently supported in BorutaPy.')

    def _get_shuffle(self, seq):
        self.random_state.shuffle(seq)
        return seq

    def _add_shadows_get_imps(self, X, y, dec_reg, _iter):
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = np.copy(X[:, x_cur_ind])
        x_cur_w = x_cur.shape[1]
        # deep copy the matrix for the shadow matrix
        x_sha = np.copy(x_cur)
        # make sure there's at least 5 columns in the shadow matrix for
        while (x_sha.shape[1] < 5):
            x_sha = np.hstack((x_sha, x_sha))
        # shuffle xSha
        x_sha = np.apply_along_axis(self._get_shuffle, 0, x_sha)
        # get importance of the merged matrix
        imp, n_splits = self._get_imp(np.hstack((x_cur, x_sha)), y, dec_reg, _iter)
        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]
        return (imp_real, imp_sha), n_splits

    def _assign_hits(self, hit_reg, cur_imp, imp_sha_max):
        # register hits for features that did better than the best of shadows
        cur_imp_no_nan = cur_imp[0]
        cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
        hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()
        
        if self.two_step:
            # two step multicor process
            # first we correct for testing several features in each round using FDR
            to_accept = self._fdrcorrection(to_accept_ps, alpha=self.alpha)[0]
            to_reject = self._fdrcorrection(to_reject_ps, alpha=self.alpha)[0]
            
            # second we correct for testing the same feature over and over again
            # using bonferroni
            to_accept2 = to_accept_ps <= self.alpha / float(_iter)
            to_reject2 = to_reject_ps <= self.alpha / float(_iter)
        
            # combine the two multi corrections, and get indexes
            to_accept *= to_accept2
            to_reject *= to_reject2
        else:
            # as in th original Boruta, we simply do bonferroni correction
            # with the total n_feat in each iteration
            to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
            to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))
        
        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]
        
        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg

    def _fdrcorrection(self, pvals, alpha=0.05):
        """
        Benjamini/Hochberg p-value correction for false discovery rate, from
        statsmodels package. Included here for decoupling dependency on statsmodels.

        Parameters
        ----------
        pvals : array_like
            set of p-values of the individual tests.
        alpha : float
            error rate

        Returns
        -------
        rejected : array, bool
            True if a hypothesis is rejected, False if not
        pvalue-corrected : array
            pvalues adjusted for multiple hypothesis testing to limit FDR
        """
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs + 1) / float(nobs)

        reject = pvals_sorted <= ecdffactor * alpha
        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        # reorder p-values and rejection mask to original order of pvals
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_

    def _nanrankdata(self, X, axis=1):
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """
        Check hyperparameters as well as X and y before proceeding with fit.
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')

    def _print_results(self, dec_reg, _iter, flag):
        n_iter = str(_iter) + ' / ' + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ['Iteration: ', 'Confirmed: ', 'Tentative: ', 'Rejected: ']

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self.verbose == 1:
                output = cols[0] + n_iter
            elif self.verbose > 1:
                output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            n_rejected = np.sum(~(self.support_ | self.support_weak_))
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            output = "\n\nBorutaPy finished running.\n\n" + result
        print(output)


# ----- By A. Lange -----

def compute_feature_importances(estimator):
    def get_tree_importance(tree):
        fi = np.zeros(tree.n_features)
        for i in range(tree.node_count):
            left = tree.children_left[i]
            right = tree.children_right[i]
            if left >= 0 and right >= 0:
                f = tree.feature[i]
                assert(f >= 0)
                fi[f] += tree.weighted_n_node_samples[i] * tree.impurity[i] - \
                    tree.weighted_n_node_samples[left] * tree.impurity[left] - \
                    tree.weighted_n_node_samples[right] * tree.impurity[right]
    
        fi /= tree.weighted_n_node_samples[0]
    
        # do not normalize every tree, aka tree_.compute_feature_importances(normalize=False)
        fi0 = fi  
    
        # normalize every tree, aka tree_.compute_feature_importances(normalize=True),
        # also used in .feature_importance_ calculation
        fi1 = np.copy(fi)
        normalizer = np.sum(fi1)
        if normalizer > 0.0:
            fi1 /= normalizer
                    
        return fi0, fi1

    if isinstance(estimator, BaseDecisionTree):
        return get_tree_importance(estimator.tree_)
    else:
        fi0, fi1 = zip(*[get_tree_importance(e.tree_) for e in estimator.estimators_])
        return np.array(fi0).sum(axis=0), np.array(fi1).sum(axis=0)


def get_avg_n_splits(estimator):
    def get_tree_n_splits(tree):
        n_splits = 0
        for i in range(tree.node_count):
            if tree.children_left[i] >= 0 and tree.children_right[i] >= 0:
                n_splits += 1
        return n_splits

    if isinstance(estimator, BaseDecisionTree):
        return get_tree_n_splits(estimator.tree_)
    else:
        return np.mean([get_tree_n_splits(e.tree_) for e in estimator.estimators_])


def finalize_normalization(fi):
    return fi / np.sum(fi)  # for fi1 the sum(fi1) equals to the number of trees


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


def get_oob_score(model, X, y):
    """
    Compute the OOB score by model.oob_score function (r2_score by default). 
    Based on scikit-learn source code but parallelized over trees.

    Parameters
    ----------
    model: RandomForestRegressor
    X    : array-like of shape (n_samples, n_features)
           The data matrix.
    y    : ndarray of shape (n_samples, n_outputs)
           The target matrix.

    Returns
    -------
    oob_score value, model.oob_score_ can be set to it. 
    """

    def accumulate_predictions(oob_pred_func, estimator, ui, X, oob_pred, n_oop_pred, lock):
        y_pred = oob_pred_func(estimator, X[ui, :])
        with lock:
            oob_pred[ui, ...] += y_pred
            n_oob_pred[ui, :] += 1

    if not (model.bootstrap and (isinstance(model.oob_score, bool) or callable(model.oob_score))):
        return None
    
    n_samples = y.shape[0]
    n_outputs = model.n_outputs_
    # for regression, n_classes_ does not exist and we create an empty
    # axis to be consistent with the classification case and make
    # the array operations compatible with the 2 settings
    oob_pred_shape = (n_samples, 1, n_outputs)

    oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float64)
    n_oob_pred = np.zeros((n_samples, n_outputs), dtype=np.int64)

    n_samples_bootstrap = _get_n_samples_bootstrap(
        n_samples,
        model.max_samples,
    )

    unsampled_indices = []
    for estimator in model.estimators_:
        ui = _generate_unsampled_indices(
            estimator.random_state,
            n_samples,
            n_samples_bootstrap,
        )
        unsampled_indices.append(ui)

    lock = threading.Lock()
    Parallel(n_jobs=model.n_jobs, require="sharedmem")(
                delayed(accumulate_predictions)(model._get_oob_predictions, e, ui, X, oob_pred, n_oob_pred, lock)
                for ui, e in zip(unsampled_indices, model.estimators_)
            )

    for k in range(n_outputs):
        if (n_oob_pred == 0).any():
            warnings.warn(
                "Some inputs do not have OOB scores. This probably means "
                "too few trees were used to compute any reliable OOB "
                "estimates."
                )
            n_oob_pred[n_oob_pred == 0] = 1
        oob_pred[..., k] /= n_oob_pred[..., [k]]
    
    if oob_pred.shape[-1] == 1:
        # drop the n_outputs axis if there is a single output
        oob_pred = oob_pred.squeeze(axis=-1)
    
    scoring_function = model.oob_score if callable(model.oob_score) else r2_score

    return scoring_function(y, oob_pred)


# def myfit(model, X, y, sample_weight=None):
#     """
#     Build a forest of trees from the training set (X, y).

#     Parameters
#     ----------
#     X : {array-like, sparse matrix} of shape (n_samples, n_features)
#         The training input samples. Internally, its dtype will be converted
#         to ``dtype=np.float32``. If a sparse matrix is provided, it will be
#         converted into a sparse ``csc_matrix``.

#     y : array-like of shape (n_samples,) or (n_samples, n_outputs)
#         The target values (class labels in classification, real numbers in
#         regression).

#     sample_weight : array-like of shape (n_samples,), default=None
#         Sample weights. If None, then samples are equally weighted. Splits
#         that would create child nodes with net zero or negative weight are
#         ignored while searching for a split in each node. In the case of
#         classification, splits are also ignored if they would result in any
#         single class carrying a negative weight in either child node.

#     Returns
#     -------
#     self : object
#         Fitted estimator.
#     """
#     # Validate or convert input data
#     X, y = model._validate_data(
#         X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
#     )
#     if sample_weight is not None:
#         sample_weight = _check_sample_weight(sample_weight, X)

#     if issparse(X):
#         # Pre-sort indices to avoid that each individual tree of the
#         # ensemble sorts the indices.
#         X.sort_indices()

#     y = np.atleast_1d(y)
#     if y.ndim == 2 and y.shape[1] == 1:
#         warnings.warn(
#                 "A column-vector y was passed when a 1d array was"
#                 " expected. Please change the shape of y to "
#                 "(n_samples,), for example using ravel()."
#         )

#     if y.ndim == 1:
#         # reshape is necessary to preserve the data contiguity against vs
#         # [:, np.newaxis] that does not.
#         y = np.reshape(y, (-1, 1))

#     if model.criterion == "poisson":
#         if np.any(y < 0):
#             raise ValueError(
#                 "Some value(s) of y are negative which is "
#                 "not allowed for Poisson regression."
#             )
#         if np.sum(y) <= 0:
#             raise ValueError(
#                 "Sum of y is not strictly positive which "
#                 "is necessary for Poisson regression."
#             )

#     model.n_outputs_ = y.shape[1]

#     y, expanded_class_weight = model._validate_y_class_weight(y)

#     if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
#         y = np.ascontiguousarray(y, dtype=DOUBLE)

#     if expanded_class_weight is not None:
#         if sample_weight is not None:
#             sample_weight = sample_weight * expanded_class_weight
#         else:
#             sample_weight = expanded_class_weight

#     if not model.bootstrap and model.max_samples is not None:
#         raise ValueError(
#             "`max_sample` cannot be set if `bootstrap=False`. "
#             "Either switch to `bootstrap=True` or set "
#             "`max_sample=None`."
#         )
#     elif model.bootstrap:
#         n_samples_bootstrap = _get_n_samples_bootstrap(
#             n_samples=X.shape[0], max_samples=model.max_samples
#         )
#     else:
#         n_samples_bootstrap = None

#     model._validate_estimator()

#     if not model.bootstrap and model.oob_score:
#         raise ValueError("Out of bag estimation only available if bootstrap=True")

#     random_state = check_random_state(model.random_state)

#     trees = [
#         model._make_estimator(append=False, random_state=random_state)
#         for i in range(model.n_estimators)
#     ]

#     # Parallel loop: we prefer the threading backend as the Cython code
#     # for fitting the trees is internally releasing the Python GIL
#     # making threading more efficient than multiprocessing in
#     # that case. However, for joblib 0.12+ we respect any
#     # parallel_backend contexts set at a higher level,
#     # since correctness does not rely on using threads.
#     trees = Parallel(
#         n_jobs=model.n_jobs,
#         verbose=model.verbose,
#         prefer="threads"
#     )(
#         delayed(_parallel_build_trees)(
#             t,
#             model.bootstrap,
#             X,
#             y,
#             sample_weight,
#             i,
#             len(trees),
#             verbose=model.verbose,
#             class_weight=model.class_weight,
#             n_samples_bootstrap=n_samples_bootstrap,
#         )
#         for i, t in enumerate(trees)
#     )

#     model.estimators_ = list(trees)

#     return model
