# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.utils import check_array
from sklearn.model_selection import cross_val_predict
from sklearn.utils.metaestimators import available_if

class StackingEstimator(BaseEstimator, TransformerMixin):
    """Meta-transformer for adding predictions and/or class probabilities as synthetic feature(s).

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, estimator, passthrough=False, proba_original = True, cv=5, stack_method="auto"):
        """Create a StackingEstimator object.

        Parameters
        ----------
        estimator: object with fit, predict, and predict_proba methods.
            The estimator to generate synthetic features from.
        """
        self.estimator = estimator
        self.passthrough=passthrough
        self.proba_original = proba_original
        self.cv = cv
        self.stack_method = stack_method
        self.method = None

    def fit(self, X, y=None, **fit_params):
        """Fit the StackingEstimator meta-transformer.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        fit_params:
            Other estimator-specific parameters.

        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        self.estimator.fit(X, y, **fit_params)

        if self.stack_method == "auto":
            if getattr(self.estimator, "predict_proba", None):
                self.method = "predict_proba"
            elif getattr(self.estimator, "decision_function", None):
                self.method = "decision_function"
            else:
                self.method = "predict"
        else:
            self.method = self.stack_method

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        if self.cv >1:
            preds = cross_val_predict(estimator=self.estimator, X=X, y=y, cv=self.cv, method=self.method, **fit_params)

            if len(preds.shape) < 2:
                preds = np.reshape(preds, (-1, 1))

            if not self.proba_original:
                if self.method == "predict_proba" and preds.shape[1]==2:
                    preds = preds[:,1:]
            else:
                if self.method=="predict_proba":
                    preds = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), preds))
                
            if self.passthrough:
                preds = np.hstack((preds,  X ))

            return preds
        
        else:
            return self.transform(X)

    def transform(self, X):
        """Transform data by adding two synthetic feature(s).

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            The transformed feature set.
        """

        preds = getattr(self.estimator, self.method)(X)

        if len(preds.shape) < 2:
            preds = np.reshape(preds, (-1, 1))

        if not self.proba_original:
            if self.method == "predict_proba" and preds.shape[1]==2:
                preds = preds[:,1:]
        else:
            if self.method=="predict_proba":
                preds = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), preds))
            
        if self.passthrough:
            preds = np.hstack((preds,  X ))

        return preds

    def _estimator_has(attr):
        """Check if we can delegate a method to the underlying estimator.
        First, we check the first fitted final estimator if available, otherwise we
        check the unfitted final estimator.
        """
        return lambda self: (
            hasattr(self.estimator, attr)
        )

    @available_if(_estimator_has("predict"))
    def predict(self, X, **predict_params):
        return self.estimator.predict(X)
    
    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_params):
        return self.estimator.predict_proba(X)
    
    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X, **predict_params):
        return self.estimator.decision_function(X)
