import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold

import plot_predicted_measured_responses as ppr

# clip_vectors should be (10000, 512) for 10000 images and 512 features
clip_vectors = np.load('../data/10k_normalized_clip_vectors.npy')
# activation should be (30000,241) for 30000 trials and 241 voxels
# 10k images but 30k trials (bc each image appears 3 times)
activation = np.load('../data/beta_L_amygdala.npy')

exp_design_file = "../experiments/nsd_expdesign.mat"
exp_design = loadmat(exp_design_file)
ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)

# The first 1000 images were shown to all subjects so it is used as a validation set
# apply the train_mask to both the clip_vectors and the activation
train_mask = ordering >= 1000

def match_ordering (ordering, clip_vectors):
    """
    [match_ordering] returns the matched ordering of the clip_vectors and the 
    voxel activation vectors.
    This is necessary because [clip_vectors] is the order of the images in the 
    dataset while [ordering] is the order of the images shown to the subjects. 
    """
    matched_clip_vectors = np.zeros((len(ordering), clip_vectors.shape[1]))
    for i in range(len(ordering)):
        matched_clip_vectors[i] = clip_vectors[ordering[i]]
    return matched_clip_vectors

def ridge_regression_cv (X, y, alphas, n_splits, n_repeats, random_state, scoring):
    """
    [ridge_regression_cv] returns the best ridge regression model with cross-validation.
    The model is trained on the input [X] and output [y] with the regularization 
    parameter [alpha].
    
    Input x: (30k, 512) -> (27k, 512)
    Output y: (30k, 241) -> (27k, 241)
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    model = RidgeCV(alphas=alphas, cv=cv, scoring=scoring)
    model = model.fit(X, y)
    return model

def print_model_metrics(model, matched_clip_vectors, activation, train_mask):
    """
    [print_model_metrics] prints the alpha, weights, best score, and R^2 of the model.
    """
    print("alpha: ", model.alpha_)
    # print("weights: ", model.coef_)
    # print("best score (MAE): ", model.best_score_)
    print("R^2:", model.score(matched_clip_vectors[train_mask], activation[train_mask]))
    

if __name__ == "__main__":
    # clip_vectors.shape = (10000, 512)
    # activation.shape = (30000, 241)
    # ordering.shape = (30000,)

    matched_clip_vectors = match_ordering(ordering, clip_vectors)

    model = ridge_regression_cv(
        matched_clip_vectors[train_mask],
        activation[train_mask], 
        alphas=[1e-3], 
        n_splits=5, 
        n_repeats=3,
        random_state=None,
        scoring='r2'
    )

    # print_model_metrics(model, matched_clip_vectors, activation, train_mask)
  
    ppr.plot_validation_response(model, matched_clip_vectors, activation, train_mask, ordering)