import numpy as np
import argparse
from scipy.io import loadmat
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold

import plot_predicted_measured_responses as ppr

# clip_vectors should be (10000, 512) for 10000 images and 512 features
exp_design_file = "../experiments/nsd_expdesign.mat"
exp_design = loadmat(exp_design_file)
master_ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)

trials = np.array([30000, 30000, 24000, 22500, 30000, 24000, 30000, 22500])

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

def train_ridge_regression_cv_model (X, y, alphas, n_splits, n_repeats, random_state, scoring):
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

def get_files_from_arguments():
    """
    [get_files_from_args] returns the files specified in the command line arguments.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Run ridge regression with a roi data file and a clip vector file.')
    # Add argument for subject number
    parser.add_argument('subject_number', help='The subject number (1-8)')
    # Add argument for the data file
    parser.add_argument('roi_data_file', help='The path to the roi data file.')
    # Add argument for the vector file
    parser.add_argument('clip_vector_file', help='The path to the clip vector file.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    return args.subject_number, args.roi_data_file, args.clip_vector_file

def ridge_regression(subject_number, roi_data_file_path, clip_vectors_file_path):
    """
    [ridge regression] returns the ridge regression model with cross validation,
    the activation vectors, and matched clip vectors
    """
    data_size = trials[subject_number - 1]
    ordering = master_ordering[:data_size]
    # The first 1000 images were shown to all subjects so it is used as a validation set
    # We train the model using the data after the first 1000 images
    train_mask = ordering >= 1000

    activation = np.load(roi_data_file_path)
    clip_vectors = np.load(clip_vectors_file_path)
    matched_clip_vectors = match_ordering(ordering, clip_vectors)

    model = train_ridge_regression_cv_model(
        matched_clip_vectors[train_mask],
        activation[train_mask], 
        alphas=[1e-3], 
        n_splits=5, 
        n_repeats=3,
        random_state=None,
        scoring='r2'
    )

    return train_mask, ordering, model, activation, matched_clip_vectors

if __name__ == "__main__":
    # clip_vectors.shape = (10000, 512)
    # activation.shape = (30000, 241)
    # ordering.shape = (30000,)

    subject_number, roi_data_file, clip_vector_file = get_files_from_arguments()
    
    train_mask, ordering, model, activation, matched_clip_vectors = ridge_regression(int(subject_number), roi_data_file, clip_vector_file)

    # print_model_metrics(model, matched_clip_vectors, activation, train_mask)
    ppr.plot_validation_response(model, matched_clip_vectors, activation, train_mask, ordering)