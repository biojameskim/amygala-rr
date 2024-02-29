import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold

# clip_vectors should be (10000, 512) for 10000 images and 512 features
clip_vectors = np.load('10k_normalized_clip_vectors.npy')
# activation should be (30000,241) for 30000 trials and 241 voxels
# 10k images but 30k trials (bc each image appears 3 times)
activation = np.load('beta_L_amygdala.npy')

exp_design_file = "experiments/nsd_expdesign.mat"
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

def create_averaged_responses_dict(ordering_array, responses):
    """
    [create_averaged_responses_dict] creates a dictionary where the key is the 
    order of the image and the value is the average response of the image.

    For example, order_to_responses = {..., 625: [response1, response2, response3], ...}
    would become: order_to_responses = {..., 625: [np.mean(response1 + response2 + response3)], ...}
    where each response is a 241-dim vector.
    """
    order_to_responses = {}

    # each order should have three responses
    # create a dictionary where key is the order and value is the list of responses
    # For example, order_to_responses = {625: [response1, response2, response3], ...}
    # where each response is a 241-dim vector.
    for i in range(len(responses)):
        key = ordering_array[i]
        if key in order_to_responses:
            order_to_responses[key].append(responses[i])
        else:
            order_to_responses[key] = [responses[i]]

    # iterate through the dictionary and average the responses to get the average measured response for each order
    # For example, order_to_responses = {625: [(response1 + response2 + response3) / 3], ...}
    for key in order_to_responses:
        order_to_responses[key] = np.mean(order_to_responses[key], axis=0)
    return order_to_responses

def plot_validation_response(model, matched_clip_vectors, activation, train_mask):
    """
    [plot_validation_response] plots the average predicted response against the 
    average measured response for each image in the validation set.
    """

    predicted_response = model.predict(matched_clip_vectors[~train_mask])
    measured_response = activation[~train_mask]
    # shape of predicted response (3000, 241)
    # shape of measured response (3000, 241)

    # validation set has 3000 trials
    ordering_array = ordering[~train_mask]

    avg_predicted_response_dict = create_averaged_responses_dict(ordering_array, predicted_response)
    avg_measured_response_dict = create_averaged_responses_dict(ordering_array, measured_response)
    
    # next step is to convert these dictionaries into arrays and plot them
    avg_predicted_response_array = np.array(list(avg_predicted_response_dict.values()))
    avg_measured_response_array = np.array(list(avg_measured_response_dict.values())) 

    # plot the average predicted response against the average measured response
    plt.scatter(avg_measured_response_array, avg_predicted_response_array)
    plt.xlabel('Average Measured Response')
    plt.ylabel('Average Predicted Response')
    plt.title('Average Predicted Response vs. Average Measured Response')

    # plot the line y = x
    x = np.linspace(0, 50, 100)
    y = x
    plt.plot(x, y, '-r', label='y=x')

    plt.show()


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

    # print_model_metrics(model)
  
    plot_validation_response(model, matched_clip_vectors, activation, train_mask)