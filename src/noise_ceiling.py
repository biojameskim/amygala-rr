import numpy as np

import plot_predicted_measured_responses as ppr
import ridge_regression as rr
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

def create_noise_ceiling(model, matched_clip_vectors, train_mask, validation, val_ordering_array):
    """
    1. For each image in the validation set, randomly choose 2 of the 3 measured responses and average them.
    2. For the remaining response, calculate the correlation between the average measured response and the remaining response.
    3. Get the predicted response for each image in the validation set. Calculate the correlation
    between the average measured response and the predicted response.

    First, create a dictionary where the key is the order of the image and the values are the 3 responses for each image.
    """
    all_predicted_responses = model.predict(matched_clip_vectors[~train_mask])
    # create a dictionary where the key is the order of the image and the value is the predicted response of the image
    predicted_responses_dict = ppr.create_predicted_responses_dict(val_ordering_array, all_predicted_responses, sort=True)
    # convert the sorted dictionary values to an array
    predicted_responses = np.array(list(predicted_responses_dict.values()))

    # create a dictionary where the key is the order of the image and the values are the 3 responses for each image
    # Sort the dict by key to preserve the order of the responses so we can perform correlation later
    response_dict = ppr.create_responses_dict(val_ordering_array, validation, sort=True) 
    
    # iterate through the dictionary and randomly choose 2 of the 3 measured responses, average them, and put them in an array. Put the remaining one in a separate array.
    # For example, order_to_responses = {625: [average_response, remaining_response], ...}

    # The order should be preserved as the indices are the same for averaged_responses and remaining_responses
    averaged_responses = np.empty((1000, 241))
    remaining_responses = np.empty((1000, 241))

    for i, key in enumerate(response_dict):    
        values = response_dict[key] 
        random_indices = np.random.choice(3, 2, replace=False) # 2 random indices
        random_choices = [values[i] for i in random_indices] # extract the 2 random choices
        average_response = np.mean(random_choices, axis=0) # average the 2 random choices
        remaining_response = np.delete(values, random_indices, axis=0)  # remove the 2 random choices from the array and get the remaining response
        averaged_responses[i] = average_response # add the average response to the [average_response] array
        remaining_responses[i] = remaining_response # add the remaining response to the [remaining_response] array

    return averaged_responses, remaining_responses, predicted_responses

def calculate_correlation(averaged_responses, remaining_responses, predicted_responses):
    """
    [calculate_correlation] calculates the correlation between the average measured response and the remaining response.
    It also calculates the correlation between the average measured response and the predicted response.
    
    Because we want to see how accurate the prediction at each voxel is, each entry
    in the return vector is the correlation at each voxel.

    averaged_responses.shape = (1000, 241)
    remaining_responses.shape = (1000, 241)
    predicted_responses.shape = (1000, 241)

    Returns correlation vectors of size (241, 1)
    """
    corr_1 = np.empty((241, 1))
    corr_2 = np.empty((241, 1))

    for i in range(0,241):
        corr_1[i] = np.correlate(averaged_responses[:,i], remaining_responses[:,i])
        corr_2[i] = np.correlate(averaged_responses[:,i], predicted_responses[:,i])

    return corr_1, corr_2

def compute_p_values(corr_1, corr_2):
    """
    [compute_p_values] computes the p-value of the 2 correlations.
    """
    ttest_corr1 = ttest_1samp(a=corr_1, popmean=0)
    ttest_corr2 = ttest_1samp(a=corr_2, popmean=0)
    
    return ttest_corr1.pvalue, ttest_corr2.pvalue
      
def plot_noise_ceiling(corr_1, corr_2):
    """
    [plot_noise_ceiling] compares the 2 correlations using a violin plot.
    """
    pass


if __name__ == "__main__":

    matched_clip_vectors = rr.match_ordering(rr.ordering, rr.clip_vectors)

    model = rr.train_ridge_regression_cv_model(
        matched_clip_vectors[rr.train_mask],
        rr.activation[rr.train_mask], 
        alphas=[1e-3], 
        n_splits=5, 
        n_repeats=3,
        random_state=None,
        scoring='r2'
    )

    validation = rr.activation[~rr.train_mask]
    val_ordering_array = rr.ordering[~rr.train_mask]

    averaged_responses, remaining_responses, predicted_responses = create_noise_ceiling(
        model, matched_clip_vectors, rr.train_mask, validation, val_ordering_array)
    
    observed_correlations, predicted_correlations = calculate_correlation(averaged_responses, remaining_responses, predicted_responses)

    pvalue1, pvalue2 = compute_p_values(observed_correlations, predicted_correlations)




