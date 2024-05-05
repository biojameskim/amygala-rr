import numpy as np

import plot_predicted_measured_responses as ppr
import ridge_regression as rr
import matplotlib.pyplot as plt
from scipy import stats

def create_noise_ceiling_avg(model, matched_clip_vectors, num_voxels, train_mask, validation, val_ordering_array):
    """
    [create_noise_ceiling] returns 3 arrays: averaged_responses, remaining_responses, predicted_responses.
    This function does NOT actually calculate the correlations but outlines the process for doing so here:

    1. For each image in the validation set, randomly choose 2 of the measured responses and average them.
        - We store these averages in [averaged_responses]
    2. For the remaining response, calculate the correlation between the average measured response and the remaining response.
        - We store these remaining responses in [remaining_responses]
    3. Get the predicted response for each image in the validation set. Calculate the correlation between the average measured response and the predicted response.
        - We store these predicted responses in [predicted_responses]

    First, create a dictionary where the key is the order of the image and the values are the responses for each image.
    Images have 1-3 responses based on how many times they were shown to each subject.
    """
    all_predicted_responses = model.predict(matched_clip_vectors[~train_mask])
    # create a dictionary where the key is the order of the image and the value is the predicted response of the image
    predicted_responses_dict = ppr.create_predicted_responses_dict(val_ordering_array, all_predicted_responses, sort=True)
    # convert the sorted dictionary values to an array
    predicted_responses = np.array(list(predicted_responses_dict.values()))

    # create a dictionary where the key is the order of the image and the values are the 3 responses for each image
    # Sort the dict by key to preserve the order of the responses so we can perform correlation later
    response_dict = ppr.create_responses_dict(val_ordering_array, validation, sort=True) 

    # The order should be preserved as the indices are the same for averaged_responses and remaining_responses
    averaged_responses = np.empty((1000, num_voxels))
    remaining_responses = np.empty((1000, num_voxels))

    for i, key in enumerate(response_dict):    
        values = response_dict[key]
        num_values = len(values)
        if num_values >= 2: # If there are at least 2 values, choose 2 without replacement
            random_indices = np.random.choice(num_values, 2, replace=False) # 2 random indices
        else: # If there is only 1 value, choose it twice
            random_indices = [0, 0]

        random_choices = [values[i] for i in random_indices] # extract the 2 random choices
        average_response = np.mean(random_choices, axis=0) # average the 2 random choices
        remaining_response = np.delete(values, random_indices, axis=0)  # remove the 2 random choices from the array and get the remaining response
        averaged_responses[i] = average_response # add the average response to the [average_response] array
        remaining_responses[i] = remaining_response # add the remaining response to the [remaining_response] array

    return averaged_responses, remaining_responses, predicted_responses


def create_noise_ceiling(model, matched_clip_vectors, num_voxels, train_mask, validation, val_ordering_array):
    """
    [create_noise_ceiling] returns 3 arrays: responses_1, responses_2, predicted_responses.
    This function does NOT actually calculate the correlations but outlines the process for doing so here.
    Once we get these arrays, we can calculate the correlation between the measured responses (responses_1) and the predicted responses.

    1. For each image in the validation set, randomly choose 2 of the measured responses.
        - We store one in [responses_1] and the other in [responses_2]
    2. Get the predicted response for each image in the validation set.
        - We store these predicted responses in [predicted_responses]

    First, create a dictionary where the key is the order of the image and the values are the responses for each image.
    Images have 1-3 responses based on how many times they were shown to each subject.
    """
    all_predicted_responses = model.predict(matched_clip_vectors[~train_mask])
    # create a dictionary where the key is the order of the image and the value is the predicted response of the image
    predicted_responses_dict = ppr.create_predicted_responses_dict(val_ordering_array, all_predicted_responses, sort=True)
    # convert the sorted dictionary values to an array
    predicted_responses = np.array(list(predicted_responses_dict.values()))

    # create a dictionary where the key is the order of the image and the values are the responses for each image
    # Sort the dict by key to preserve the order of the responses so we can perform correlation later
    response_dict = ppr.create_responses_dict(val_ordering_array, validation, sort=True) 

    # The order should be preserved as the indices are the same for averaged_responses and remaining_responses
    responses_1 = np.empty((1000, num_voxels))
    responses_2 = np.empty((1000, num_voxels))

    for i, key in enumerate(response_dict):    
        values = response_dict[key] 
        num_values = len(values)
        if num_values >= 2: # If there are at least 2 values, choose 2 without replacement
            random_indices = np.random.choice(num_values, 2, replace=False) # 2 random indices
        else: # If there is only 1 value, choose it twice
            random_indices = [0, 0]

        random_choices = [values[i] for i in random_indices] # extract the 2 random choices
        responses_1[i] = random_choices[0]
        responses_2[i] = random_choices[1]

    return responses_1, responses_2, predicted_responses

def calculate_corr_and_pvalue(resp1, resp2, num_voxels):
    """
    [calculate_corr_and_pvalue] calculates the correlations and pvalues between responses 1 and 2.
    
    We want to calculate the corr between the average measured response and the remaining response.
    We also want to calculate the corr between the average measured response and the predicted response.
    
    Because we want to see how accurate the prediction at each voxel is, each entry
    in the return vector is the correlation at each voxel.

    averaged_responses.shape = (1000, num_voxels)
    remaining_responses.shape = (1000, num_voxels)
    predicted_responses.shape = (1000, num_voxels)

    Returns [correlations] correlation vector of size (num_voxels, 1)
    Returns [p_values] p-value vector of size (num_voxels, 1)
    """
    correlations = np.empty((num_voxels, 1))
    p_values = np.empty((num_voxels, 1))

    for i in range(0, num_voxels):
        corr, p = stats.pearsonr(resp1[:, i], resp2[:, i])
        correlations[i] = corr
        p_values[i] = p

    return correlations, p_values

def num_sig_voxels(pvals, alpha=0.05):
    """
    [num_sig_voxels] returns the number of significant voxels given the p-values and alpha.
    Adjusts the p-values by the false discovery rate.
    """
    fdr = stats.false_discovery_control(pvals)
    indices = np.where(fdr < alpha)
    return len(indices)
      
def plot_correlations(correlations, correlations2):
    """
    [plot_correlations] compares the 2 correlations using a violin plot.
    """
    x_pos1 = 1
    x_pos2 = 2

    plt.violinplot(correlations, positions=[x_pos1])
    plt.violinplot(correlations2, positions=[x_pos2])
    
    plt.xlabel("Correlation Type")
    plt.ylabel("Correlation")
    plt.title("Correlation Densities")
    
    plt.xticks([x_pos1, x_pos2], ['measured_1 vs measured_2', 'measured_1 vs predicted'])

    plt.show()


if __name__ == "__main__":

    subject_number, roi_data_file, clip_vector_file = rr.get_files_from_arguments()

    train_mask, ordering, model, activation, matched_clip_vectors = rr.ridge_regression(int(subject_number), roi_data_file, clip_vector_file)

    validation = activation[~train_mask]
    val_ordering_array = ordering[~train_mask]

    num_voxels = activation.shape[1] # number of voxels in this ROI

    responses_1, responses_2, predicted_responses = create_noise_ceiling(
        model, matched_clip_vectors, num_voxels, train_mask, validation, val_ordering_array)
    
    observed_correlations, pvals1 = calculate_corr_and_pvalue(responses_1, responses_2, num_voxels)
    predicted_correlations, pvals2 = calculate_corr_and_pvalue(responses_1, predicted_responses, num_voxels)

    plot_correlations(observed_correlations, predicted_correlations)

    



