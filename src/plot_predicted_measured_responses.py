import numpy as np
import matplotlib.pyplot as plt

def create_responses_dict(ordering_array, responses, sort=False):
    """
    [create_responses_dict] creates a dictionary where the key is the 
    order of the image and the values are the three responses of the image.
    If [sort] is True, then the dictionary is sorted by key.

    For example, [order_to_responses = {..., 625: [response_1, response_2, response_3], ...}]
    where each response is a 241-dim vector. 
    """
    order_to_responses = {}

    for i in range(len(ordering_array)):
        key = ordering_array[i]
        if key in order_to_responses:
            order_to_responses[key].append(responses[i])
        else:
            order_to_responses[key] = [responses[i]]

    if sort:
        return dict(sorted(order_to_responses.items()))
    return order_to_responses

def average_dictionary_values(dictionary):
    """
    [average_dictionary_values] averages the values of the dictionary.

    For example, [order_to_responses = {..., 625: [np.mean(response_1 + response_2 + response_3)], ...}]
    where each response is a 241-dim vector.
    """
    for key in dictionary:
        dictionary[key] = np.mean(dictionary[key], axis=0)
    return dictionary

def create_predicted_responses_dict(ordering_array, responses, sort=False):
    """
    [create_predicted_responses_dict] creates a dictionary where the key is the
    order of the image and the value is the predicted response of the image.
    If [sort] is True, then the dictionary is sorted by key.

    Since the predicted response is the same for each image, we can just take the
    first response and use that as the predicted response.
    For example, order_to_responses = {..., 624: [response_1], 625: [response_1], ...}
    """
    order_to_responses = {}

    for i in range(len(ordering_array)):
        key = ordering_array[i]
        if key not in order_to_responses:
            order_to_responses[key] = responses[i]

    if sort:
        return dict(sorted(order_to_responses.items()))
    
    return order_to_responses

def plot_validation_response(model, matched_clip_vectors, activation, train_mask, ordering_array):
    """
    [plot_validation_response] plots the average predicted response against the 
    average measured response for each image in the validation set.
    """

    # ~train_mask is the validation set
    predicted_responses = model.predict(matched_clip_vectors[~train_mask])
    measured_responses = activation[~train_mask]
    # shape of predicted responses (3000, 241)
    # shape of measured responses (3000, 241)

    # validation set has 3000 trials
    ordering_array = ordering_array[~train_mask]

    predicted_response_dict = create_predicted_responses_dict(ordering_array, predicted_responses)
    avg_measured_response_dict = average_dictionary_values(create_responses_dict(ordering_array, measured_responses))
    
    # next step is to convert these dictionaries into arrays and plot them
    predicted_response_array = np.array(list(predicted_response_dict.values()))
    avg_measured_response_array = np.array(list(avg_measured_response_dict.values())) 

    # plot the average predicted response against the average measured response
    plt.scatter(avg_measured_response_array, predicted_response_array)
    plt.xlabel('Average Measured Response')
    plt.ylabel('Predicted Response')
    plt.title('Predicted Response vs. Average Measured Response')

    # plot the line y = x
    x = np.linspace(-3, 3, 100)
    y = x
    plt.plot(x, y, '-r', label='y=x')

    plt.show()