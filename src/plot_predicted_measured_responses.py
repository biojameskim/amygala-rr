import numpy as np
import matplotlib.pyplot as plt

def create_averaged_responses_dict(ordering_array, responses):
    """
    [create_averaged_responses_dict] creates a dictionary where the key is the 
    order of the image and the value is the average response of the image.

    For example, the first loop create the dict:    
        [order_to_responses = {..., 625: [response_1, response_2, response_3], ...}]
    The second loop averages the responses to get the average measured response for each order:
        [order_to_responses = {..., 625: [np.mean(response_1 + response_2 + response_3)], ...}]
    where each response is a 241-dim vector.
    """
    order_to_responses = {}

    # each order should have three responses
    # create a dictionary where key is the order and value is the list of responses
    # For example, order_to_responses = {625: [response1, response2, response3], ...}
    # where each response is a 241-dim vector.

    for i in range(len(ordering_array)):
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

def create_predicted_responses_dict(ordering_array, responses):
    """
    [create_predicted_responses_dict] creates a dictionary where the key is the
    order of the image and the value is the predicted response of the image.

    Since the predicted response is the same for each image, we can just take the
    first response and use that as the predicted response.
    For example, order_to_responses = {..., 624: [response_1], 625: [response_1], ...}
    """
    order_to_responses = {}

    for i in range(len(ordering_array)):
        key = ordering_array[i]
        if key not in order_to_responses:
            order_to_responses[key] = responses[i]
    
    return order_to_responses

def plot_validation_response(model, matched_clip_vectors, activation, train_mask, ordering_array):
    """
    [plot_validation_response] plots the average predicted response against the 
    average measured response for each image in the validation set.
    """

    predicted_response = model.predict(matched_clip_vectors[~train_mask])
    measured_response = activation[~train_mask]
    # shape of predicted response (3000, 241)
    # shape of measured response (3000, 241)

    # validation set has 3000 trials
    ordering_array = ordering_array[~train_mask]

    avg_predicted_response_dict = create_predicted_responses_dict(ordering_array, predicted_response)
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
    x = np.linspace(-3, 3, 100)
    y = x
    plt.plot(x, y, '-r', label='y=x')

    plt.show()