### put all necessary imports here
import numpy as np

def calculate_probabilities(list_labels, uniq_labels):
    label_counts = {}
    for label in list_labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    label_probabilities = {}
    total_labels = len(list_labels)
    for label in uniq_labels:
        count = label_counts.get(label, 0)
        label_probabilities[label] = count / total_labels
    
    return label_probabilities

###Example
'''list_labels = ['A', 'B', 'A', 'C', 'B', 'B', 'A']
uniq_labels = list(set(list_labels)) # get unique labels from list_labels
label_probabilities = calculate_probabilities(list_labels, uniq_labels)
print(label_probabilities)'''



def calc_entropy_from_probabilities(list_probas):
    entropy_value = 0
    for proba in list_probas:
        if proba > 0:
            entropy_value += -proba * np.log2(proba)
    return entropy_value


def information_gain(old_entropy,new_entropies,count_items):
    overall_new_entropy = 0
    total_items = sum(count_items)
    for i in range(len(new_entropies)):
        proportion = count_items[i] / total_items
        overall_new_entropy += new_entropies[i] * proportion
        
    igain = old_entropy - overall_new_entropy
    return igain


def initialize_weights(number_features):
    weights = np.array([2] * number_features)
    return weights
    
def get_entropy_from_groups(new_entropies, count_items):
    """
    Calculates the overall entropy based on the count of items in each group and their entropies.
    :param new_entropies: list of entropies for each group
    :param count_items: list of counts for each group
    :return: overall entropy
    """
    total_count = sum(count_items)
    overall_entropy = 0
    for i in range(len(new_entropies)):
        proportion = count_items[i] / total_count
        overall_entropy += new_entropies[i] * proportion
    return overall_entropy



def get_entropy(threshold, res, y_test):
    """
    Calculate the overall new entropy of a binary split based on a given threshold.

    Parameters:
    threshold (float): the threshold value for binary splitting
    res (numpy.ndarray): 1D numpy array containing predicted values
    y_test (numpy.ndarray): ground truth labels

    Returns:
    float: overall new entropy
    """

    # initialize two groups to split the y_test based on the threshold
    group1 = []
    group2 = []

    # iterate over all predicted values in res
    for i in range(res.shape[0]):
        # if the predicted value is less than the threshold, add the corresponding label to group1
        if res[i] < threshold:
            group1.append(y_test[i])
        # otherwise, add the corresponding label to group2
        else:
            group2.append(y_test[i])

    # calculate the probabilities and entropy for group1 using calculate_probabilities and calc_entropy_from_probabilities functions
    proba_gr1 = calculate_probabilities(group1, np.unique(group1).tolist())
    proba_gr1 = list(proba_gr1.values()) 
    entropy_group1 = calc_entropy_from_probabilities(proba_gr1)
    count_group1 = len(proba_gr1)

    # calculate the probabilities and entropy for group2 using calculate_probabilities and calc_entropy_from_probabilities functions
    proba_gr2 = calculate_probabilities(group2, np.unique(group2).tolist())
    proba_gr2 = list(proba_gr2.values()) 
    entropy_group2 = calc_entropy_from_probabilities(proba_gr2)
    count_group2 = len(proba_gr2)

    # calculate the overall new entropy using get_entropy_from_groups function
    new_entropies = [entropy_group1, entropy_group2]
    count_items = [count_group1, count_group2]
    overall_new_entropy = get_entropy_from_groups(new_entropies, count_items)

    return overall_new_entropy

################################################
############## PSO related functions ###########
################################################

def objective_fn(param1,param2,X,y):
    # multiply the weights with each feature and calculate the sum
    res=np.sum(X * param1, axis=1)  
#     print(res)
    #calculate entropy: hint: use the get_entropy function
    entropy= get_entropy(param2, res, y)
    # call the get_entropy function with the correct parameters.
    # you only need to pass the threshold, res vector and the y
    return entropy


def objective_fn_vector(params1,params2,X,y):
    results=[]
    for i in range(params1.shape[0]):
        param1 = params1[i]
        param2 = params2[i]
        # call the objective_fn above to get the entropy
        res=objective_fn(param1,param2,X,y)
#         print(param2,res)
        results.append(res)
    
    return np.array(results)


    