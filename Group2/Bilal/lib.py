### put all necessary imports here
import numpy as np


def calculate_probabilities(list_labels, uniq_labels):
    '''
    Author: Sara Nassar 
    this function calculates the probabilities of each label in the list of labels
    it is calculated by number of labels in class A/all labels
    number of labels in class B/all labels
    and so on
    '''
    
    # A dictionary to store the probabilities
    probabilities = dict.fromkeys(uniq_labels, 0)
    
    # Total number of labels
    total_labels = len(list_labels)
    
    for label in uniq_labels:
        # Counting the number of times the label occurs in the list
        count = list_labels.count(label)
        
        # Calculating the probability of the label
        probability = count / total_labels
        
        # Storing the calculated probability in the dictionary
        probabilities[label] = probability
        
    return probabilities   



def calc_entropy_from_probabilities(list_probas):
    '''
    Author: Sara Nassar 
    list_probas is the list of probabiities
    the formula for entropy is
    sum(-proba*log(proba))
    
    '''
    
    entropy_value = 0

    for proba in list_probas:
        # If the probability is not zero
        if proba != 0:
            entropy_value += -proba * np.log(proba)
     
    return entropy_value


def information_gain(old_entropy,new_entropies,count_items):
    '''
    Author: Sara Nassar 
    from the list of new entropies, calculate the overall new entropy
    
    formula is something like:
    overall_new_entropy = entropy1*proportion1 + entropy2*proportion2+ entropy3*proportion3 ...
    
    igain=old_entropy-overall_new_entropy
    '''
    
    overall_new_entropy = 0
    
    # Calculating the total number of items
    total_items = sum(count_items)
    
    for i in range(len(new_entropies)):
        # Calculating the proportion of items in the current partition
        proportion = count_items[i] / total_items
        
        # Adding the entropy of the current partition weighted by its proportion to the overall new entropy
        overall_new_entropy += new_entropies[i] * proportion
        
    # Calculating the information gain
    information_gain = old_entropy - overall_new_entropy
    
    return information_gain


def initialize_weights(number_features):
    '''
    the first set of weights corresponding to the features
    For now, it defaults to 2
    '''
    
    weights=np.array([np.random.uniform() for i in range(number_features)])
    return weights
    
    
def get_entropy_from_groups(new_entropies,count_items):
    overall_new_entropy = 0
    
    # Calculating the total number of items
    total_items = sum(count_items)
    
    for i in range(len(new_entropies)):
        # Calculating the proportion of items in the current partition
        proportion = count_items[i] / total_items
        
        # Adding the entropy of the current partition weighted by its proportion to the overall new entropy
        overall_new_entropy += new_entropies[i] * proportion
        
    return overall_new_entropy    

def get_entropy(threshold,res,y_test):

    # make two groups
    group1=[]
    group2=[]

    for i in range(res.shape[0]):
        if res[i]<threshold:
            group1.append(y_test[i])
        else:
            group2.append(y_test[i])




    proba_gr1=calculate_probabilities(group1,np.unique(group1).tolist())
    proba_gr1=list(proba_gr1.values()) 
    entropy_group1=calc_entropy_from_probabilities(proba_gr1)
    count_group1=len(proba_gr1)

    proba_gr2=calculate_probabilities(group2,np.unique(group2).tolist())
    proba_gr2=list(proba_gr2.values()) 
    entropy_group2=calc_entropy_from_probabilities(proba_gr2)
    count_group2=len(proba_gr2)

    new_entropies=[entropy_group1,entropy_group2]
    count_items=[count_group1,count_group2]
    overall_new_entropy=get_entropy_from_groups(new_entropies,count_items)
    return overall_new_entropy

################################################
############## PSO related functions ###########
################################################
def objective_fn(param1,param2,X,y):
    '''
    param1 and param2 are the parameters that we want to optimize
    say param1 is the weight vector and  param2 is the threshold
    '''
    # multiply the weights with each feature and calculate the sum
    res=np.sum(X * param1, axis=1)  
#     print(res)
    #calculate entropy: hint: use the get_entropy function
    entropy=get_entropy(param2,res,y)
    return entropy
    
    
def objective_fn_vector(params1,params2,X,y):
    '''
    params1 is an array of weight vectors
    params2 is an array of thresholds
    '''
    results=[]
    for i in range(params1.shape[0]):
        param1=params1[i]#get ith set of weights
        param2=params2[i]# get ith threshold
        # call the objective_fn above to get the entropy
        res=objective_fn(param1,param2,X,y)
#         print(param2,res)
        results.append(res)
    
    return np.array(results)

    