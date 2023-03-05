### put all necessary imports here
import numpy as ...

def calculate_probabilities(list_labels, uniq_labels):
    
    probabilities = {}
    total = len(list_labels)
    for label in uniq_labels:
        count = list_labels.count(label)
        probability = count / total
        probabilities[label] = probability
    return probabilities



def calc_entropy_from_probabilities(list_probas)
  entropy_value = 0
    for proba in list_probas:
        if proba > 0:
            entropy_value += -proba * np.log(proba)
    return entropy_value
    return list_proba


def information_gain(old_entropy,new_entropies,count_items):
    
    overall_entropy = 0 
    numberOfItems = sum (count_items) # will be 10 (4+6)
    for i in range(len(new_entropies)):
        ratios = count_items[i]/numberOfItems
        overall_entropy += new_entropies[i] * ratios
        
    
    
    igain = old_entropy - overall_entropy    
    return igain


def initialize_weights(number_features):
  
    '''
    the first set of weights corresponding to the features
    defaults to 1
    '''
    
    weights=np.array([2 for i in range(number_features)])
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
    


    