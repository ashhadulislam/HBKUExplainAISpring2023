### put all necessary imports here
import numpy as ...

def calculate_probabilities(list_labels, uniq_labels):
    '''
    this function calculates the probabilities of each label in the list of labels
    it is calculated by number of labels in class A/all labels
    number of labels in class B/all labels
    and so on
    '''



def calc_entropy_from_probabilities(list_probas)
    '''
    Author: Sara Nassar 
    list_probas is the list of probabiities
    the formula for entropy is
    sum(-proba*log(proba))
    '''


def information_gain(old_entropy,new_entropies,count_items):
    '''
    from the list of new entropies, calculate the overall new entropy
    
    formula is something like:
    overall_new_entropy = entropy1*proportion1 + entropy2*proportion2+ entropy3*proportion3 ...
    
    igain=old_entropy-overall_new_entropy
    '''


def initialize_weights(number_features):
    '''
    the first set of weights corresponding to the features
    For now, it defaults to 2
    '''
    
    
def get_entropy_from_groups(new_entropies,count_items):
    '''
    put some comment here
    '''


def get_entropy(threshold,res,y_test):
    '''
    put some comment here
    '''

################################################
############## PSO related functions ###########
################################################
def objective_fn(param1,param2,X,y):
    '''
    param1 and param2 are the parameters that we want to optimize
    say param1 is the weight vector and  param2 is the threshold
    '''


def objective_fn_vector(params1,params2,X,y):
    '''
    params1 is an array of weight vectors
    params2 is an array of thresholds
    '''


    