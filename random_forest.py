"""
Random Forest Lab

Jane Emeline Slagle
Section deux
12/1/22
"""
import os
import numpy as np
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# import graphviz
# from uuid import uuid4

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
         
        #return true if sample's feature located at index column (so sample at that column index) is greater than or equal to value
        if sample[self.column] >= self.value:
            return True
        else:
            return False

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(self.value))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    
    #initialize left, right arrays:
    left = []
    right = []
    
    #now fill in the arrays:
    for row in data:   #want partition samples (rows) of data so loop through the rows
        if question.match(row):  #left array contains samples that match method returned as True
            left.append(row)
        else:                    #right array contains samples that match method returned as False
            right.append(row)
    
    #Jake from State Farm says to cast left, right as arrays before the return statement and before return them as None:
    left = np.array(left)
    right = np.array(right)
    
    #if left or right is empty return it as None:        
    if len(left) == 0:
        left = None
    if len(right) == 0:
        right = None
        
    return left, right

# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]

# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


#Problem 2
def gini(data):
    """Return the Gini impurity of given array of data.
    Parameters:
        data (ndarray): data to examine
    Returns:
        (float): Gini impurity of the data"""
       
    #gini impurity formula is given in def 1.1, for the formula we need N, K, N_k
    
    N = num_rows(data)   #get N: N is num_rows
    
    dicty_dict = class_counts(data)   #get K: number of keys in class_counts dict so first get the dict
   
    #now do the gini impurity formula
    sum_thing = 0        #get the sum in gini impurity formula
    for k in dicty_dict:  
        sum_thing += (dicty_dict[k]/N)**2  #N_k: these are the actual values of the dict
    
    gini_val = 1 - sum_thing  #now get the gini value
    
    return gini_val
    
def info_gain(left, right, G):
    """Return the info gain of a partition of data.
    Parameters:
        left (ndarray): left split of data
        right (ndarray): right split of data
        G (float): Gini impurity of unsplit data
    Returns:
        (float): info gain of the data"""
        
    #need to account for if they are None type objects or whatever:
    if left is None:
        len_left = 0
    else:
        len_left = len(left)
        
    if right is None:
        len_right = 0
    else:
        len_right = len(right)
        
    D = len_left + len_right  #D is num of samples (rows) in D
   
    #now do the info_gain equation given by def 1.2:
    return G - ((len_left / D) * gini(left)) - ((len_right / D) * gini(right))
    
# Problem 3, Problem 7
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 7
    Returns:
        (float): Best info gain
        (Question): Best question"""
    
    best_gain = 0           
    best_question = None
    gini_ting = gini(data)  #will need the gini index below
    n,m = data.shape
    
    cols = np.arange(m-1)   #choose the columns to go through: dont want to go through the last one
    
    #problem 7 part of the problem:
    if random_subset:       #if its true
        cols = np.random.choice(cols, int(np.sqrt(m-1)), replace = False)
    
    #want iterate through the cols and then the rows:
    for col in cols:   #loop through each row
        for val in range(n-1):  #then loop through each value in the row, dont look at last column
            quest = Question(col, data[val, col], feature_names)  #make a Question class object w/ column and value
            left, right = partition(data, quest)       #use partition() to split dataset into left, right partitions
            
            #check if either left or right parition has less samples than min_samples_leaf
            #but first have to check that it isnt None type
            if left is None or right is None:
                continue
            else:
                if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                    continue    #want discard this partition and iterate to next one
            
            #then calculate info_gain() of these 2 partitions w/ Gini impurity of the dataset:
            info_stuff = info_gain(left, right, gini_ting)
            
            #check if info_gain is greater than best_gain:
            if info_stuff > best_gain:
                #set this info_gain and its corresponding Question equal to best_gain, best_question:
                best_gain = info_stuff
                best_question = quest
    
    return best_gain, best_question
   
# Problem 4
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        if data is None: 
            pass
        else:
            self.prediction = class_counts(data)  #prediction attribute is same as what class_counts func returns

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        self.question = question
        self.left = left_branch
        self.right = right_branch

# Prolem 5
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    
    #build tree recursively how it tells us to in lab manual:
    if data.shape[0] < min_samples_leaf*2:
        return Leaf(data)   #want to return data as a Leaf
        
    else:
        opt_gain, opt_quest = find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False)
        
        if opt_gain == 0 or current_depth >= max_depth:
            return Leaf(data)
        
        else:  #if node isnt leaf case
            left, right = partition(data, opt_quest)  #split data into L, R partitions
            
            current_depth = 1
            
            #recursively define right, left branches:
            left_branch = build_tree(left, feature_names, current_depth = current_depth)
            right_branch = build_tree(right, feature_names, current_depth = current_depth)
            
            #now return Decision_Node ojbect using optimal question found earlier and the L, R branches of tree:
            return Decision_Node(opt_quest, left_branch, right_branch)
           
# Problem 6
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    
    #determine if the tree is a leaf:
    if isinstance(my_tree, Leaf):
       d_ting = my_tree.prediction 
       
       #want return label that corresp w/ most samples in the Leaf, lab manual says to do it this way
       return list(d_ting.keys())[np.argmax(d_ting.values())]
    
    #if tree is not a Leaf case:
    else:
        #recursively call the matching of the tree
        if my_tree.question.match(sample):  #if my_tree.question.match method is True w/ given sample
            return predict_tree(sample, my_tree.left)  #recursively call predict_tree() w/ my_tree.left
        else:
           return predict_tree(sample, my_tree.right)  #otherwise recursively call predict_tree() w/ my_tree.right
    
def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    
    #get the labels and the data
    data = dataset[:,:-1]   #labels are in the last column so only get the data
    labels = dataset[:,-1]  #get the labels from the dataset
    
    prop = 0  #parameter to store proportion of samples that the tree labels correctly
    
    #loop through each row in the dataset
    for i, row in enumerate(dataset):
        pred = predict_tree(row, my_tree)  #get predicted label using predict_tree()
        actual = labels[i]                 #get the tree label
        
        if pred == actual:   #now check if the predicted label is same as the actual label
            prop += 1        #if it is: add 1 to the proportion parameter of samples that tree labelled correctly
    
    return prop / len(data)  #return the actual proportion now

# Problem 7
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    predictions = []  #initialize list to store all of the predicted labels in
    
    #iterate through each tree
    for tree in forest:
        predictions.append(predict_tree(sample, tree))  #find label assigned to the sample by calling predict_tree()
        
    #return label predicted by majority of the trees
    return max(set(predictions), key = predictions.count)  #this snazzy line will do just that, booYUH baby!
    
def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    
    proportion = 0  #initialize parameter that will use to get accuracy want return here
    
    #loop through each row in the dataset, add a counter if it is the same
    for row in dataset: 
        if predict_forest(row[:-1], forest) == row[-1]:  #account for the last column or whatever since those are the labels
            proportion += 1   #add one to counter for when find proportionality!
    
    return proportion / len(dataset)  #return accruacy of forest's predictions

# Problem 8
def prob8():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    
    #load in the parkinsons dataset:
    parkin = np.loadtxt('parkinsons.csv', delimiter = ',')[:,1:]
    parkin_vals = parkin[:,:-1]
    labels = parkin[:,-1]
    features = np.loadtxt('parkinsons_features.csv', delimiter = ',', dtype = str, comments = None)
    
    #get the random training and testing data:
    randos = random.sample(range(0, len(parkin)), 130)  #randomly select 130 samples
    train = randos[:100]                               #use 100 in training your forest
    test = randos[100:]                                #use 30 more in testing it
    
    #train the data:
    train_me = parkin[train]
    test_me = parkin[test]
    train_labels = labels[train]
    test_labels = labels[test]
    
    #time it and get the accuracy to return in the tuple for MY very own implementation of my forest:
    my_forest = []
    start_my_forest = time.time()
    
    for _ in range(10):
        my_forest.append(build_tree(train_me, features, current_depth =0, random_subset = True))
    
    accur_1 = analyze_forest(test_me, my_forest)  #need get accuracy
    
    end_my_forest = time.time()
    time_my_forest = end_my_forest - start_my_forest
    
    first_tuple = (accur_1, time_my_forest)       #first tuple want return: accuracy, time of your implementation
    
    #time it and get accuracy for sklearn method:
    rfc = RandomForestClassifier(n_estimators = 5, min_samples_leaf = 15, max_depth = 4)
    
    start_sk_time = time.time()
    
    rfc.fit(train_me, train_labels)
    
    accur_2 = rfc.score(test_me, test_labels)
    
    end_sk_time = time.time()
    
    time_sk = end_sk_time - start_sk_time
    
    second_tuple = (accur_2, time_sk)  #second tuple want return
    
    #use scikit-learn with whole parkinsons dataset using default parameters:
    rfc = RandomForestClassifier(n_estimators = 5, min_samples_leaf = 15, max_depth = 4)
    parkin_train, parkin_test, labels_train, labels_test = train_test_split(parkin, labels, test_size = 0.2)  #use 80, 20 test train split
    
    start_sci_time = time.time()
    
    rfc.fit(parkin_train, labels_train)
    
    accur_3 = rfc.score(parkin_test, labels_test)
    
    end_sci_time = time.time()
    
    time_sk = end_sci_time - start_sci_time
    
    third_tuple = (accur_3, time_sk)  #get 3rd tuple to return
    
    #return the 3 tuples now:
    return first_tuple, second_tuple, third_tuple

## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree):
    """Draws a tree"""
    #Remove the files if they already exist
    for file in ['Digraph.gv','Digraph.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    
    
if __name__ == '__main__':
    #test prob 1:
    animals = np.loadtxt('animals.csv', delimiter = ',')  #load in the data
    features = np.loadtxt('animal_features.csv', delimiter = ',', dtype = str, comments = None)
    names = np.loadtxt('animal_names.csv', delimiter = ',', dtype = str)
    
    """#initialize question and test partition function
    question = Question(column=1, value=3, feature_names=features)
    left, right = partition(animals, question)
    print(len(left), len(right))
    
    question = Question(column=1, value=75, feature_names=features)
    left, right = partition(animals, question)
    print(left, len(right))"""
    
    #test prob 2:
    """print(gini(animals))
    print(info_gain(animals[:50], animals[50:], gini(animals)))"""
    
    #test prob 3:
    #print(find_best_split(animals, features))
    
    #test prob 6:
    """score = 0
    for _ in range(25):  #test single tree 25 times:
        party_rock = np.random.shuffle(animals)       #shuffle data
        my_tree = build_tree(animals[:80], features)  #train tree
        score += analyze_tree(animals[80:], my_tree)  #test tree
    
    print(score / 25)"""
    
    #test prob 7:
    #test functions on animals.csv:
    """np.random.shuffle(animals)
    my_forest = []
    for _ in range(10):
        my_forest.append(build_tree(animals[80:], features, current_depth = 0, random_subset = True))
    
    print(analyze_forest(animals[80:], my_forest))
    
    #compare results to the non-randomized versions:
    avg = 0
    for _ in range(10):
        avg += analyze_forest(animals[80:], my_forest)
    print(avg)"""
    
    #test prob 8:
    #print(prob8())
    
    pass

