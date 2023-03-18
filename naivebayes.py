import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        return

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        #want create new dataframe with 2 rows, one row is spam and other is ham and have a column for each vocab word
        words = {}                        #initialize dict
        X = X.str.replace('[^\w\s]', '')  #remove punctuation from the words (this is X data)
        
        #X is array of strings, want split strings into each word so that can add it to dict
        for i, message in enumerate(X.str.split()):
            for word in message:   #go through each word
                if word in words:
                #check if word already in dict, if is then increment the number associated with it (like if spam or ham) by 1
                    if y.iloc[i] == "spam":   
                        words[word][0] += 1   
                    else:
                        words[word][1] += 1
                else:  
                #if words isnt already in dict then add it to dict
                #in the dict: keys are words, values are a list: like tuple with number of times word appears in spam, then number of 
                #times word appears in ham
                    if y.iloc[i] == "spam":
                        val = [1,0]
                    else:
                        val = [0,1]
                    words[word] = val
                    
        #then convert dict to pandas dataframe, jake says its easy peasy, want save the dataframe as self.data:
        self.data = pd.DataFrame(words)
        
        self.data.index = ["spam", "ham"]    #want index of dataframe to be spam and ham
        
        #get prob of message being spam or ham:
        self.num_spam = np.sum(y == "spam")
        self.num_ham = np.sum(y == "ham")
        self.prob_spam = self.num_spam/len(y)
        self.prob_ham = self.num_ham/len(y)
        
    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        #get probabilities in equation 11.2: 
        
        #initialize the probs for both spam, ham to be 1:
        spam_ham_prob = {message:{x:1. for x in ["spam","ham"]} for message in X} 
        
        #then go through each word and change the probability to be the actual prob of whether each message is spam or ham:
        for message in X:
            unique, wordCount = np.unique(message.split(), return_counts = True)
            for i, word in enumerate(unique):
                #only want check the words that we trained with and have data for so have the next if statement:
                if word in self.data.columns:
                    #in here do the part inside the product of 7.2 (probs are products), do it for both spam and ham
                    spam_ham_prob[message]["spam"] *= ((self.data.loc["spam", word] / self.data.sum(axis=1)[0])**wordCount[i])
                    spam_ham_prob[message]["ham"] *= ((self.data.loc["ham", word] / self.data.sum(axis = 1)[1])**wordCount[i])
        
        #want return array so convert the dict of the probs into an array
        probs = np.zeros([len(X), 2])  #init the array to be filled with 0s
        
        #then loop through the probabilities found above and add them into the array just initialized:
        for i in range(len(spam_ham_prob.values())):
            probs[i,0] = (list(spam_ham_prob.values())[i]["ham"])*self.prob_ham
            probs[i,1] = (list(spam_ham_prob.values())[i]["spam"])*self.prob_spam
            
        return np.array(probs) 

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        labels = []  #init the labels as an empty list
        probs = self.predict_proba(X)  #call prob 2 to get the probs
        
        #finish equation 11.2:
        #then loop through the probs and check if prob of spam is greater than prob of ham to get the correct label
        #have 2 columns here: 1st col is spam, 2nd col is ham
        for message in range(len(X)):
            if probs[message, 0] >= probs[message, 1]:
                labels.append("ham")
            else:
                labels.append("spam")
                
        return np.array(labels) 

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        words = X.str.split().explode().unique()  #get all of the unique words
        
        #init both probs to be 0 bc adding now instead of taking a product so can have 0 in this case
        spam_ham_prob = {message:{"spam":0, "ham":0} for message in X}  #this is the P(C=k) in equation 11.6
        
        #equation 11.7:
        for message in X:
            unique, wordCount = np.unique(message.split(), return_counts = True)
            for i, word in enumerate(unique):
                #only want check the words that we trained with and have data for so have the next if statement:
                if word in self.data.columns:
                    #in here do the part inside the product of 7.2 (probs are products), do it for both spam and ham
                    spam_ham_prob[message]["spam"] += np.log(((self.data.loc["spam", word]+1) / (self.data.sum(axis=1)[0]+2))**wordCount[i])
                    spam_ham_prob[message]["ham"] += np.log(((self.data.loc["ham", word]+1) / (self.data.sum(axis = 1)[1]+2))**wordCount[i])
                    
        #want return array so convert the dict of the probs into an array
        probs = np.zeros([len(X), 2])  #init the array to be filled with 0s
        
        #then loop through the probabilities found above and add them into the array just initialized:
        for i in range(len(spam_ham_prob.values())):
            probs[i,0] = (list(spam_ham_prob.values())[i]["ham"])+np.log(self.prob_ham)
            probs[i,1] = (list(spam_ham_prob.values())[i]["spam"])+np.log(self.prob_spam)
            
        return np.array(probs) 

    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        labels = []  #init the labels as an empty list
        probs = self.predict_log_proba(X)  #call prob 2 to get the probs
        
        probs[:,0] += np.log(self.prob_ham)
        probs[:,1] += np.log(self.prob_spam)
        
        for message in range(len(X)):
            if probs[message, 0] >= probs[message, 1]:
                labels.append("ham")
            else:
                labels.append("spam")
                
        return np.array(labels) 

class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables
    '''

    def __init__(self):
        return

    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels

        Returns:
            self: this is an optional method to train
        '''
        words = {}                        #initialize dict
        unique_words = X.str.split().explode().unique()
        X = X.str.replace('[^\w\s]', '')  #remove punctuation from the words (this is X data)
        
        #X is array of strings, want split strings into each word so that can add it to dict
        for i, message in enumerate(X.str.split()):
            for word in message:   #go through each word
                if word in words:
                #check if word already in dict, if is then increment the number associated with it (like if spam or ham) by 1
                    if y.iloc[i] == "spam":   
                        words[word][0] += 1   
                    else:
                        words[word][1] += 1
                else:  
                #if words isnt already in dict then add it to dict
                #in the dict: keys are words, values are a list: like tuple with number of times word appears in spam, then number of 
                #times word appears in ham
                    if y.iloc[i] == "spam":
                        val = [1,0]
                    else:
                        val = [0,1]
                    words[word] = val
                    
        #then convert dict to pandas dataframe, jake says its easy peasy, want save the dataframe as self.data:
        self.data = pd.DataFrame(words)
        
        self.data.index = ["spam", "ham"]    #want index of dataframe to be spam and ham
        
        #want store computed rates in dicts where the key is the word and the value is the associated r:
        #so init them as dicts
        self.spam_rates = dict()
        self.ham_rates = dict()
        
        #now do the stupid r equation 11.11 stuff:
        for word in unique_words:  #loop through the words
            self.ham_rates[word] = (words[word][1] + 1) / (self.data.sum(axis=1)[1] + 2)     #1 is ham
            self.spam_rates[word] = (words[word][0] + 1) / (self.data.sum(axis = 1)[0] + 2)  #0 is spam
            
        self.num_spam = np.sum(y == "spam")
        self.num_ham = np.sum(y == "ham")
        self.prob_spam = self.num_spam/len(y)
        self.prob_ham = self.num_ham/len(y)
        
    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam
                column 0 is ham, column 1 is spam
        '''
        #init both probs to be 0 bc adding now instead of taking a product so can have 0 in this case
        spam_ham_prob = {message:{"spam": np.log(self.prob_spam), "ham": np.log(self.prob_ham)} for message in X}  #P(C=k) in equation 11.6
        
        #calc the prob of each message being spam or ham, this is based on the words in the message:
        for message in X:
            n = len([x for x in message.split()])
            unique, wordCount = np.unique(message.split(), return_counts = True)
            
            for i, word in enumerate(unique):
                if word in self.data.columns and word in message:
                    #only want check the words that we trained with and have data for so have the next if statement:
                    r_ham = self.ham_rates[word]
                    r_spam = self.spam_rates[word]
                    ni = wordCount[i]
                    spam_ham_prob[message]["spam"] += (np.log(stats.poisson.pmf(ni, r_spam*n)))
                    spam_ham_prob[message]["ham"] += (np.log(stats.poisson.pmf(ni, r_ham*n)))
                                 
        #want return array so convert the dict of the probs into an array
        probs = np.zeros([len(X), 2])  #init the array to be filled with 0s
        
        #then loop through the probabilities found above and add them into the array just initialized:
        for i in range(len(spam_ham_prob.values())):
            probs[i,0] = list(spam_ham_prob.values())[i]["ham"]  
            probs[i,1] = list(spam_ham_prob.values())[i]["spam"]
        
        return np.array(probs) 
        
    def predict(self, X):
        '''
        Use self.predict_log_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        labels = []  #init the labels as an empty list
        probs = self.predict_log_proba(X) 
        
        for message in range(len(X)):
            if probs[message, 0] >= probs[message, 1]:
                labels.append("ham")
            else:
                labels.append("spam")
                
        return np.array(labels) 

def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''

    #i am going crazy, i cant take this anymore
    
    #get the fit training data like how it says to in lab manual:
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)
    
    #now can use transformed training data to fit a MultinomialNB model from sklearn.naive_bayes like it says to in lab manual:
    clf = MultinomialNB()
    clf = clf.fit(train_counts, y_train)
    
    #now classify the data using the predict() method of the MultinomailNB model like how it says to in lab manual:
    test_counts = vectorizer.transform(X_test)
    labels = clf.predict(test_counts)  #this is the classification of X_test that want to return
    
    return labels

if __name__ == '__main__':
    df = pd.read_csv('sms_spam_collection.csv')  #load in the sms dataset
        
    #separate data into messages and labels:
    X = df.Message   #training data. contains strings that are SMS messages
    y = df.Label     #training labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.7)
    
    NB = NaiveBayesFilter()
    NB.fit(X[:300], y[:300])
    #test prob 1:
    #print(NB.data.loc['ham','in'])
    #print(NB.data.loc['spam','in'])
    
    #test prob 2:
    #print(NB.predict_proba(X[530:535]))
    
    #test prob 3:
    #print(NB.predict(X[530:535]))
    
    #test prob 4:
    #print(NB.predict_log_proba(X[530:535]))
    #print(NB.predict_log(X[530:535]))
    
    #test prob 5:
    PB = PoissonBayesFilter()
    #PB.fit(X[:300], y[:300])
    #print(PB.ham_rates['in'])
    #print(PB.spam_rates['in'])
    
    #test prob 6:
    #print(PB.predict_log_proba(X[530:535]))
    #print(PB.predict(X[530:535]))
    
    #test prob 7:
    actual_labels = sklearn_method(X[:300], y[:300], X[530:535])
    
    NB.fit(X[:300], y[:300])
    NB_labels = NB.predict_log(X[530:535])
    #print(accuracy_score(actual_labels, NB_labels))
    
    PB.fit(X[:300], y[:300])
    PB_labels = PB.predict(X[530:535])
    #print(accuracy_score(actual_labels, PB_labels))
    
    pass
