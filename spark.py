# solutions.py

import pyspark
from pyspark.sql import SparkSession
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MCE



# --------------------- Resilient Distributed Datasets --------------------- #

### Problem 1
def word_count(filename='huck_finn.txt'):
    """
    A function that counts the number of occurrences unique occurrences of each
    word. Sorts the words by count in descending order.
    Parameters:
        filename (str): filename or path to a text file
    Returns:
        word_counts (list): list of (word, count) pairs for the 20 most used words
    """ 
    #initialize SparkSession object:
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
    
    #load file in as PySpark RDD:
    huck_finn = spark.sparkContext.textFile("huck_finn.txt") 
    
    #count number of occurences of each word:
    words = huck_finn.flatMap(lambda row: row.split()) #do flat map so that each word is on one row (its own row, one word per row) so that can count num of words easily
                                                       #split gets each word by itself in a row
    words = words.map(lambda row: (row,1))             #mapping func applies a function to each row of our RDD, it makes it so we have a tuple of (word, 1)
                                                       #do 1 because then can start counting the word at 1, call it on words bc want to do this with the split words
    
    words = words.reduceByKey(lambda x, y:x + y)       #this looks @ RDD as a dict
                                                       #if finds 2 keys that are the same: it will add the values together and increase the count of the word by 1 so this actually gets the word count
    
    #sort words by descending order:
    answer = list(words.sortBy(lambda row: -1*row[1]).collect()[:20]) #sort words in descending order and get the 1st 20 words, convert it to list
   
    spark.stop()  #need to end the spark session
    
    return answer
    
### Problem 2
def monte_carlo(n=10**5, parts=6):
    """
    Runs a Monte Carlo simulation to estimate the value of pi.
    Parameters:
        n (int): number of sample points per partition
        parts (int): number of partitions
    Returns:
        pi_est (float): estimated value of pi
    """
    #first need to initialize SparkSession object:
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
            
    #create RDD w/ n*parts parameter amount of sample pts and partition it with parts parameter:
    #need to sample uniformly: np.random.random lets us sample from 0 to 1
    #multiplying it by 2 stretches the intervalto [0,2] so subtract off 1 to shift it to left 1 and be on interval [-1,1] like want to w/ uniform distrib.
    #the tuple w/ n*parts, 2 says have n*parts number of rows and the 2 creates a tuple like (x,y) pts
    samp_pts = spark.sparkContext.parallelize(2 * np.random.random((n*parts, 2)) - 1, parts)  #Jake Murphy from state farm expained this to me, thanks Jake!
    
    #use .filter and a lambda function to check if points are inside unit circle or not (their sum is less than or equal to radius, which is 1 here)
    samp_pts = samp_pts.filter(lambda samp_pt: samp_pt[0]**2 + samp_pt[1]**2 <= 1)  #sample_point[0] is x, sample_point[1] is y
    
    #calculate percentage of pts w/in circle
    perc_inside = samp_pts.count() / (n*parts)  #.count returns num of elements in the RDD
   
    spark.stop()  #need to end the spark session
    
    return perc_inside*4  #told in problem that multiplying perc of pts w/in circle by 4 gives us estimate for area of circle and thus estimate for pi
    
# ------------------------------- DataFrames ------------------------------- #

### Problem 3
def titanic_df(filename='titanic.csv'):
    """
    Calculates some statistics from the titanic data.
    
    Returns: the number of women on-board, the number of men on-board,
             the survival rate of women, 
             and the survival rate of men in that order.
    """
    #initialize spark object first:
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
            
    #load file into pyspark dataframe:
    titanic = spark.read.csv(filename)
    
    #find num of women on board:
    num_women = titanic.filter(titanic._c3 == "female").count() #column _c3 has sex of passengers so filter, count only by women 
    
    #find num of men on board:
    num_men = titanic.filter(titanic._c3 == "male").count()     #filter, count sex column by men
    
    #find survival rate of women:
    women_survived = titanic.filter(titanic._c0 == 1).filter(titanic._c3 == "female").count() #_c0 col says whether survivied or not: 1 means survived
    w_survivial_rate = women_survived / num_women              
    
    #find survival rate of men:
    men_survived = titanic.filter(titanic._c0 == 1).filter(titanic._c3 == "male").count()
    m_survivial_rate = men_survived / num_men 
    
    spark.stop() 
    
    #return each 4 values in order given as tuple of floats:
    return ((num_women, num_men, w_survivial_rate, m_survivial_rate))
    
### Problem 4
def crime_and_income(crimefile='london_crime_by_lsoa.csv',
                     incomefile='london_income_by_borough.csv', major_cat='Robbery'):
    """
    Explores crime by borough and income for the specified major_cat
    Parameters:
        crimefile (str): path to csv file containing crime dataset
        incomefile (str): path to csv file containing income dataset
        major_cat (str): major or general crime category to analyze
    returns:
        numpy array: borough names sorted by percent months with crime, descending
    """
    #create spark object:
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
            
    #load 2 files in as pyspark dataframes:
    crime_df = spark.read.csv(crimefile, header = True, inferSchema = True)
    income_df = spark.read.csv(incomefile, header = True, inferSchema = True)
   
    #only care about the crime specified in major_cat parameter so want to filter crime_df to only have that one crime in major_category column
    crime_df = crime_df.filter(crime_df.major_category == major_cat)
    
    #only want one row for each borough, but crime_df has multiple so need to use groupBy on it to fix this
    groups = crime_df.groupBy("borough").sum("value")   #want to groupby boroughs, but want to get the total number of major_cat crimes that happened in each borough
                                                        #value column counts how many of the crimes occured so sum over that column for each borough this gives us table
                                                        #with 1 row for each borough and a column with the total major_cat crimes for each 
    
    #now are able to join our 2 dataframes on the borough column (but be careful: want to join w/ groups now instead of crime_df)
    new_df = groups.join(income_df, on = "borough")     #joining on borough will make it so that have 1 row for each borough
    new_df = new_df.drop("mean-08-16")                  #get rid of this thing, no one wants it!
    
    #order by total num of crimes for major_cat, descending:
    new_df = new_df.orderBy("sum(value)", ascending = False)
    
    new_df = new_df.withColumnRenamed("sum(value)", "major_cat_total_crime") #rename the column name to more accurately represent what it is
    
    data_array = np.array(new_df.collect()) #convert dataframe to numpy array
    
    #create scatter plot of number of major_cat_crimes by median income for each borough
    plt.scatter(data_array[:,1].astype(float), data_array[:,2].astype(float), color = "deeppink")
    plt.xlabel("Number of major_cat Crimes", color = "mediumvioletred")
    plt.ylabel("Median Income for Each Borough", color = "mediumvioletred")
    plt.title("Number of major_cat Crimes By Median Income for Each Borough", color = "mediumvioletred")
    plt.show()
    
    spark.stop()
    
    return data_array
    
### Problem 5
def titanic_classifier(filename='titanic.csv'):
    """
    Implements a classifier model to predict who survived the Titanic.
    Parameters:
        filename (str): path to the dataset
    Returns:
        metrics (tuple): a tuple of metrics gauging the performance of the model
            ('accuracy', 'weightedRecall', 'weightedPrecision')
    """
    #create spark object:
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
            
    #load 2 files in as pyspark dataframes:
    schema = ('survived INT, pclass INT, name STRING, sex STRING, age FLOAT, sibsp INT, parch INT, fare FLOAT')
    titanic = spark.read.csv(filename, schema = schema)
    
    #use pyspark.ml package to train classifier: doing the same exact thing as the ex given above the problem but want to outperform LogisticRegression so use RFClass instead is the only change make
    
    #prepare data by converting the 'sex' column to binary categorical variable
    sex_binary = StringIndexer(inputCol='sex', outputCol='sex_binary')
    
    onehot = OneHotEncoder(inputCols=['pclass'], outputCols=['pclass_onehot'])   #one-hot-encode pclass (Spark automatically drops a column)
    
    features = ['sex_binary', 'pclass_onehot', 'age', 'sibsp', 'parch', 'fare']  #create single features column
    features_col = VectorAssembler(inputCols=features, outputCol='features')
    
    #now we create a transformation pipeline to apply the operations above
    #this is very similar to the pipeline ecosystem in sklearn
    pipeline = Pipeline(stages=[sex_binary, onehot, features_col])
    titanic = pipeline.fit(titanic).transform(titanic)
    
    titanic = titanic.drop('pclass', 'name', 'sex') #drop unnecessary columns for cleaner display (note the new columns)
    
    train, test = titanic.randomSplit([0.75, 0.25], seed=11) #split into train/test sets (75/25)
    
    #now do RandomForestClassification instead of LogisiticRegression:
    rf = RandomForestClassifier(labelCol='survived', featuresCol='features')
    
    #run a train-validation-split to fit best elastic net param
    #ParamGridBuilder constructs a grid of parameters to search over
    #have different hyperparameters for RF than for LR so this is another change that need to make too
    paramGrid = ParamGridBuilder()\
                    .addGrid(rf.maxBins, [5, 3, 12]).build()
                    
    #TrainValidationSplit will try all combinations and determine best model using the evaluator (see also CrossValidator)
    tvs = TrainValidationSplit(estimator=rf,
                               estimatorParamMaps=paramGrid,
                               evaluator=MCE(labelCol='survived'),
                               trainRatio=0.75,
                               seed=11)
    
    #train the classifier by fitting our tvs object to the training data
    clf = tvs.fit(train)
    
    #use the best fit model to evaluate the test data
    results = clf.bestModel.evaluate(test)
    
    accuracy = results.accuracy
    weightedRecall = results.weightedRecall
    weightedPrecision = results.weightedPrecision
    
    spark.stop()
    
    return ((accuracy, weightedRecall, weightedPrecision))
    
    #Jake Murphy pulled me through this lab. Big props to him
    
if __name__ == '__main__':
    
    #test prob 1:
    #print(word_count(filename='huck_finn.txt'))
    
    #test prob 2:
    #print(monte_carlo(n=10**5, parts=6))
    
    #test prob 3:
    #print(titanic_df(filename = 'titanic.csv'))
    
    #test prob 4:
    #print(crime_and_income(crimefile='london_crime_by_lsoa.csv', incomefile='london_income_by_borough.csv', major_cat='Robbery'))
                     
    #test prob 5:
    #print(titanic_classifier(filename='titanic.csv'))
    
    pass
