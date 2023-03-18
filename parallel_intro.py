# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
import time
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def initialize():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    client = Client()    #use ipyparallel module to send instructions to controller via Client object. want initialize client object in this func
    client.ids           #tells you how many engines you have running: have 8 engines
    
    #use client object to create DirectView class:
    dview = client[:]    #group all engines into a DirectView: want create DirectView w/ all available engines
    
    dview.execute("import scipy.sparse as sparse")  #import scipy.sparse as sparse on all engines
    
    client.close()       #need include after EVERY function
    
    return dview
   
# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    client = Client()      #create client object
    dview = client[:]      #create DirectView
    dview.block = True     #set directview to use blocking
    
    #distribute the variables: need to set up the dict like they have in lab manual
    dview.push(dx)         #initialize variables from dict on all engines
    
    #check value of each variable on all engines:
    #inside this for loop want to pull variables back and make sure they havent changed: raise error if they have
    for key in dx.keys():  #loop through all the keys to loop through all engines
        pull = dview.pull(key)        #check value of the key
        count = pull.count(dx[key])   #count how often that value occurs (dx[key] will map to the variable/value have)
        assert count == len(pull), "THE VARIABLES HAVE CHANGED, OH NO!" #check if variables have changed by checking if the length has changed
        
    client.close()

# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    #need to create the dview class to use in the prob: 
    client = Client()     
    dview = client[:]   #dview is made to include all the engines
    dview.block = True    
    
    #instruct each engine make n draws from standard normal distrib:
    dview.execute("import numpy as np") 
    dview.execute("draws = np.random.normal(size= " + str(n) + ")")  #execute method runs commands on engines, this will run it on all of the engines
    
    #want mean, min, max draws:
    dview.execute("means = np.mean(draws)") #these execute commands will apply to all of the engines:
    dview.execute("mins = np.min(draws)")
    dview.execute("maxs = np.max(draws)")
    
    #now actually get the list of mean, min, max values that we want by calling it from the dview class thingy:
    means = dview["means"]
    mins = dview["mins"]
    maxs = dview["maxs"]
    
    client.close()
    return means, mins, maxs
  
# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    #need to create the dview class to use in the prob: 
    client = Client()     
    dview = client[:]   
    dview.block = True 
    
    n_vals = [1000000, 5000000, 10000000, 15000000]
    parallel_times = []
    serial_times = []
    
    #get all parallel times 1st: get parallel time from prob 3 func
    for n in n_vals:  #loop through all n values and find time for each
        start = time.time()
        prob3(n)
        end = time.time()
        the_time = end-start
        parallel_times.append(the_time)
        
    #get the serial times:
    for n in n_vals:
        start = time.time() #want time how long it takes to run for ALL engines have so start the time out here and end it outside the inner for loop
        for i in range(len(client.ids)):  #want to loop through N times where N is num of engines have, and we know we have 8 engines
            draw = np.random.normal(size = n) #want run drawing function N times
            means = np.mean(draw)  #want record the statistics for the draw: so want the mean, min, max
            mins = np.min(draw)
            maxs = np.max(draw)
        end = time.time()
        da_time = end-start
        serial_times.append(da_time)
     
    #plot execution times against n:
    plt.plot(n_vals, parallel_times, label = "parallel", color = "mediumpurple")
    plt.plot(n_vals, serial_times, label = "serial", color = "orangered")
    plt.title("Computational Times for Parallel and Serial Processes")
    plt.legend(loc = "best")
    plt.show()
    
    client.close()
    
# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    #need to create the dview class to use in the prob: 
    client = Client()     
    dview = client[:]   
    dview.block = True 
    dview.execute("import numpy as np")  #will need numpy functions when sum over trap func so import it 
    
    #get h value which will need for trapezoidal function: h is the length btw x values
    h = (b-a)/(n-1)
    
    #get range of x values: this is evenly dividing the points
    x_vals = np.linspace(a, b, n)
    
    #map splits btw engines, executes function on given elements so make map w/ trapezoidal function to apply trapezoid function on all processors have at once like want to 
    #x_vals[:-1] gets only the range of x values have in range of a,b
    sum_it_up = np.sum(dview.map(lambda x: h*(f(x) + f(x+h))/2, x_vals[:-1])) #x_k+1 is same as x+h bc h is the amount btw each x have
    
    client.close()
    
    return sum_it_up
    
#test all probs:
if __name__ == '__main__':
    #test prob 1:
    #print("prob 1:")
    #print(initialize())
    #print("")
    
    #test prob 2:
    #dx = {"a":0, "b":2, "c":5}  #test w/ a rando dict
    #print("prob 2:")
    #print(variables(dx))
    #print("")
    
    #test prob 3:
    #print("prob 3:")
    #print(prob3())
    #print("")
    
    #test prob 4:
    #print("prob 4:")
    #prob4()
    #print("")
    
    #test prob 5:
    #print("prob 5:")
    #print(parallel_trapezoidal_rule(lambda x: x, 0, 1, n=200))
    
    pass
