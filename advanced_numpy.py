# advanced_numpy.py
"""Python Essentials: Advanced NumPy.
Jane Slagle
Extra lab
9/5/22
"""
import numpy as np
from sympy import isprime
from matplotlib import pyplot as plt
import time

def prob1(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob1(A)
        array([0, 0, 3])
    """
    A_copy = np.copy(A)  #make copy of array A
    mask = A_copy < 0    #want change values of array that are neg so use mask to get all neg elements
    A_copy[mask] = 0     #apply mask to array, change neg values to be 0
    
    return A_copy        #return the changed copy

def prob2(arr_list):
    """return all arrays in arr_list as one 3-dimensional array
    where the arrays are padded with zeros appropriately."""
    max_x = 0  #var will store max x dim. will need for when pad 0s on
    max_y = 0  #var will store max y dim
    
    #have for loop bc have to squeeze each array individually
    for i, arr in enumerate(arr_list):   #want index num, actual element so use enumerate
        arr_list[i] = np.squeeze(arr)    #squeeze to get rid of extra dim stuff for each array 
        
        #figure out what x, y dim need to make each array into for when pad 0s on:
        if arr_list[i].shape[0] > max_y: #check y dim (rows) for each array
            max_x = arr_list[i].shape[0] #keep checking to find very max one that all arrays will have match
        if arr_list[i].shape[1] > max_x: #do same for x dim (columns)
            max_y = arr_list[i].shape[1]
    
    #pad zeros to make all arrays same dim:
    for j, arr in enumerate(arr_list):   
        if max_y != arr_list[0]:         #pad on rows of 0s
            zero_block = np.zeros((max_y - arr.shape[0], arr.shape[1])) #second var will keep same num of columns
            arr_list[j] = np.vstack((arr, zero_block)) #put extra rows of 0s on bottom
        if max_x != arr_list[1]:         #pad on cols of 0s
            zero_block = np.zeros((arr.shape[0], max_x - arr.shape[1]))
            arr_list[j] = np.hstack((arr, zero_block)) #put extra cols of 0s on right
    
    #now stack along 3rd dim using dstack:
    if len(arr_list) == 1: #special case in case array list only has 1 array in it
        return arr_list[0]
    else:
        final = np.dstack((arr_list[0], arr_list[1])) #stack only 1st 2 arrays together
    
    if len(arr_list) == 2:
        return final       #return only 1st 2 arrays stacked together if only 2 arrays in list
    else:                  
        for arr in arr_list[2:]:  #loop through all elements starting after 1st 2 arrays (remember arr are the elements here)
            final = np.dstack((final, arr)) #if list has more than 2 arrays in it: will keep adding the rest of arrays on to end
        return final
        
def prob3(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob3(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    return A / A.sum(axis = 1, keepdims = True) #divide each row of matrix by its row sum. axis = 1 gets you rows
                                                #use array broadcasting instead of loop
                                                #if dont use keepdims var then will divide row sums along columns, not rows how want
    
# this is provided for problem 4    
def LargestPrime(x,show_factorization=False):
    # account for edge cases.
    if x == 0 or x == 1:
        return np.nan
    
    # create needed variables
    forced_break = False
    prime_factors = [] # place to store factors of number
    factor_test_arr = np.arange(1,11)
    
    while True:
        # a factor is never more than half the number
        if np.min(factor_test_arr) > (x//2)+1:
            forced_break=True
            break
        if isprime(x):  # if the checked number is prime itself, stop
            prime_factors.append(x)
            break
        
        # check if anythin gin the factor_test_arr are factors
        div_arr = x/factor_test_arr
        factor_mask = div_arr-div_arr.astype(int) == 0
        divisors = factor_test_arr[factor_mask]
        if divisors.size > 0: # if divisors exist...
            if divisors[0] == 1 and divisors.size > 1:   # make sure not to select 1
                i = 1 
            elif divisors[0] == 1 and divisors.size == 1:  # if one is the only one don't pick it
                factor_test_arr=factor_test_arr+10
                continue
            else:   # othewise take the smallest divisor
                i = 0
            
            # if divisor was found divide number by it and 
            # repeat the process
            x = int(x/divisors[i])
            prime_factors.append(divisors[i])
            factor_test_arr = np.arange(1,11)
        else:  # if no number was found increase the test_arr 
               # and keep looking for factors
            factor_test_arr=factor_test_arr+10
            continue
    
    if show_factorization: # show entire factorization if desired
        print(prime_factors)
    if forced_break:  # if too many iterations break
        print(f"Something wrong, exceeded iteration threshold for value: {x}")
        return 0
    return max(prime_factors)

def prob4(arr,naive=False):
    """Return an array where every number is replaced be the largest prime
    in its factorization. Implement two methods. Switching between the two
    is determined by a bool.
    
    Example:
        >>> A = np.array([15, 41, 49, 1077])
        >>> prob4(A)
        array([5,41,7,359])
    """
    largest_primes = [] #list to store largest prime of each element in array for naive case
    
    if naive == False:  #use np.vectorize() method if naive boolean is false
        vectorized = np.vectorize(LargestPrime)    #vectorize function
        return vectorized(arr) 
        
    else:               #use naive for loop method if naive boolean is true
        for i in arr:   #loop through all elements in array (i is element)
            largest_primes.append(LargestPrime(i)) #use naive for loop on each element to find each element's largest prime
                                                   #add to the list where storing largest primes of each element
        return np.array(results)                   #turn list into array, return it
           
def prob5(x,y,z,A,optimize=False,split=True):
    """takes three vectors and a matrix and performs 
    (np.outer(x,y)*z.reshape(-1,1))@A on them using einsum."""
   
    #the problem broken down into each step:
    #x has shape i
    #y has shape j
    #z has shape i bc outer product of xy has shape ij and want broadcast z as column onto ij matrix so z needs have same shape as cols
    #A has shape jk bc need to multiply by matrix of shape ij and when multiply matrices: need same inner shape, but outer shape can be anything
    #outer_product = np.einsum("i, j -> ij", x, y)
    #array_broadcasted = np.einsum("ij, i -> ij", outer_produt, z)
    #mat_mul = np.einsum("ij, jk -> ik", array_broadcasted, A)
    
    #now combine all of the steps written out above into one:
    #start out with i, j and then need i for next step and then need jk for next step and want to end with shape ik
    return np.einsum("i,j,i,jk -> ik", x, y, z, A, optimize = optimize)
   
def naive5(x,y,z,A):
    """uses normal numpy functions to do what prob5 does"""
    return np.outer(x,y)*z.reshape(-1,1)@A  #given non-einsum equivalent code in problem

def prob6():
    """Times and creates plots that generate the difference in
    speeds between einsum and normal numpy functions
    """
    #time einsum func from prob 5 vs its numpy func equivalent for vecs size 3-500, arrays size (3,3)-(500,500)
    einsum_times = []  #list to store einsum times in
    opt_times = []     #einsum func takes longer to run than numpy so need to show it when optimized
    naive_times = []   #list to store times for numpy equivalent
    
    sizes = np.arange(3, 501) #range for the sizes want to time over
    
    for n in sizes:    #loop through all of the sizes want to time over
        x = np.random.random(n)       #make the x,y,z vectors and A matrix that will need to plug into the 2 funcs
        y = np.random.random(n)
        z = np.random.random(n)
        A = np.random.random((n,n))
        
        #time without optimizing:
        start1 = time.time() 
        prob5(x,y,z,A)
        end1 = time.time()
        
        #time with optimizing:
        start2 = time.time()
        prob5(x,y,z,A, optimize = True)
        end2 = time.time()
        
        #time the naive method:
        start3 = time.time()
        naive5(x,y,z,A)
        end3 = time.time()
        
        #add all of the times to their lists:
        einsum_times.append(end1-start1)
        opt_times.append(end2-start2)
        naive_times.append(end3-start3)
        
    #plot timing results:
    plt.title("Einsum vs. Numpy")
    
    #need subplots so that can plot with and without optimizing:
    plt.subplot(1,2,1)
    plt.plot(sizes, einsum_times, color = "mediumvioletred", label = "Einsum") #first plot without optimizing
    plt.plot(sizes, naive_times, color = "navy", label = "Numpy")
    
    plt.subplot(1,2,2)
    plt.plot(sizes, opt_times, color = "mediumvioletred", label = "Einsum Opt") #then plot w/ optimizing
    plt.plot(sizes, naive_times, color = "navy", label = "Numpy")
    
    plt.legend(loc = "best")
    plt.xlabel("Input size")
    plt.ylabel("Time")
    
    plt.show()
    
#extra problems that Dr. Jarvis added YAY:
def Jarvis_prob():
    def np_cosine(X):
        return np.cos(X)
   
    def naive_cosine(X):
        for i in range(n):
            X[i] = np.cos(X[i])
        return X
    
    def better_cosine(X):
        A = np.zeros(len(X)) #initialize the matrix as matrix of 0s
        for i in range(len(X)):
            A[i] = np.cos(X[i])  #fill in the matrix of 0s so that its not being replaced each time
   
    n = 10**6
    A = np.random.randn(n)
    
    #follow all of the steps that Dr. Jarvis outlined in the email:
    print(A[:5])             #print 1st 5 values of A
    start1 = time.time()     #time np_cosine func
    np_cosine(A)
    end1 = time.time()
    
    print(A[:5])             #print 1st 5 values of A again
    start2 = time.time()     #time naive_cosine func
    naive_cosine(A)
    end2 = time.time()
    
    start3 = time.time()     #time better_cosine func that wrote
    better_cosine(A)
    end3 = time.time()
    
    cos_time = end1-start1   #get all of the actual time results now
    naiv_time = end2-start2
    better_time = end3-start3
    
    print(cos_time, naiv_time)  #compare the 1st 2 times and give how much faster it is
    print("The first function is", naiv_time/cos_time, "times faster than the second function! WHOOP WHOOP")
    
    print(A[:5])             #print 1st 5 values of A yet again and say why its diff now
    print("The first 5 values of A are different now because when we do the naive method of cosine on A, it changes it. The naive function updates A in place, modifiying its values to be their cosine values")
    
    print("The better cosine function is", naiv_time/better_time, "times faster than the naive function! WHOAH")
    
    B = A.reshape(10**3,10**3)
    
    def max_row(X):          #write func max_row to iterate through B rows, return max of each row w/out numpy funcs
        maxs = []  #list to store the max of each row in
        for row in X:  #loop through rows of B
            maxs.append(max(row))
        return np.array(maxs) 
        
    start4 = time.time()     #time max_row func
    maxs1 = max_row(B)
    end4 = time.time()
    
    start5 = time.time()     #time np.max func
    maxs2 = np.max(B, axis=1)
    end5 = time.time()
    
    print("DO max_row and np.max have the same result??? LET'S SEE:", np.allclose(maxs1, maxs2)) #check to see if 2 give same results (they shld)
    max_row_time = end4-start4    
    np_max_time = end5-start5
    
    #give time difference:
    print("The max row function is", max_row_time/np_max_time, "times faster than np.max function. That's what I'm talkin about")
    print("FINE, I vow to never again use a for-loop if a NumPy function can be used instead")   #VOW IT!!!!
    
