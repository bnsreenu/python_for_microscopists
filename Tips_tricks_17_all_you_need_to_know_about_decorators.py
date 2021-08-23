# https://youtu.be/ZSMRgFQRSoU
"""
Author: Dr. Sreenivas Bhattiprolu 

A decorator in python allows us to add new functionality to an existing 
object (function or class) by not requiring us to modify the object's structure. 

Decorators allow us to wrap another function to extend the behavior of the 
wrapped function, without permanently modifying it. They are typically called 
before defining another function that we'd lke to decorate.

Functions are first class objects in python. This means they support 
the following operations. 

- Stored in a variable. 
- Passed as an argument to another function. 
- Defined inside another function.
- Returned from another function. 
- Store in data structures such as lists.

Decorators leverage this behavior of functions. 

"""

#Start by understanding the basics and the above topics...

#0. What is a function? Well, it returns values based on supplied arguments. 
# 
def fahrenheit2celcius(F):
    C = (F-32)/1.8
    
    return C

print(fahrenheit2celcius(90))

####################################################################
# 1. Storing the Function inside a variable
def fahrenheit2celcius(F):
    C = (F-32)/1.8
    
    return C

#Assign the function to a variable
F2C = fahrenheit2celcius
print(F2C(90))

####################################################################
#2 Passing a function as an argument to another function.
#Let us define our first function that converts F to C
def fahrenheit2celcius(F):
    C = (F-32)/1.8
    
    return C

#Let us define a function that calls another function, in our case fahrenheit2celcius
#Let us also provide temp as an argument. 
#This function return the input function and supplies temp as an argument to it. 
def temperature_conversion(some_function, temp=65):
    print("We are assigning F value to: ", temp)
    
    return some_function(temp)

t = 90
converted_temp = temperature_conversion(fahrenheit2celcius, temp=t)
print(t, " F, in celcius is: ", converted_temp)

####################################################################
#3 Defining a function inside another function

def fahrenheit2kelvin(F):
    def fahrenheit2celcius(F):
        C = (F-32)/1.8
        return C
    C = fahrenheit2celcius(F)
    K = C + 273
    return(K)

F2K = fahrenheit2kelvin(90)
print(t, " F, in Kelvin is: ", F2K)  

####################################################################
#4 Returning a function from another function

#Here, the top (primary) function returns the nested function. 
def return_fahrenheit2celcius_func():
    
    def fahrenheit2celcius(F):
        C = (F-32)/1.8
        return print(F, "F in Celcius is:", C,"C")
    
    return fahrenheit2celcius

#Now, we can retrieve the nested function by calling the main function.
#If we supply an argument we will get the nested function as an object.
F2C_function = return_fahrenheit2celcius_func() #Object

#Now we can use the retrieved nested function like a normal function 
#where we provide an argument. 
F2C_function(32)

#________________________________________________________
#Remember that the enclosed function can access the variables from the 
#primary enclosing function. Therefore, arguments can be supplied
#via the primary function. 

def return_fahrenheit2celcius_func(F):
    
    def fahrenheit2celcius():
        C = (F-32)/1.8
        return print(F, "F in Celcius is:", C,"C")
    
    return fahrenheit2celcius

F2C_function = return_fahrenheit2celcius_func(32) #Object with variable F defined as 32
F2C_function()
##########################################################################
"""
DECORATORS

Let us define a decorator that converts F to C
"""
###################################################################
#Decorator that will report the input argument of a function and also
#reports the execution time. 
#Defining a wrapper function with name 'wrapper' (name can be anything)
import time

def F2C_decorator(func):
    def wrapper(F):  #Argument will be passed to the function that we will be decorating
        print("________________________________")
        print("The input value supplied is: ", F)
        print("Above print statements from the decorator. Now entering the function... \n")
        start = time.time()
        func(F)
        end = time.time()
        print("\nExited the function. This and below from the decorator")
        print("________________________________")
        print("Calculation finished in ", end-start)
                
    return wrapper

@F2C_decorator
def fahrenheit2celcius(F):
    C = (F-32)/1.8
    time.sleep(1) #Adding sleep time so we see something in the execution time
    print(F, "F in Celcius is: ", C, "(From the function)")
    return C

fahrenheit2celcius(32)


#_________________________________________________________
#Another function but the same decorator. 
@F2C_decorator
def celcius2kelvin(C):
    K = C + 273
    time.sleep(1) #Adding sleep time so we see something in the execution time
    print(C, "C in Kelvin is: ", K, "(From the function)")
    return C

celcius2kelvin(32)
#___________________________________________________
#Another function but the same decorator. 
@F2C_decorator
def lower_to_upper_text(text):
    upper = text.upper()
    time.sleep(1) #Adding sleep time so we see something in the execution time
    print(upper)
    return upper

text="hello from inside the function"
lower_to_upper_text(text)

#######################################################################

#Decorator with arbitrary arguments
######################################################################

import time

def gen_purpose_decorator(func):
    def wrapper(*args,**kwargs):  #Argument will be passed to the function that we will be decorating
        print("\nThe input positional args supplied are: ", args)
        print("The input keyword args supplied are: ", kwargs ,"\n")
        
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        
        print("\nExited the function. This and below from the decorator")
        print("Calculation finished in ", end-start)
                
    return wrapper

@gen_purpose_decorator
def junk_func():
    time.sleep(1) #Adding sleep time so we see something in the execution time
    print("I'm an empty function with no args") 

junk_func()

#_________________________________________________________
#Function with arguments
@gen_purpose_decorator
def fahrenheit2celcius(F):
    C = (F-32)/1.8
    time.sleep(1) #Adding sleep time so we see something in the execution time
    print(F, "F in Celcius is: ", C, "(From the function)")
    return C

fahrenheit2celcius(90)

#Another function with multiple input args
@gen_purpose_decorator
def do_some_math(a,b,c):
    result = (a+b)*c
    time.sleep(1) #Adding sleep time so we see something in the execution time
    print("Result of (a+b)*c is", result, "(From the function)")
    
do_some_math(2,8,5)

#_________________________________________________________
#Function with parameter and keyword args
@gen_purpose_decorator
def f2c_with_name(F, name="John", age=45):
    C = (F-32)/1.8
    time.sleep(1) #Adding sleep time so we see something in the execution time
    if age > C:
        print(name, ", You are older than the current temperature, ", C, "C")
    else:
        print(name, ", You are younger than the current temperature, ", C, "C")
    

f2c_with_name(110, name="David", age=30)

#######################################################################
"""
What in the world is @wraps decorator?

@wraps updates the wrapper function to look like wrapped function 
by copying attributes like the doc string, function name, args list, etc. 

Can be very useful if you want to provide API to your code to others. 
"""
################################################################
def my_decorator(func):
    def wrapper(*args, **kwargs):
        """This is a wrapper function"""
        func()
    return wrapper

@my_decorator
def fahrenheit2celcius(F):
    """
    Parameters
    ----------
    F : Temp in fahrenheit.

    Returns
    -------
    C : Celcius.

    """
    C = (F-32)/1.8
    
    return C

@my_decorator
def fahrenheit2kelvin(F):
    """    
    Parameters
    ----------
    F : Temp. in Fahrenheit.

    Returns
    -------
    Temp. in Kelvin.

    """
    C = (F-32)/1.8
    K = C + 273
    return(K)

#Name and docstring for any function using the decorator will only report
#the details of the wrapper function but not the actual functions. 
print(fahrenheit2celcius.__name__)
print(fahrenheit2celcius.__doc__)  #Access the doc string using __doc__ or help
print(fahrenheit2kelvin.__name__)
print(fahrenheit2kelvin.__doc__)

#Same behavior when you type 'help'
help(fahrenheit2celcius)

#_______________________________________________________________
# Example of using @wraps to extract doc string from a function. 
from functools import wraps

def my_decorator(func):
    def wrapper(*args, **kwargs):
        """This is a wrapper function"""
        func()
    return wrapper

#Now let us use @wraps decorator
def my_decorator_with_wraps(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """This is a wrapper function"""
        func()
    return wrapper

@my_decorator_with_wraps
def fahrenheit2celcius(F):
    """
    Parameters
    ----------
    F : Temp in fahrenheit.

    Returns
    -------
    C : Celcius.

    """
    C = (F-32)/1.8
    
    return C

@my_decorator_with_wraps
def fahrenheit2kelvin(F):
    """    
    Parameters
    ----------
    F : Temp. in Fahrenheit.

    Returns
    -------
    Temp. in Kelvin.

    """
    C = (F-32)/1.8
    K = C + 273
    return(K)

#Name and docstring for any function using the decorator will only report
#the details of the wrapper function but not the actual functions. 
print(fahrenheit2celcius.__name__)
print(fahrenheit2celcius.__doc__)
print(fahrenheit2kelvin.__name__)
print(fahrenheit2kelvin.__doc__)

#Same behavior when you type 'help'
help(fahrenheit2celcius)
#############################################################

# Example of using @wraps to extract doc string of a class and its methods. 

def my_decorator_with_wraps(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """This is a wrapper function"""
        func()
    return wrapper

#Decorate the class with the wrapper. 
#@my_decorator
@my_decorator_with_wraps
class convert_temp:
    """
    This is a class that converts F to C or K. 
    
    Call the fahrenheit2celcius function for F to C
    Call the fahrenheit2kelvin function for F to K
    """
    
    def __init__(self):
        """
        Convert temp class is initialized.

        """
    
    def fahrenheit2celcius(self, F):
        """
        Function to convert Fahrenheit to Celcius
    
        Parameters
        ----------
        F : Temp in fahrenheit.
    
        Returns
        -------
        C : Celcius.
    
        """
        self.F = F
        C = (self.F - 32)/1.8
    
        return C

    def fahrenheit2kelvin(self, F):
        """    
        Function to convert Fahrenheit to Kelvin
        
        Parameters
        ----------
        F : Temp. in Fahrenheit.
    
        Returns
        -------
        Temp. in Kelvin.
    
        """
        self.F = F
        C = (self.F-32)/1.8
        K = C + 273
        return(K)


help(convert_temp)  #Access the class doc string
help(convert_temp.fahrenheit2celcius)  #Access the function doc string. 

###########################################################################
#######################################################################
from functools import wraps
def execution_time(func):
    @wraps(func)
    def wrapper(*args,**kwargs):  #Argument will be passed to the function that we will be decorating        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Function name is: ", func.__name__, 
              "\nTotal time for execution=", end-start)
        return result
                
    return wrapper

@execution_time
def do_some_math(a,b,c):
    result = (a+b)*c
    time.sleep(1) #Adding sleep time so we see something in the execution time
    print("Result of (a+b)*c is", result, "(From the function)")
    
do_some_math(2,8,5)

#__________________________________________________________

#############################################################################
"""
Practical use of a decorator
Save time by not rewriting code to find execution times. 
Let us deocrate model.fit with a wrapper that calculates execution time
and prints the function name. . 

Dataset:
https://www.kaggle.com/uciml/pima-indians-diabetes-database
"""
#########################################################################

from functools import wraps
def execution_time(func):
    @wraps(func)
    def wrapper(*args,**kwargs):  #Argument will be passed to the function that we will be decorating        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Function name is: ", func.__name__, 
              "\nTotal time for execution=", end-start)
        return result
                
    return wrapper


#We will use the above execution_time decorator.
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Define X and Y. 
#First 8 columns are inputs and the 9th is the label (0 or 1)
X = dataset[:,0:8]
y = dataset[:,8]

# Model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# fit the model without decorator.... 
model.fit(X, y, epochs=10, batch_size=16)


#fit model with the decorator to report execution time
@execution_time
def fit_model():
    model.fit(X, y, epochs=10, batch_size=16)
    return model

model = fit_model()

@execution_time
def predict_on_train_data():
    result = model.predict(X)
    return result
prediction =  predict_on_train_data()

