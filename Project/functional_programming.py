
# coding: utf-8

# In[3]:


# only final data structures

# pythin doesn't support final, and the constatnt of python is declare as CAPITAL LETTER
PI = 3.1416


# In[7]:


# Python program to illustrate functions 
# Functions can return another function 
    
def create_adder(x): 
    def adder(y): 
        return x + y 
    
    return adder 
    
add_15 = create_adder(15) 
    
print(add_15(10))


# In[6]:


#annonymus / lamda function

x = lambda a, b: a * b
print(x(5, 6))

