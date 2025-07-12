#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Function to print lower triangular pattern
def lower_triangular(n):
    print("Lower Triangular Pattern:")
    for i in range(n):
        print("*" * (i + 1))

# Function to print upper triangular pattern
def upper_triangular(n):
    print("\nUpper Triangular Pattern:")
    for i in range(n):
        print(" " * i + "*" * (n - i))

# Function to print pyramid pattern
def pyramid(n):
    print("\nPyramid Pattern:")
    for i in range(n):
        print(" " * (n - i - 1) + "*" * (2 * i + 1))

# Set number of rows
rows = 5

# Call functions to print patterns
lower_triangular(rows)
upper_triangular(rows)
pyramid(rows)


# In[ ]:




