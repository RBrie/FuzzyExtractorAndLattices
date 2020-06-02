"""
Matrix Representation for LDLC algorithms
Written by Renaud Brien
"""

import numpy as np
from pathlib import Path

"""
This file is to provide the necessary representation of (sparse) matrices as lists of lists of tuples:
The matrix : 
A = [ 1, 2, 0, 3
    0, 4, 5, 6
    7, 0, 8, 9]
Will be represented as
A_row = [ [(0,1), (1,2), (3,3)], 
        [(1,4), (2,5), (3,6)],
        [(0,7), (2,8), (3,9)] ]

and/or
A_col = [ [(0,1), (2,7)],
        [(0,2), (1,4)],
        [(1,5), (2,8)],
        [(0,3), (1,6), (2,9)] ]
"""

default_path = "C:/Coding/LatticeOutput" #For saving the lattices

"""
To generate tuple (shape, A_row, A_col) : 
tuple_mat = sparse_rep(matrix)

to save:
save_mat_tuple_to_file( tuple_mat, filename):

to load:
tuple_mat = read_rep_from_file( filename )

"""

# Assume that matrix inputs are 2 dimentional numpy arrays.

##Making the Sparse representation
#Returns A_row as above
def get_sparse_rep_by_row(matrix):
    #shape = matrix.shape
    l_rep = []
    for i in range(matrix.shape[0]):
        l = []
        for j in range(matrix.shape[1]):
            if matrix[i,j] != 0 :
                t = (j, matrix[i,j])
                l.append(t)
        l_rep.append(l)
    #return (shape, l_rep)
    return l_rep

#Return A_col
def get_sparse_rep_by_col(matrix):
    #shape = matrix.shape
    l_rep = []
    for j in range(matrix.shape[1]):
        l = []
        for i in range(matrix.shape[0]):
            if matrix[i,j] != 0 :
                t = (i, matrix[i,j])
                l.append(t)
        l_rep.append(l)
    #return (shape, l_rep)
    return l_rep

#Build A_col from Shape and A_row.
def get_col_rep_from_row(shape, a_row):
    a_col = []
    for j in range(shape[1]):
        a_col.append([])
    for i in range(shape[0]):
        for p in a_row[i]:
            a_col[p[0]].append( (i, p[1]))
    return a_col
    

#Returns the tuple : ( matrix_shape, A_row )
def sparse_row_rep(matrix):
    shape = matrix.shape
    row_list = get_sparse_rep_by_row(matrix)
    return (shape, row_list)
    
#def sparse_col_rep(matrix):
#    shape = matrix.shapre
#    col_list = get_sparse_rep_by_col(matrix)
#    return (shape, col_list)

#Returns the tuple : (matrix_shape, A_row, A_col)
def sparse_rep(matrix):
    shape = matrix.shape
    row_list = get_sparse_rep_by_row(matrix)
    col_list = get_sparse_rep_by_col(matrix)
    return (shape, row_list, col_list)
    
def print_rep(representation):
    print(representation[0])
    for i in representation[1]:
        print(i)

#Returns a string containing matrix_shape, the elements of A_row, the elements of A_col
# separated by the new line character: '\n'    
def string_rep(representation):
    s = ""
    s += str(representation[0])
    for i in representation[1]:
        s += '\n' + str(i)
    if len(representation) > 2:
        for i in representation[2]:
            s += '\n' + str(i)
    return s

##Loading the representation from strings
#Take string representation and returns a tuple : 
#    ( matrix_shape, A_row)
def load_row_rep(rep):
    ls = rep.split('\n')
    shape = eval(ls[0])
    rows = shape[0]
    l = []
    for i in range(rows):
        l.append(eval(ls[i+1]))
    #mat = np.array(shape[0],shape[1])
    return (shape, l)
    
#Take a sting representation and returns the tuple:
#   ( matrix_shape, A_row, A_col), or (matrix_shape, A_row) if A_col cannot be found.
def load_rep(rep):
    ls = rep.split('\n')
    shape = eval(ls[0])
    rows = shape[0]
    cols = shape[1]
    lr = []
    lc = []
    for i in range(rows):
        lr.append(eval(ls[i+1]))
    if (len(ls) > rows + cols):
        for j in range(cols):
            lc.append(eval(ls[j+1+rows]))
    if len(lc) > 0 :
        return (shape, lr, lc)
    else:
        return (shape, lr)

def build_matrix_from_row_list(shape, a_row):
    mat = np.zeros(shape)
    for i in range(shape[0]):
        l = a_row[i]
        for j in l:
            mat[i, j[0]] = j[1]
    return mat
    
##Saving and Loading string to/from files

def save_rep_to_file( string_rep, filename, folder="C:/Coding/LatticeOutput"):
    datafolder = Path(folder)
    file_to_open = datafolder / filename
    try :
        with open(file_to_open, 'w') as text_file:
            print(string_rep, file=text_file)
        return True
    except:
        return False
        
def save_mat_tuple_to_file( tup, filename, folder="C:/Coding/LatticeOutput"):
    string = string_rep(tup)
    datafolder = Path(folder)
    file_to_open = datafolder / filename
    try :
        with open(file_to_open, 'w') as text_file:
            print(string, file=text_file)
        return True
    except:
        return False
        
def read_rep_from_file_string( filename, folder="C:/Coding/LatticeOutput"):
    datafolder = Path(folder)
    file_to_open = datafolder / filename
    try:
        with open(file_to_open, 'r') as text_file:
            s = text_file.read()
        return load_rep(s)
    except:
        return ()

def read_rep_from_file( filename, folder="C:/Coding/LatticeOutput"):
    datafolder = Path(folder)
    file_to_open = datafolder / filename
    try:
        with open(file_to_open, 'r') as text_file:
            shape = eval(text_file.readline())
            rows = shape[0]
            cols = shape[1]
            lr = []
            lc = []
            col_rep = True
            for i in range(rows):
                lr.append(eval(text_file.readline()))
            for j in range(cols):
                r = text_file.readline()
                if r != '':
                    lc.append(eval(r))
                else:
                    col_rep = False
            
            if col_rep :
                return (shape, lr, lc)
            else:
                return (shape, lr)
    except:
        return ()
        
    
    
    
    
    
    
