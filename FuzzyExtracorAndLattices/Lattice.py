"""
This file contains methods and functions to generate a (low density) Latin Square (lsq) Matrix for use in LDLC.
Encodings of the matrix are as a pair (permutationMatris, signMatrix) which we can use to
    easily store or build a lqs. 
These encoding can store an 'n by n' redular lsq of weight d into two 'd by n' matrices. 
Written by Renaud Brien
"""

import numpy as np
import os, sys, glob

class LatticeLSQ:
    """A Class to contain all the methods related to generating Latin Square Lattices for
        use as Low Density Lattice Codes parity-check matrices"""
    
    valSeq3 = [1, 0.57735, 0.57735]
    
    def __init__():
        pass
    
    def gen_lsq_value_sequence_1(self,d):
        seq = []
        seq.append(1)
        for i in range(d-1):
            seq.append( (d**(-0.5)) )
        return seq
    
    """ n is the size of the matrix, d in the number of non-zero elements in each row/column"""
    def generate_latin_square_encoding (self, n, d, seed=None):
        
        int_seed = None
        if (seed == None):
            int_seed = np.random.seed()
        else:
            int_seed = seed
        
        output = []
        
        generator = np.random.RandomState(int_seed)         #Set a random generator, with seed
        
        permutationMat = np.empty(shape=(d, n))             #Generate a placeholder matrix
    
        for i in range(d):                                  #Filling the matrix up with permutations
            permutationMat[i] = generator.permutation(n)
        
        #Removing Loops from matrix...
        c = 0
        loopless_columns = 0
        
        while (loopless_columns < n):
            changed_perm = -1
            twoLoops = False
            fourLoops = False
            #searching for twoLoops
            for i in range (d):
                for j in range(i+1, d):
                    if ( permutationMat[i,c] == permutationMat[j,c] ):
                        #2-loop was found
                        changed_perm = i
                        twoLoops = True
                        break
                break
            #If no 2-loops are found, search for 4-loops
            if (twoLoops == False):
                for c0 in range(n):
                    if (c0 != c):
                        col = permutationMat[:,c0]
                        comparedCols = np.in1d(col.ravel(), permutationMat[:,c]).reshape(col.shape)
                        #print(comparedCols)
                        if (np.count_nonzero(comparedCols) > 1):
                            changed_perm = np.where(comparedCols)[0][0]
                            break
            if (changed_perm != -1) :
                index = c
                while (index == c):
                    index = generator.randint(0,n)
                temp = permutationMat[changed_perm, c]
                permutationMat[changed_perm, c] = permutationMat[changed_perm, index]
                permutationMat[changed_perm, index] = temp
                loopless_columns = 0
            else :
                #No loop found at column c
                loopless_columns = loopless_columns + 1
            c = (c + 1) % n
            #print(permutationMat)
        
        output.append(permutationMat)
        
        #Generating Sign Matrix
        signMat = np.empty(shape=(d, n))
        for i in range(d):
            singMat[i] = generator.randint(2, size=n)*2 - 1
        
        output.append(signMat)
        
        return output
        
    
    
    


def generate_lsq_value_sequence(d):
    seq = []
    seq.append(1)
    for i in range(d-1):
        seq.append( (d**(-0.5)) )
    return seq
    
def generate_lsq_value_sequence_ones(d):
    seq = []
    for i in range (d):
        seq.append( 1 )
    return seq

def generate_latin_square_encoding (n, d, int_seed):
    
    output = []
    
    #d = len(valueArray)
    
    generator = np.random.RandomState(int_seed)         #Set a random generator, with seed
    
    permutationMat = np.empty(shape=(d, n))             #Generate a placeholder matrix

    for i in range(d):                                  #Filling the matrix up with permutations
        permutationMat[i] = generator.permutation(n)
    
    #Removing Loops from matrix...
    c = 0
    loopless_columns = 0
    
    while (loopless_columns < n):
        changed_perm = -1
        twoLoops = False
        fourLoops = False
        #searching for twoLoops
        for i in range (d):
            for j in range(i+1, d):
                if ( permutationMat[i,c] == permutationMat[j,c] ):
                    #2-loop was found
                    changed_perm = i
                    twoLoops = True
                    break
            break
        #If no 2-loops are found, search for 4-loops
        if (twoLoops == False):
            col = permutationMat[:,c]
            for c0 in range(n):
                if (c0 != c):
                    #col = permutationMat[:,c0]
                    comparedCols = np.in1d(col, permutationMat[:,c0])#.reshape(col.shape)
                    #print(comparedCols)
                    if (np.count_nonzero(comparedCols) > 1):
                        changed_perm = np.where(comparedCols)[0][0]
                        break
        if (changed_perm != -1) :
            index = c
            while (index == c):
                index = generator.randint(0,n)
            temp = permutationMat[changed_perm, c]
            permutationMat[changed_perm, c] = permutationMat[changed_perm, index]
            permutationMat[changed_perm, index] = temp
            loopless_columns = 0
        else :
            #No loop found at column c
            loopless_columns = loopless_columns + 1
        c = (c + 1) % n
        #print(permutationMat)
    
    output.append(permutationMat)
    
    #Generating Sign Matrix
    signMat = vec = generator.randint(2, size=n)*2 - 1
    for i in range(1, d):
        vec = generator.randint(2, size=n)*2 - 1
        signMat = np.vstack((signMat, vec))
    
    output.append(signMat)
    
    return output
    
def gen_lsq_from_encoding ( n, lsq_encoding, valueArray ) :
    
    latSquare = np.zeros((n, n))
    permutationMat = lsq_encoding[0]
    signMat = lsq_encoding[1]
    d = len(valueArray)
    
    for i in range(n):
        for j in range(d):
            rowVal = permutationMat[j,i]
            latSquare[rowVal ,i] = valueArray[j] * signMat[j,i]
    
    return latSquare
    
def gen_lsq_from_seed (n, seed, valueArray ):
    m = generate_latin_square_encoding(n, len(valueArray), seed)
    return gen_lsq_from_encoding(n, m, valueArray)
    
def gen_lsq_file_encoding_ex(n, lsq_encoding, valueArray):
    d = len(valueArray)
    print(n, d)   #dimension, weight
    print(valueArray)           #values
    
    #add index where each value of valueArray is found
    #first get index of values in each row:
    perm = lsq_encoding[0]
    sign = lsq_encoding[1]
    indexInRow =[]
    for i in range(perm.shape[0]):
        indexInRow.append(invertPermutation(perm[i]))
    
    #print each row
    for i in range(n):
        val = []
        for j in range(d):
            k = int(indexInRow[j][i])
            val.append( sign[j][k] * k )
        print(val)
    
    #print each col:
    for i in range(n):
        val = []
        for j in range(d):
            val.append( sign[j][i] * int(perm[j][i]) )
        print(val)
    

def invertPermutation(array):
    #print(array)
    x = np.zeros(len(array))
    for i in range(len(array)):
        k = int(array[i])
        x[k] = i
    return x
    





