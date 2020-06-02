import numpy as np
import os, sys, glob
#from sklearn.externals import joblib
import joblib

"""
Contains The LSQ generation algorithm that returns a list of permutations to build a Latin Square.
Also contains methods to transform these lsqs to a representation that can be used for the 
    belief propagation algorithm of lattice codes.
As well as a save/load methods to allow the pre-generation of the lsq matrices, as the algorithm 
    to generate a lsq can be somewhat expensive to compute.
    
Save path follow the pattern:
default_path/d{weight}/{size}/seed{seedvalue}.joblib
"""

default_path = 'C:/Coding/PythonFuzzyOutputs/LsqLattices'

valSeq3 = [1, 0.57735, 0.57735]

valseq5 = [1, 0.4472135955, 0.4472135955, 0.4472135955, 0.4472135955]

class LatticeLSQ:
    """A Class to contain all the methods related to generating Latin Square Lattices for
        use as Low Density Lattice Codes parity-check matrices"""
    
    def __init__(self, n, d, seed=None, default_save_path=default_path):
        self.n = n
        self.d = d
        if seed==None:
            self.seed = np.random.seed()
        else:
            self.seed = seed
        
        file_dir = default_path+'/'+f'd{self.d}' + '/' + str(self.n)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        
        file_name = f'd{self.d}' + '/' + str(self.n)+'/'+f'seed_{self.seed}'                
        file = default_path+'/'+file_name+'.joblib'
        
        if os.path.exists(file):
            self.load_lsq(file_name)
        else:
            tuple = self.generate_latin_square_encoding(self.n, self.d, self.seed)
            self.permutations = tuple[0]
            self.signs = tuple[1]
            self.save_lsq(file_name)
        
        pass
    
    def gen_lsq_value_sequence_1(self,d):
        seq = []
        seq.append(1)
        for i in range(d-1):
            seq.append( (d**(-0.5)) )
        return seq
        
    def save_lsq(self, name):
        file = default_path+'/'+name+'.joblib'
        joblib.dump(self, file)
        
    def load_lsq(self, name):
        file = default_path+'/'+name+'.joblib'
        if os.path.exists(file):
            # Load it with joblib
            #if VERBOSE: print('Loading', file)
            gcbObj = joblib.load(file)
            self.n = gcbObj.n
            self.d = gcbObj.d
            self.seed = gcbObj.seed
            self.permutations = gcbObj.permutations
            self.signs = gcbObj.signs
            return True
        else:
            return False
    
    
    """ n is the size of the matrix, d in the number of non-zero elements in each row/column"""
    def generate_latin_square_encoding (self, n, d, seed=None):
        
        int_seed = None
        if (seed == None):
            int_seed = np.random.seed()
        else:
            int_seed = seed
        
        #output = []
        
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
        
        #output.append(permutationMat)
        
        #Generating Sign Matrix
        signMat = np.empty(shape=(d, n))
        for i in range(d):
            signMat[i] = generator.randint(2, size=n)*2 - 1
        
        #output.append(signMat)
        
        return (permutationMat, signMat)
    
    def generate_matrix(self, value_sequence=None):
        seq = None
        if value_sequence==None:
            seq = self.gen_lsq_value_sequence_1(self.d)
        elif len(value_sequence) != self.d:
            return False
        else:
            seq = value_sequence
        
        latSquare = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(self.d):
                rowVal = int(self.permutations[j,i])
                latSquare[rowVal ,i] = seq[j] * self.signs[j,i]
        
        return latSquare
        
    """
    This method return a triple (shape, A_row, A_col) that can be used in the belief_propagation algorithm.
    
    For Example:
    A_row = [ row1, row2, ..., row_n ]
    row[i] = [ index_1, index_2, ..., index_d ],
    with index_i = (j, k) such that M[i,j] = k
    
    A_col will be defined similarly.
    
    """
    def get_sparse_rep(self, value_sequence=None):
        seq = None
        if value_sequence==None:
            seq = self.gen_lsq_value_sequence_1(self.d)
        elif len(value_sequence) != self.d:
            return False
        else:
            seq = value_sequence
            
        shape = (self.n, self.n)
        
        A_col = []
        
        for i in range(self.n):
            col_i = []
            for j in range(self.d):
                col_i.append( (int(self.permutations[j,i]), self.signs[j,i]*seq[j]) )
            col_i.sort()
            A_col.append(col_i)
        
        A_row = []
        for i in range(self.n):
            A_row.append([])
           
        for i in range(self.n):
            for p in A_col[i]:
                A_row[p[0]].append( (i, p[1]))
                
        return (shape, A_row, A_col)
    
def generate_many_encoding_for_tests( n_list, d_list, seed_low, seed_high):
    for d in d_list:
        for n in n_list:
            print(f'd = {d}, n = {n}')
            for seed in range(seed_low, seed_high+1):
                print(seed, end=' ')
                new_lsq = LatticeLSQ(n, d, seed)
            print('')
    print('Done!')


def test_determinant( n, d, beta=1):
    lsq = LatticeLSQ(n, d, 0)
    
    scaled_seq = []
    for i in range(d):
        scaled_seq.append( valSeq3[i]/beta )
    
    h = lsq.generate_matrix( scaled_seq )
    g = np.linalg.inv(h)
    
    det_g = np.linalg.det(g)
    
    print( 'det(H) = ', np.linalg.det(h) )
    print( 'det(G) = ', det_g )
    
    sigma = ( det_g**(2.0/n) )/(2*math.e*math.pi)
    
    print( 'Var Bound = ', sigma )
    

def test_determinant_var( n, d, sigma2):
    lsq = LatticeLSQ(n, d, 0)
    
    #det_g_orig = np.linalg.det( np.linalg.inv( lsq.generate_matrix( valSeq3 ) ) )
    det_g_orig = np.linalg.det( np.linalg.inv( lsq.generate_matrix( valSeq5 ) ) )
    
    #scaling_factor = np.sqrt( 2*math.e*math.pi * sigma2 )
    scaling_factor = np.sqrt( sigma2 / 0.0585 ) / (det_g_orig**(1.0/n) )
    scaled_seq = []
    if (d == 3):
        valseq = valSeq3
    elif(d == 5):
        valseq = valSeq5
    for i in range(d):
        scaled_seq.append( valSeq[i] / scaling_factor )
    
    h = lsq.generate_matrix( scaled_seq )
    g = np.linalg.inv(h)
    
    det_g = np.linalg.det(g)
    
    print( 'det(H) = ', np.linalg.det(h) )
    print( 'det(G) = ', det_g )
    
    sigma = ( det_g**(2.0/n) )/(2*math.e*math.pi)
    
    print( 'Var Bound = ', sigma )
    

















