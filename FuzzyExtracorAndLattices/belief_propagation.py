"""
Code that does belief propagation for LDLC. Sometime called "sum-product" algorightm.
Written by Renaud Brien
"""

import numpy as np
import math
import Lattice
import matrix_representation
import LatticeCodeLSQ as lclsq
from LatticeCodeLSQ import LatticeLSQ

"""
Matrices/Graphs will be represented as a list of list:
The matrices are sparse, and the non-zero values are +- w, with w in the list 'weight'
A_row = [ row1, row2, ..., row_n ]
row[i] = [ index_1, index_2, ..., index_d ],
   with index_i = (j, k) such that M[i,j] = k
   
A_col will be defined similarly.

"""

#Example Matrix:
filename_example = "test_matrix_belief.txt"
#val_seq2 = [1, .25, .0625]
#test_matrix = Lattice.gen_lsq_from_seed(10, 0, val_seq2)

#test_tuple = matrix_representation.sparse_rep(test_matrix)
#matrix_representation.save_mat_tuple_to_file( test_tuple, filename_example )

#to load:
tuple_mat = matrix_representation.read_rep_from_file( filename_example )


"""
Belief Propagation has two main steps: The Horizontal and Vertical Steps.
Horizontal is the "additive" step
Vertical is the "multiplicative" step

"""

class GaussianMixturesReduction:
    
    """
    Note: A Gaussian Mixture is a function 
        f(z) = Sum_i ( c_i N(z; mean, var) ), with Sum_i (c_i) = 1 and c_i >= 0
    That is, a (normalized) sum of Gaussians.
    
    A Gaussian will be represented as a tripple : (mean, variance, 1)
    A Mixture will be represented as a list of tripples : (t_1, ..., t_n),
        with t_i = (mean_i, variance_i, c_i).
        
    Note: if sum (c_i) != 1, we need to normalize it by setting "c_i = c_i/sum(c_j)",
        otherwise, we aren't dealing with a pdf anymore.
    
    Mixture Reduction (GMR) is done to approximate a mixture with a shorter mixture, 
    all while minimizing the KL Divergence.
    """
    
    #Moment Matching to reduce a Gaussian mixture of 2 Gaussian down to one. Keeps the scaling
    def mm_pair(self, t_1, t_2):
        
        c = math.fsum( [t_1[2], t_2[2] ] )
        
        m1 = t_1[0]
        v1 = t_1[1]
        c1 = t_1[2]/c
        
        m2 = t_2[0]
        v2 = t_2[1]
        c2 = t_2[2]/c
        
        m = math.fsum( [c1*m1, c2*m2] )
        
        v =  math.fsum( [c1*(v1 + m1**2), c2*(v2 + m2**2), -(m**2)] )
        
        return (m, v, c)
        
    def mm(self, list):
        if len(list)<2:
            return list[0]
        else:
            c = 0
            m = 0
            v_temp = 0
            
            for t in list:
                #print(t)
                c += t[2]
                m += t[2]*t[0]
                v_temp += t[2]*(t[1] + t[0]**2)
                
            m = m / c
            v_temp = v_temp / c
            
            v = v_temp - m**2
            
            return (m, v, c)
    
    def gql(self, t_1, t_2):
        
        c = t_1[2] + t_2[2]
        
        m1 = t_1[0]
        v1 = t_1[1]
        c1 = t_1[2]/c
        
        m2 = t_2[0]
        v2 = t_2[1]
        c2 = t_2[2]/c
        
        m = c1*m1 + c2*m2
        
        v = c1*(v1 + m1**2) + c2*(v2 + m2**2) - (m**2)
        
        #print( m, v)
        
        res = math.fsum( [1 / (2 * math.sqrt( math.pi * v) ),
                          (c1**2) / (2 * math.sqrt( math.pi * v1) ),
                          (c2**2) / (2 * math.sqrt( math.pi * v2) ),
                          -(( 2 * c1) / math.sqrt( 2 * math.pi * (v + v1)) ) * math.exp( -((m - m1)**2) / (2 * (v + v1))),
                          -(( 2 * c2) / math.sqrt( 2 * math.pi * (v + v2)) ) * math.exp( -((m - m2)**2) / (2 * (v + v2))),
                          (( 2 * c1 * c2) / math.sqrt( 2 * math.pi * (v1 + v2)) ) * math.exp( -(m1 - m2)**2 / (2 * (v1 + v2)))])
        
        return res
    
    """
    the gmr method takes a gaussian mixture (as a list of triples), a threshold theta and a
    maximum number of Gaussian to return. 
    Theta is a upper bound on the pairwise 'gql' of the mixture to merge. This is to minimize 
    KL divergence between the reduced mixture. High theta will make the reduction to continue longer.
    """
    def gmr(self, mixture_list, theta, max_size):
        
        #print("gmr input list:")
        #print(mixture_list)
        
        current_list = mixture_list.copy()
        lsize = len(current_list)
        
        if lsize == 1:
            return current_list
        
        err = self.gql(current_list[0], current_list[1])
        pair = (current_list[0],current_list[1])
        
        #print(current_list)
        #print(pair)
        
        for i in range(lsize - 1):
            for j in range(i+1, lsize):
                t1 = current_list[i]
                t2 = current_list[j]
                
                c_err = self.gql(t1, t2)
                
                if c_err < err :
                    err = c_err
                    pair = (t1,t2)
        
        while ( c_err < theta ) | ( lsize > max_size ):
            
            #print( "c_err: ", c_err, "lsize: ", lsize )
            #print(current_list)
            #print(pair)
            
            if lsize > 1:
                    
                t3 = self.mm_pair(pair[0],pair[1])
                
                current_list.remove(pair[0])
                current_list.remove(pair[1])
                current_list.append(t3)
                
                lsize -= 1
                
                if lsize > 1:
                    err = self.gql(current_list[0], current_list[1])
                    pair = (current_list[0],current_list[1])
                    
                    for i in range(lsize - 1):
                        for j in range(i+1, lsize):
                            t1 = current_list[i]
                            t2 = current_list[j]
                            
                            c_err = self.gql(t1, t2)
                            
                            if c_err < err :
                                err = c_err
                                pair = (t1,t2)
                else:
                    t = current_list[0]
                    new_t = (t[0],t[1],1)
                    current_list.remove(t)
                    current_list.append(new_t)
                    return current_list
            else:
                t = current_list[0]
                new_t = (t[0],t[1],1)
                current_list.remove(t)
                current_list.append(new_t)
                return current_list            
            
        return current_list
    
#Ex for a det = 1, 10x10 Lattice : 
# sgd = SingleGaussianDecoder(tuple_mat, 0.05855)

class SingleGaussianDecoder:
    
    gmr = GaussianMixturesReduction()
    
    def __init__(self, mat_triple, channel_variance):
        self.h_dim = mat_triple[0]
        self.h_row = mat_triple[1]
        self.h_col = mat_triple[2]
        self.sigma2 = channel_variance
    
    def decode_3(self, received_message, max_iterations):
        
        var_to_check_messages = []
        check_to_var_messages = []
        
        row_num = self.h_dim[0]
        col_num = self.h_dim[1]
        
        #Initializing Messages to Check Nodes:
        for row in self.h_row:
            inc_node_message = []
            for i in row:
                index = i[0]
                mes = (received_message[index], self.sigma2)
                inc_node_message.append(mes)
            var_to_check_messages.append(inc_node_message)
        
        #print("Initialized messages")
        
        #Doing the Iteration Loop for max_iteration-1 times:
        for l in range(max_iterations - 1):
            
            #print("Loop ", l)
            
            #Check Node:
            check_message_out = []
            for i in range(col_num):
                check_message_out.append([])
            
            for i in range(len(self.h_row)):
                #print("Check node ", i)
                #print("     Input: ", var_to_check_messages[i])
                node_out = self.single_check_node(self.h_row[i], var_to_check_messages[i])
                
                #print("     Output: ", node_out)
                
                for j in range(len(self.h_row[i])):
                    col = self.h_row[i][j][0]
                    check_message_out[col].append(node_out[j])
            
            check_to_var_messages = check_message_out
            
            #Variable Node:
            var_message_out = []
            for i in range(row_num):
                var_message_out.append([])
            
            for i in range(len(self.h_col)):
                #print("Variable node ", i)
                #print("     Input: ", check_to_var_messages[i])
                node_out = self.single_var_node_3(received_message[i], self.h_col[i], check_to_var_messages[i])
                
                #print("     Output: ", node_out)
                
                for j in range(len(self.h_col[i])):
                    row = self.h_col[i][j][0]
                    var_message_out[row].append(node_out[j])
            
            var_to_check_message = var_message_out
        
        #End loop, final iteration:
        #Check Node:
        check_message_out = []
        for i in range(col_num):
            check_message_out.append([])
        
        for i in range(len(self.h_row)):
            node_out = self.single_check_node(self.h_row[i], var_to_check_messages[i])
            for j in range(len(self.h_row[i])):
                col = self.h_row[i][j][0]
                check_message_out[col].append(node_out[j])
        
        check_to_var_messages = check_message_out
        
        #Var Node:
        approximation = []
        
        for i in range(len(self.h_col)):
            node_out = self.single_var_node_3_final(received_message[i], self.h_col[i], check_to_var_messages[i])
            approximation.append(node_out)
            
        #Compute the decoded message:
        # First computing Hx, x being the approximation.
        decoded = []
        for row in self.h_row:
            b = 0
            for i in row:
                b += i[1]*approximation[i[0]]
            decoded.append(round(b))
        
        return decoded
    
    #Single Check Node step
    ''' Takes the Incoming messages as a list of pairs : ( mean, variance ) 
        Returns a list of pairs (mean, variance).
        
        The pairs are the messages coming from the nodes in the same order and index as hrow.
        '''
    #The outputs will be appended to the correct column inputs out of this function
    def single_check_node(self, hrow, incoming_message ):
        d = len(hrow)
        
        if (d==1):
            return incoming_message
        
        m_forward = []
        m_backward = []
        v_forward = []
        v_backward = []
        
        
        m_forward.append(hrow[0][1]*incoming_message[0][0])
        v_forward.append((hrow[0][1]**2)*incoming_message[0][1])
        
        m_backward.append(hrow[d-1][1]*incoming_message[d-1][0])
        v_backward.append((hrow[d-1][1]**2)*incoming_message[d-1][1])
        
        for i in range(1,d):
            
            m = m_forward[i-1] + hrow[i][1]*incoming_message[i][0]
            v = v_forward[i-1] + (hrow[i][1]**2)*incoming_message[i][1]
            
            m_forward.append(m)
            v_forward.append(v)
        
        
        for i in reversed(range(d-1)):
            
            m = m_backward[0] + hrow[i][1]*incoming_message[i][0]
            v = v_backward[0] + (hrow[i][1]**2)*incoming_message[i][1]
            
            m_backward.insert(0,m)
            v_backward.insert(0,v)
            
        output = []
        
        m1 = -1/hrow[0][1] * (m_backward[1])
        v1 = 1/(hrow[0][1]**2) * (v_backward[1])
        
        output.append( (m1,v1) )
        
        
        for i in range(1,d-1):
            
            m = -1/hrow[i][1] * (m_forward[i-1] + m_backward[i+1])
            v = 1/(hrow[i][1]**2) * (v_forward[i-1] + v_backward[i+1])
            
            output.append( (m,v) )
            
        md = -1/hrow[0][1] * (m_backward[1])
        vd = 1/(hrow[0][1]**2) * (v_backward[1])
        
        output.append( (md,vd) )
        
        return output

    #Single Variable Node step, 3 gaussian cyclic extension
    ''' Takes the Incoming messages as a list of pairs : ( mean, variance ) 
        Also takes the channel message y_i as the Gaussian pair (y_i, channel_variance)
        Returns a list of pairs (mean, variance).
        
        The pairs are the messages coming from the nodes in the same order and index as hcol.
        
        The periodic extension only takes the 3 periods closest to the channel message.
        '''
    #The outputs will be appended to the correct row inputs out of this function for future iterations
    def single_var_node_3(self, y_i, hcol, incoming_message):
        
        d = len(hcol)
        y = (y_i, self.sigma2)
        
        #Forward Recursion
        alpha = []
        alpha.append( [(y_i, 2*self.sigma2, 1)] )
        
        for i in range(d):
            m = incoming_message[i][0]
            v = incoming_message[i][1]
            h = hcol[i][1]
                        
            #periodic extension
            b = int(round(h*(m - y[0])))
            b_set = [ b-1, b, b+1 ]
            
            #print("Var Node with y_i = ", y_i, " h = ", h, " m = ", m, " and b = ", b)
            #print(" m + b/h = ", m - b/h )
            
            rho = []
            for j in b_set:
                rho.append( (m - j/h, v, 1) )
            
            alph_i = self.gaussian_mixture_product( alpha[i], rho )
            alpha.append( self.gmr.gmr(alph_i, 0.01, 1) )
        
        #Backward Recursion
        beta = []
        beta.append( [(y_i, 2*self.sigma2, 1)] )
        
        for i in reversed(range(d)):
            m = incoming_message[i][0]
            v = incoming_message[i][1]
            h = hcol[i][1]
            
            #periodic extension
            b = int(round(h*(m - y[0])))
            b_set = [ b-1, b, b+1 ]
            
            rho = []
            for j in b_set:
                rho.append( (m - j/h, v, 1) )
            
            beta_i = self.gaussian_mixture_product( beta[0], rho )
            beta.insert(0, self.gmr.gmr(alph_i, 0.01, 1) )
        
        #Output to node k is alpha[k]*beta[k+1]
        output = []
        for i in range(d):
            mixture = self.gaussian_mixture_product( alpha[i], beta[i+1] )
            pair = self.gmr.mm( mixture )
            
            #check for instabilities for the variance:
            m = pair[0]
            v = 0
            if pair[1] < 0.001:
                v = 0.001
            else:
                v = pair[1]
                
            output.append((m,v))
                
        return output
        
    #Single Variable Node step, 3 gaussian cyclic extension
    ''' Takes the Incoming messages as a list of pairs : ( mean, variance ) 
        Also takes the channel message y_i as the Gaussian pair (y_i, channel_variance)
        Returns a single approximation x-tilde_i.
        
        The pairs are the messages coming from the nodes in the same order and index as hcol.
        
        The periodic extension only takes the 3 periods closest to the channel message.
        '''
    #The outputs will be appended to the correct row inputs out of this function for future iterations
    def single_var_node_3_final(self, y_i, hcol, incoming_message):
        
        d = len(hcol)
        y = (y_i, self.sigma2)
        
        #Forward Recursion
        alpha = []
        alpha.append( [(y_i, 2*self.sigma2, 1)] )
        
        for i in range(d):
            m = incoming_message[i][0]
            v = incoming_message[i][1]
            h = hcol[i][1]
            
            #periodic extension
            b = int(round(h*(m - y[0])))
            b_set = [ b-1, b, b+1 ]
            
            rho = []
            for j in b_set:
                rho.append( (m - j/h, v, 1) )
            
            alph_i = self.gaussian_mixture_product( alpha[i], rho )
            alpha.append( self.gmr.gmr(alph_i, 0.01, 1) )
        
        #Final approximation is the mean of MM(alpha[d]*root(y))
        
        mixture = self.gaussian_mixture_product( alpha[d], [(y_i, 2*self.sigma2, 1)] )
        pair = self.gmr.mm( mixture )
        output = pair[0]
                
        return output


    #Function to compute the product of two Gaussian mixtures
    ''' The mixtures are provided as lists of triple (mean, var, weight)
    Suppose mix_1 has I elements, mix_2 has J elements
    
    Outputs a mixture as a list of I*J triples (mean, var, weight);
    The l-th element of the list, l = J(i-1)+j, is the product of the i-th element of mix_1
    and the j-th element of mix_2.
    '''
    def gaussian_mixture_product(self, mix_1, mix_2 ):
        
        #print( mix_1, mix_2 )
        
        total_c = 0
        prelim_res = []
        result = []
        
        for i in mix_1:
            for j in mix_2:
                prod = self.gaussian_product_pair(i,j)
                total_c += prod[2]
                prelim_res.append(prod)
                
        #print(prelim_res)
        
        if total_c != 1 :
            for t in prelim_res:
                c2 = t[2]/total_c
                t2 = ( t[0], t[1], c2 )
                result.append(t2)
        else:
            result = prelim_res
        
        #removing mixtures with a zero coefficient    
        final_result = []
        for triple in result:
            if triple[2] != 0:
                final_result.append(triple)
        
        return final_result

    #Function to compute the product of two gaussian mixture elements
    def gaussian_product_pair(self, triple_1, triple_2 ):
        
        m1 = triple_1[0]
        v1 = triple_1[1]
        c1 = triple_1[2]
        
        m2 = triple_2[0]
        v2 = triple_2[1]
        c2 = triple_2[2]
        
        #v = (v1**-1 + v2**-1)**-1
        v = v1*v2 / (v1 + v2)
        m = v * ( m1/v1 + m2/v2 )
        
        # Found the Correct equation
        c = ( c1*c2 / (math.sqrt(2*math.pi*(v1+v2))) ) * math.exp(- (m1 - m2)**2 /(2 * (v1+v2)) )
        
        return (m, v, c)

        





#Channel Noise Simulation:
def add_awgn_noise( codeword, sig2):
    
    generator = np.random
    transmit = []
    
    for i in range(len(codeword)):
        c = codeword[i] + generator.normal(0, np.sqrt(sig2))
        transmit.append(c)
    
    return transmit

def test_decoder_zero_2( number_to_test, iterations, size=20, sigma2=0.0585, beta=1):
    good = 0
    bad = 0
    correct_bit = 0
    decode_error = 0
    lsqSeed = np.random.randint(0,100)
    valSeq3 = [1, 0.57735, 0.57735]
    
    lsq = lclsq.LatticeLSQ(size, 3, lsqSeed)
    det_g_orig = abs(np.linalg.det( np.linalg.inv( lsq.generate_matrix( valSeq3 ) ) ) )
    
    scaling_factor = np.sqrt( beta*sigma2 / 0.0585 ) # / (det_g_orig**(1.0/size) )
    #scaling_factor = 1
    
    scaled_seq = []
    for i in range(3):
        scaled_seq.append( valSeq3[i] / scaling_factor )
    
    #Generating the Lattice Code matix H
    ldlc_mat_rep = lsq.get_sparse_rep(scaled_seq)
    
    #print( ldlc_mat_rep[0], '\n', ldlc_mat_rep[1], '\n', ldlc_mat_rep[2])
    
    #return 'Done'
    
    decoder = SingleGaussianDecoder(ldlc_mat_rep, sigma2)
    
    codeword = []
    for i in range(size):
        codeword.append(0)
    
    for i in range(number_to_test):
        y= add_awgn_noise( codeword, sigma2)
        b = decoder.decode_3(y, iterations)
        
        equal = True
        for j in range(size):
            if codeword[j] != b[j]:
                equal = False
                decode_error += 1
            else:
                correct_bit += 1

        if equal:
            good += 1
            
    return(good, correct_bit, decode_error)


#var_seq20 =  [4.62286, 1.314322, 1.57526, 0.29853, 2.80514, 0.50294, 0.4401, 0.74574, 0.4581,  0.5511, 6.3983, 0.44405, 1.90892, 0.43178, 1.42120, 0.32074, 1.99688, 4.79709, 0.55878, 0.84500]
def testing_decoder_variable_noise(n, seq_of_variance, number_to_test, iterations, beta=1):
    good = 0
    bad = 0
    
    sigma2 = seq_of_variance[0]
    for i in range(1,n):
        if sigma2 > seq_of_variance[i]:
            sigma2 = seq_of_variance[i]
    
    ratio = []
    for i in range(n):
        ratio.append(int(math.ceil( seq_of_variance[i]/sigma2 )))
    
    #Loading a LSQ for decoding
    lsqSeed = np.random.randint(0,100)
    lsq = lclsq.LatticeLSQ(n, 3, lsqSeed)
    
    det_g_orig = abs(np.linalg.det( np.linalg.inv( lsq.generate_matrix( valSeq3 ) ) ) )
    
    scaling_factor = np.sqrt( beta*sigma2 / 0.0585 ) / (det_g_orig**(1.0/n) )
    #scaling_factor = 1
    
    scaled_seq = []
    for i in range(3):
        scaled_seq.append( valSeq3[i] / scaling_factor )
    
    #Generating the Lattice Code matix H
    ldlc_mat_rep = lsq.get_sparse_rep(scaled_seq)
    
    decoder = SingleGaussianDecoder( ldlc_mat_rep, sigma2)
    
    codeword = np.zeros(n).tolist()
    
    print(f'Ratio is : {ratio}')
    
    for i in range(number_to_test):
        y = []
        for ind in range(n):
            y.append( codeword[ind] + np.random.normal(0, np.sqrt(seq_of_variance[ind])) )
        
        b = decoder.decode_3( y, iterations)
        
        equal = True
        for j in range(n):
            if codeword[j] != b[j]:
                equal = False
        if equal:
            good += 1
        print(b)
        
    return good
    
def test_decoder_zero_from_rep( number_to_test, iterations, representation, sigma2=0.0585):
    good = 0
    bad = 0
    #lsqSeed = np.random.randint(0,100)
    valSeq3 = [1, 0.57735, 0.57735]
    
    #lsq = lclsq.LatticeLSQ(size, 3, lsqSeed)
    #det_g_orig = abs(np.linalg.det( np.linalg.inv( lsq.generate_matrix( valSeq3 ) ) ) )
    
    #scaling_factor = np.sqrt( beta*sigma2 / 0.0585 ) / (det_g_orig**(1.0/size) )
    #scaling_factor = 1
    
    #scaled_seq = []
    #for i in range(3):
    #    scaled_seq.append( valSeq3[i] / scaling_factor )
    
    #Generating the Lattice Code matix H
    #ldlc_mat_rep = lsq.get_sparse_rep(scaled_seq)
    
    #print( ldlc_mat_rep[0], '\n', ldlc_mat_rep[1], '\n', ldlc_mat_rep[2])
    
    #return 'Done'
    
    decoder = SingleGaussianDecoder(representation, sigma2)
    size = rep[0][0]
    
    codeword = []
    for i in range(size):
        codeword.append(0)
    
    for i in range(number_to_test):
        y= add_awgn_noise( codeword, sigma2)
        b = decoder.decode_3(y, iterations)
        
        equal = True
        for j in range(size):
            if codeword[j] != b[j]:
                equal = False
        
        if equal:
            good += 1
        
    return(good)    
    
    
    