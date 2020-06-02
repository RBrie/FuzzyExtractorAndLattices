"""
Code for a fuzzy extractor using lattices. And testing functions.
Written by Renaud Brien
"""

import numpy as np
import math
import os
import Lattice
import matrix_representation
import belief_propagation as bp
import LatticeCodeLSQ as lclsq
from LatticeCodeLSQ import LatticeLSQ
import Fuzzy_Extractor as FE

"""
Test script for loking at things
"""

training_faces_path = 'C:/Coding/ARFaces/ARFacesTrain'  #path to images for training PCA
testing_faces_path = 'C:/Coding/ARFaces/ARFacesTest'    #path to images for testin gextractor
output_directory = 'C:/Coding/PythonFuzzyOutputs'

valSeq3 = [1, 0.57735, 0.57735]
valSeq5 = [1, 0.4472135955, 0.4472135955, 0.4472135955, 0.4472135955]

VERBOSE = False

class UserDataLattice:
    
    def __init__(self):
        pass
        
    def new_user(self, training_images, nb_of_components, main_extractor, beta=1, noRP=False, lam = 1024 ):
        
        self.lsqSeed = np.random.randint(0,100)
        self.nb_of_components = nb_of_components
        self.beta = beta
        
        self.matSeed = np.random.seed()
        self.trSeed = np.random.seed()
        
        self.lam = lam
        #self.shiftSeed = np.random.seed()
        self.shiftSeed = 0
        
        """
        self.lsqSeed = 0
        self.matSeed = 0
        self.trSeed = 0
        #"""
        
        if noRP:
            self.extractor = main_extractor
        else:
            self.extractor = FE.FeRP_Ext( main_extractor, nb_of_components, 2500, self.matSeed, self.trSeed)
        
        features = self.extractor.extract_features(training_images)
        
        mean_vector = np.mean(features,0)
        var_vector = np.var(features,0)
        #print(f'nb comp : {self.nb_of_components} \nfeatures size : {mean_vector.shape}')
        min_var = np.amin(var_vector)
        max_var = np.amax(var_vector)
        
        #self.sigma2 = min_var
        self.sigma2 = max_var
        #Scaling value to make the LDLC work with the chosen variance
        scaling_factor = ( self.beta*self.sigma2 / 0.0585 )**(-0.5) 
        
        #"""
        scaled_seq = []
        for val in valSeq3:
            scaled_seq.append(val*scaling_factor)
        
        #Generating the Lattice Code matix H
        lsq = lclsq.LatticeLSQ(self.nb_of_components, 3, self.lsqSeed)
        self.ldlc_mat_rep = lsq.get_sparse_rep(scaled_seq)
        """
        scaled_seq = []
        for val in valSeq5:
            scaled_seq.append(val*scaling_factor)
        
        #Generating the Lattice Code matix H
        lsq = lclsq.LatticeLSQ(self.nb_of_components, 5, self.lsqSeed)
        self.ldlc_mat_rep = lsq.get_sparse_rep(scaled_seq)
        #"""
        
        #Building G for the LDLC
        H = matrix_representation.build_matrix_from_row_list(self.ldlc_mat_rep[0], self.ldlc_mat_rep[1])
        self.G = np.linalg.inv(H)
        
        '''
        print(f'sigma = {self.sigma2}')
        sigbound = ( (abs(np.linalg.det(self.G)))**(2.0/self.nb_of_components) )/(2*math.e*math.pi)
        print(f'Sigma bound = {sigbound}')
        '''
        
        #Building the Decoder
        self.decoder = bp.SingleGaussianDecoder(self.ldlc_mat_rep, self.sigma2)
        
        #Decoding the mean vector:
        mean_message = self.decoder.decode_3( mean_vector, 100 )
        
        mean_codeword = self.G.dot( np.array(mean_message) )
        
        #Computing Helper data : h = x_bar - Q(x_bar)
        self.helper_data = mean_vector - mean_codeword
        
        #FOR TESTING PURPOSES : The correct Codeword
        
        """
        #Randomization can be added here
        randomShift = []
        """
        self.correct_message = mean_message
        self.correct_codeword = mean_codeword 
        
    def authenticate( self, image ):
        
        features = self.extractor.extract_features([image])[0]
        #transmitted = features.tolist()
        #print(transmitted)
        #val_to_decore = features - self.helper_data
        message = self.decoder.decode_3( features - self.helper_data, 100 )
        
        return (message == self.correct_message)
        
    def authenticate_ROC( self, image ):
        
        lam = self.lam
        
        features = self.extractor.extract_features([image])[0] - self.helper_data
        
        #print("Features : ", features )
        
        generator = np.random.RandomState(self.shiftSeed)
        shift = []
        for i in range(len(features)):
            shift.append( generator.randint( -lam, lam ) )
        
        message = self.decoder.decode_3( features, 100 )
        
        message_difference = np.array(self.correct_message)-np.array(message)
        eucl_dist_to_codeword = np.linalg.norm(self.correct_codeword - features)
        
        max_score = 0
        
        for i in range(len(features)):
            dist = abs( (features[i] - self.correct_codeword[i] ) )
            score = dist / ((self.beta*self.sigma2)**0.5)
            
            #if(i == 0):
                #print(f'Code Dist = {self.user_codebook.steps[i]} \n ImageDist = {dist} \n Score = {score}')
                #print(f'Score = {score}')
            
            if (score > max_score):
                max_score = score
            
            #print(score, max_score)
            
        #print(' ')
        
        output_string = f'Decoded Message Difference is : \n{message_difference}\n'
        output_string += f'Distance to Codeword is : \n{eucl_dist_to_codeword}\n'
        output_string += f'Max Score (in amount of sigma) i : \n{max_score}\n'
        
        #print(output_string)
        return max_score, message_difference

def test_pca_auth(n=20, beta=1, test_indices=[5]):
    #nb_components_pca = 200
    #rp_comp = int(nb_components_pca/2)
    rp_comp = n
    nb_components_pca = int(2*n)
    #test_indices = [5]
    
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)
    
    #testname = "LatticeLsqResults_RPRP"
    testname = "LatticeLsqResults_PCARP"
    #testname = "LatticeLsqResults_PCA"
    #nb_components_pca = n
    
    testname += f'_d3_'
    #testname += f'_d5_'
    
    for index in test_indices:
        testname += str(index)
    
    #"""
    pca_name = 'PCA/pca' + str(nb_components_pca)
    pcaExt = FE.PCA_ext()
    
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = FE.directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, nb_components_pca)
        pcaExt.save_PCA(pca_name)
    """   
    #rpMmain
    pcaExt = FE.RP_ext(user_im[0][0][0].size, nb_components_pca)
    #"""
    
    enrollment_sets = user_im[0]
    testing_sets = user_im[1]
    
    number_of_users = len(enrollment_sets)
    
    print("PCA components : ", rp_comp)
    print(f'beta = {beta}')
    
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    
    for i in range(number_of_users): 
        print(i, end=' ')
        user = UserDataLattice()
        #user.new_user(enrollment_sets[i], rp_comp, pcaExt, beta=beta, noRP=False)
        
        user.new_user(enrollment_sets[i], rp_comp, pcaExt, beta=beta, noRP=False)
        #print(f'\nUser {i}:')
        
        for j in range(number_of_users):
            testim = testing_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(len(testim)):
                #print( user.authenticate( testim[k] ) )
                #score, mess_diff = user.authenticate_ROC( testim[k] )
                auth = user.authenticate( testim[k] )
                #print( auth, ' ', score )
                #print(f'{i} with {j}.{k} gave message_difference of \n{mess_diff}\n')
                if (truthval == 1):
                    if auth:
                        truePos += 1
                    else:
                        falseNeg += 1
                else:
                    if auth:
                        falsePos += 1
                    else:
                        trueNeg += 1
    
    print("\nResults are:\n", "       |  T     F  \n",f'Test T |  {truePos}    {falseNeg} \n', f'Test F |  {falsePos}   {trueNeg} \n')

    
    
    #with open(output_directory+'/LatResults/'+testname+'.txt', 'a') as file:
    with open(output_directory+'/LatResults/ForComp/'+testname+'.txt', 'a') as file:
        file.write(f'{rp_comp}, {beta}, {truePos}, {falseNeg}, {falsePos}, {trueNeg}\n')
    
    #print('done')

def test_data_information(n=20, unif=False):
    rp_comp = n
    nb_components_pca = int(rp_comp*2)
    #rp_comp = int(nb_components_pca/2)
    test_indices = [5]
    
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)
    
    #"""
    pca_name = 'PCA/pca' + str(nb_components_pca)
    pcaExt = FE.PCA_ext()
    
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = FE.directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, nb_components_pca)
        pcaExt.save_PCA(pca_name)
    """
    
    pcaExt = FE.RP_ext(user_im[0][0][0].size, nb_components_pca)
    #"""
    user_enrollment = user_im[0]
    user_testing = user_im[1]
    
    if unif:
        uniformMatrix = np.random.uniform(-1, 1, (n,n))
    
    for i in range(5): 
    #for i in range(len(user_enrollment)):
        
        extractor = FE.FeRP_Ext( pcaExt, rp_comp, 2500)
        #extractor = FE.FeRP_Ext( rpMain, rp_comp, 2500)
        #extractor = pcaExt
        
        #features = pcaExt.extract_features(user_enrollment[i])
        features = extractor.extract_features(user_enrollment[i])
        
        if unif:
            features = np.dot(features, uniformMatrix)
        
        string_output = f'Feature information for User : {i} \n'    
        
        min_vec = np.amin(features, 0)
        max_vec = np.amax(features, 0)
        ranges = (max_vec - min_vec)/2
        midpoint = (max_vec + min_vec)/2
        
        #string_output += f'min : \n {min_vec} \n max: \n {max_vec} \n\n midpoint: \n {midpoint} \n ranges: \n {ranges} \n\n'
        
        mean_vector = np.mean(features,0)
        var_vector = np.var(features,0)
        
        min_var = np.amin(var_vector)
        max_var = np.amax(var_vector)
        
        string_output += f'Variance: \n {var_vector}'
        #string_output += f'mean vector: \n {mean_vector} \n Variance: \n {var_vector} \n'#\n Highest Variance: {max_var}'
        string_output += f'\n ------------------------------------ \n'
        
        #print(string_output)
        #print(f'Feature information for User : {i} \n')
        print(f'Smallest Variance : {min_var} \nHighest Variance : {max_var} \n')
    
    return var_vector
    
def test_data_variances_single_rp(n, test_indices=[5]):
    rp_comp = n
    
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)
    
    rpMain = FE.RP_ext(user_im[0][0][0].size, n)
    
    user_enrollment = user_im[0]
    user_testing = user_im[1]
    
    for i in range(1): 
    #for i in range(len(user_enrollment)):
        
        features = rpMain.extract_features(user_enrollment[i])
        
        mean_vector = np.mean(features,0)
        var_vector = np.var(features,0)
        
        min_var = np.amin(var_vector)
        max_var = np.amax(var_vector)
        
        sorted_var = var_vector.tolist()
        sorted_var.sort()
        
        string_output = f'Feature information for User : {i} \n' 
        print(string_output)
        
        step = 4
        j = 0
        block = sorted_var[j: j+step]
        while len(block)>0:
            print(block)
            j += step
            block = sorted_var[j: j+step]
        
        print('\n -------------- \n')
    


def test_data_variances_single_rp_2(n, user1, user2, test_indices=[5]):
    rp_comp = n
    
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)
    
    rpMain = FE.RP_ext(user_im[0][0][0].size, n)
    
    user_enrollment = user_im[0]
    user_testing = user_im[1]
    
    
    features_1 = rpMain.extract_features(user_enrollment[user1])
    var_vector1 = np.var(features_1,0)
    
    features_2 = rpMain.extract_features(user_enrollment[user2])
    var_vector2 = np.var(features_2,0)
        
    string_output = f'Feature information for Users {user1} and {user2} \n' 
    print(string_output)
    
    step = 4
    j = 0
    
    while j<n:
        print(var_vector1[j:j+step])
        print(var_vector2[j:j+step])
        print('')
        j += step
 
def test_loop_pca_auth():
    
    #for n in [20, 40, 50, 100]:
    for n in [20]:#50, 100, 200]:#, 500, 1000]:
    #for n in [20, 20, 20, 40, 40, 40, 50, 50, 50, 100, 100, 100]:
        for beta in [1, 1, 1]:#1.5, 2, 3]:
        #for beta in [500]:#[0.1, 0.5, 1, 1.5, 2, 5]:
            for test_indices in [[0],[1],[2],[3],[4],[5],[6]]:#,[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]]:
                print(test_indices)
                test_pca_auth(n, beta, test_indices)
    
    print('Done!')

def test_loop_pca_auth_2():
    
    #for n in [20, 40, 50, 100]:
    #for n in [20, 50, 100, 200, 500, 1000]:
    #for n in [20, 20, 40, 40, 50, 50, 100, 100]:#, 500]:
    for n in [100]:
        for beta in [1, 1, 1, 1]:
        #for beta in [1, 1.5, 2, 3]:
        #for beta in [0.1, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]:
        #for beta in [0.05, 0.25, 0.5, 0.75, 1]:
            test_pca_auth(n, beta)
    
    print('Done!')



def test_correct_message(n=20, beta=1, d=3, fe="PCARP"):
    
    rp_comp = n
    nb_components_pca = int(2*n)
    test_indices = [5]
    
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)
    
    pcaExt = None
    testname = ""
    
    if (fe=="PCARP"):
        testname = "LatticeLsqResults_PCARP"
        pca_name = 'PCA/pca' + str(nb_components_pca)
        pcaExt = FE.PCA_ext()
        
        file = f'{output_directory}/{pca_name}.joblib'
        if os.path.exists(file):
            pcaExt.load_PCA(pca_name)
        else:
            training_images = FE.directory_loader(training_faces_path)
            pcaExt.train_PCA(training_images, nb_components_pca)
            pcaExt.save_PCA(pca_name)
    elif (fe=="RPRP"):
        testname = "LatticeLsqResults_RPRP"
        pcaExt = FE.RP_ext(user_im[0][0][0].size, nb_components_pca)
    else:
        return "Not an accepted feature extractor try PCARP or RPRP"
    
    #testname = "LatticeLsqResults_RPRP"
    #testname = "LatticeLsqResults_PCARP"
    #testname = "LatticeLsqResults_PCA"
    #nb_components_pca = n
    
    testname += f'_d{d}'
    """
    pca_name = 'PCA/pca' + str(nb_components_pca)
    pcaExt = FE.PCA_ext()
    
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = FE.directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, nb_components_pca)
        pcaExt.save_PCA(pca_name)
    #"""
    #rpMmain
    #pcaExt = FE.RP_ext(user_im[0][0][0].size, nb_components_pca)
    #"""
    
    enrollment_sets = user_im[0]
    testing_sets = user_im[1]
    
    number_of_users = len(enrollment_sets)
    
    print("Final components : ", rp_comp)
    print(f'beta = {beta}')
    
    print("User codeword message is:")
    
    for i in range(number_of_users): 
        print(f'User {i}:')
        user = UserDataLattice()
        user.new_user(enrollment_sets[i], rp_comp, pcaExt, beta=beta, noRP=False)
        
        print(user.correct_message, '\n')
        
    print('done')





def test_message_diff(n=20, beta=1, d=3, fe="PCARP"):
    
    rp_comp = n
    nb_components_pca = int(2*n)
    test_indices = [5]
    
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)
    
    pcaExt = None
    testname = ""
    
    if (fe=="PCARP"):
        testname = "LatticeLsqResults_PCARP"
        pca_name = 'PCA/pca' + str(nb_components_pca)
        pcaExt = FE.PCA_ext()
        
        file = f'{output_directory}/{pca_name}.joblib'
        if os.path.exists(file):
            pcaExt.load_PCA(pca_name)
        else:
            training_images = FE.directory_loader(training_faces_path)
            pcaExt.train_PCA(training_images, nb_components_pca)
            pcaExt.save_PCA(pca_name)
    elif (fe=="RPRP"):
        testname = "LatticeLsqResults_RPRP"
        pcaExt = FE.RP_ext(user_im[0][0][0].size, nb_components_pca)
    else:
        return "Not an accepted feature extractor try PCARP or RPRP"
    
    #testname = "LatticeLsqResults_RPRP"
    #testname = "LatticeLsqResults_PCARP"
    #testname = "LatticeLsqResults_PCA"
    #nb_components_pca = n
    
    testname += f'_d{d}'
    """
    pca_name = 'PCA/pca' + str(nb_components_pca)
    pcaExt = FE.PCA_ext()
    
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = FE.directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, nb_components_pca)
        pcaExt.save_PCA(pca_name)
    #"""
    #rpMmain
    #pcaExt = FE.RP_ext(user_im[0][0][0].size, nb_components_pca)
    #"""
    
    enrollment_sets = user_im[0]
    testing_sets = user_im[1]
    
    number_of_users = len(enrollment_sets)
    
    print("Final components : ", rp_comp)
    print(f'beta = {beta}')
    
    print("User message difference is:")
    
    for i in range(number_of_users): 
        print(f'User {i}:')
        user = UserDataLattice()
        user.new_user(enrollment_sets[i], rp_comp, pcaExt, beta=beta, noRP=False)

        for j in range(number_of_users):
            testim = testing_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(len(testim)):
                #print( user.authenticate( testim[k] ) )
                score, mess_diff = user.authenticate_ROC( testim[k] )
                
                print(f'... vs User {j}_{k}: \n', mess_diff, '\n')
    
    print("Done)")






        




