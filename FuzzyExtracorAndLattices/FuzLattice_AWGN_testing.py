"""
Code for a fuzzy extractor using lattices. And testing using AWGN simulated data.
Written by Renaud Brien
"""

import numpy as np
import math
import time, datetime
import os, sys, glob
import Lattice
import matrix_representation
import belief_propagation as bp
import LatticeCodeLSQ as lclsq
from LatticeCodeLSQ import LatticeLSQ
import Fuzzy_Extractor as FE
#from sklearn.externals import joblib
import joblib
from operator import itemgetter


training_faces_path = 'C:/Coding/ARFaces/ARFacesTrain'  #path to images for training PCA
testing_faces_path = 'C:/Coding/ARFaces/ARFacesTest'    #path to images for Testing the extractor
saved_mean_faces_path = 'C:/Coding/FaceMeans'
output_directory = 'C:/Coding/PythonFuzzyOutputs'       #path for test outputs/lattice inputs

valSeq3 = [1, 0.57735, 0.57735]
valSeq5 = [1, 0.4472135955, 0.4472135955, 0.4472135955, 0.4472135955]

class PCA_mean_faces:

    def __init__(self, n):

        file_name_faces = 'PCA_mean_'+str(n)
        file_faces = saved_mean_faces_path+'/'+file_name_faces+'.joblib'

        if os.path.exists(file_faces):
            self.load_data(file_name_faces)
        else:

            pca_name = 'PCA/pca' + str(n)
            pcaExt = FE.PCA_ext()

            file = f'{output_directory}/{pca_name}.joblib'
            if os.path.exists(file):
                pcaExt.load_PCA(pca_name)
            else:
                training_images = FE.directory_loader(training_faces_path)
                pcaExt.train_PCA(training_images, nb_components_pca)
                pcaExt.save_PCA(pca_name)

            user_im = FE.loader_test_images(testing_faces_path, test_index=[])[0]

            self.faces = []
            self.minVar = []
            self.maxVar = []

            for i in range(len(user_im)):
                features = pcaExt.extract_features(user_im[i])

                mean_vector = np.mean(features,0)
                var_vector = np.var(features,0)
                min_var = np.amin(var_vector)
                max_var = np.amax(var_vector)

                self.faces.append(mean_vector)
                self.minVar.append(min_var)
                self.maxVar.append(max_var)

            self.save_data(file_name_faces)
        #End of Init

    def save_data(self, name):
        file = saved_mean_faces_path+'/'+name+'.joblib'
        joblib.dump(self, file)

    def load_data(self, name):
        file = saved_mean_faces_path+'/'+name+'.joblib'
        if os.path.exists(file):
            # Load it with joblib
            #if VERBOSE: print('Loading', file)
            gcbObj = joblib.load(file)
            self.faces = gcbObj.faces
            self.minVar = gcbObj.minVar
            self.maxVar = gcbObj.maxVar
            return True
        else:
            return False

class PCARP_mean_faces:

    def __init__(self, n, seed, save=False):

        file_name_faces = 'PCARP_mean_'+str(n)+'_seed_'+str(seed)
        file_faces = saved_mean_faces_path+'/'+file_name_faces+'.joblib'

        if os.path.exists(file_faces):
            print('loading...')
            self.load_data(file_name_faces)
        else:
            pca_n = int(2*n)
            pca_name = 'PCA/pca' + str(pca_n)
            pcaExt = FE.PCA_ext()

            file = f'{output_directory}/{pca_name}.joblib'
            if os.path.exists(file):
                pcaExt.load_PCA(pca_name)
            else:
                training_images = FE.directory_loader(training_faces_path)
                pcaExt.train_PCA(training_images, nb_components_pca)
                pcaExt.save_PCA(pca_name)

            user_im = FE.loader_test_images(testing_faces_path, test_index=[])[0]

            self.faces = []
            self.minVar = []
            self.maxVar = []

            generator = np.random.RandomState(seed)
            matSeed = generator.randint(2**31 -1)
            trSeed = generator.randint(2**31 -1)

            extractor = FE.FeRP_Ext( pcaExt, n, 2500, matSeed, trSeed)

            for i in range(len(user_im)):
                features = extractor.extract_features(user_im[i])

                mean_vector = np.mean(features,0)
                var_vector = np.var(features,0)
                min_var = np.amin(var_vector)
                max_var = np.amax(var_vector)

                self.faces.append(mean_vector)
                self.minVar.append(min_var)
                self.maxVar.append(max_var)

            if(save):
                self.save_data(file_name_faces)
        #End of Init

    def save_data(self, name):
        file = saved_mean_faces_path+'/'+name+'.joblib'
        joblib.dump(self, file)

    def load_data(self, name):
        file = saved_mean_faces_path+'/'+name+'.joblib'
        if os.path.exists(file):
            # Load it with joblib
            #if VERBOSE: print('Loading', file)
            gcbObj = joblib.load(file)
            self.faces = gcbObj.faces
            self.minVar = gcbObj.minVar
            self.maxVar = gcbObj.maxVar
            return True
        else:
            return False

class AWGN_output_generator:

    def __init__(self, mean_vector, variance, seed=None):

        self.mean_vector = mean_vector
        self.n = len(mean_vector)
        self.var = variance

        if (seed == None):
            self.seed = np.random.seed()
        else:
            self.seed = seed

        self.gen = np.random.RandomState(seed)

    def newOutput(self):
        noise = self.gen.normal(0, np.sqrt(self.var), self.n)
        return self.mean_vector + noise

class UserLattice:

    def __init__(self):
        pass

    def new_user(self, mean_vector, variance, beta=1, lsqSeed=None):

        if lsqSeed == None:
            self.lsqSeed = np.random.randint(0,100)
        else:
            self.lsqSeed = lsqSeed

        self.n = len(mean_vector)
        self.sigma2 = variance
        self.beta = beta

        #Scaling value to make the LDLC work with the chosen variance
        scaling_factor = ( self.beta*self.sigma2 / 0.0585 )**(-0.5)


        scaled_seq = []
        for val in valSeq3:
            scaled_seq.append(val*scaling_factor)

        #Generating the Lattice Code matix H
        lsq = lclsq.LatticeLSQ(self.n, 3, self.lsqSeed)
        self.ldlc_mat_rep = lsq.get_sparse_rep(scaled_seq)

        #Building G for the LDLC
        H = matrix_representation.build_matrix_from_row_list(self.ldlc_mat_rep[0], self.ldlc_mat_rep[1])
        self.G = np.linalg.inv(H)

        '''
        print(f'sigma2 = {self.sigma2}')
        sigbound = ( (abs(np.linalg.det(self.G)))**(2.0/self.n) )/(2*math.e*math.pi)
        print(f'Sigma2 bound = {sigbound}')
        '''

        #Building the Decoder
        self.decoder = bp.SingleGaussianDecoder(self.ldlc_mat_rep, self.sigma2)

        #Decoding the mean vector:
        mean_message = self.decoder.decode_3( mean_vector, 100 )

        mean_codeword = self.G.dot( np.array(mean_message) )

        #Computing Helper data : h = x_bar - Q(x_bar)
        self.helper_data = mean_vector - mean_codeword

        #FOR TESTING PURPOSES : The correct Codeword
        self.correct_message = mean_message
        self.correct_codeword = mean_codeword

    def authenticate( self, features ):
        message = self.decoder.decode_3( features - self.helper_data, 100 )
        return (message == self.correct_message)

    def auth_mess_diff(self, features):
        message = self.decoder.decode_3( features- self.helper_data, 100 )
        message_difference = np.array(self.correct_message)-np.array(message)
        print(message_difference)





def test_pcarp_with_RPseed(n, input_numbers, seed, beta=1):

    pcarpFaces = PCARP_mean_faces(n, seed)

    faces = pcarpFaces.faces
    minVar = pcarpFaces.minVar
    maxVar = pcarpFaces.maxVar

    nb_users = len(faces)

    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    print(f'Testing lsq with PCARP with RP seed {seed} on {input_numbers} inputs per users')
    print(f'Using minVar and Beta = {beta}')

    #for i in range(1):
    for i in range(nb_users):

        print(i, end=' ')
        #print('')

        user = UserLattice()
        user.new_user(faces[i], minVar[i], beta)

        #for j in range(1):
        for j in range(nb_users):
            feat = AWGN_output_generator(faces[j], minVar[j])
            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(input_numbers):
                feature = feat.newOutput()
                auth = user.authenticate( feature )
                #auth = user.auth_mess_diff( feature )
                #auth = user.authenticate( faces[j] )
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

    print(f'{n}, {seed}, {input_numbers}, {beta}, {truePos}, {falseNeg}, {falsePos}, {trueNeg}\n')


def test_pcarp(n, input_numbers, beta=1, seeding=None, varUsed="mid"):

    if seeding==None:
        seed = np.random.randint(2**16-1)
    else:
        seed = seeding

    #testname = "Lattice_PCARP_AWGN_minVar"
    #testname = "Lattice_PCARP_AWGN_maxVar"
    #testname = "Lattice_PCARP_AWGN_midVar"

    testname=""
    if varUsed == "min":
        testname = "Lattice_PCARP_AWGN_minVar"
    elif varUsed == "max":
        testname = "Lattice_PCARP_AWGN_maxVar"
    elif varUsed == "mid":
        testname = "Lattice_PCARP_AWGN_midVar"

    pcarpFaces = PCARP_mean_faces(n, seed)

    faces = pcarpFaces.faces
    minVar = pcarpFaces.minVar
    maxVar = pcarpFaces.maxVar

    nb_users = len(faces)

    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    print(f'Testing lsq with PCARP with RP seed {seed} on {input_numbers} inputs per users')
    print(f'Using minVar and Beta = {beta}')

    #for i in range(1):
    for i in range(nb_users):

        print(i, end=' ')
        #print('')

        user = UserLattice()
        #user.new_user(faces[i], minVar[i], beta)
        #user.new_user(faces[i], maxVar[i], beta)
        #user.new_user(faces[i], (maxVar[i]+minVar[i])/2, beta)

        if varUsed == "min":
            user.new_user(faces[i], minVar[i], beta)
        elif varUsed == "max":
            user.new_user(faces[i], maxVar[i], beta)
        elif varUsed == "mid":
            user.new_user(faces[i], (maxVar[i]+minVar[i])/2, beta)

        #for j in range(1):
        for j in range(nb_users):
            #feat = AWGN_output_generator(faces[j], minVar[j])
            #feat = AWGN_output_generator(faces[j], maxVar[j])
            #feat = AWGN_output_generator(faces[j], (maxVar[i]+minVar[i])/2)

            feat = None
            if varUsed == "min":
                feat = AWGN_output_generator(faces[j], minVar[j])
            elif varUsed == "max":
                feat = AWGN_output_generator(faces[j], maxVar[j])
            elif varUsed == "mid":
                feat = AWGN_output_generator(faces[j], (maxVar[i]+minVar[i])/2)

            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(input_numbers):
                feature = feat.newOutput()
                auth = user.authenticate( feature )
                #auth = user.auth_mess_diff( feature )
                #auth = user.authenticate( faces[j] )
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

    with open(output_directory+'/LatResults/AWGN/'+testname+'.txt', 'a') as file:
        file.write(f'{n}, {input_numbers}, {beta}, {truePos}, {falseNeg}, {falsePos}, {trueNeg}, {seed}\n')

    print(f'{n}, {input_numbers}, {beta}, {truePos}, {falseNeg}, {falsePos}, {trueNeg}, \t\t {seed}\n')



def test_pcarp_diffExt(n, input_numbers, beta=1, varUsed="mid"):

    testname=""

    if varUsed == "min":
        testname = "Lattice_PCARP_diffExt_AWGN_minVar"
    elif varUsed == "max":
        testname = "Lattice_PCARP_diffExt_AWGN_maxVar"
    elif varUsed == "mid":
        testname = "Lattice_PCARP_diffExt_AWGN_midVar"
    else:
        return "Wrong input. Choose between: min, max, mid"

    nb_users = 26

    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    print(f'Testing lsq with PCARP with different RP extractors on {input_numbers} inputs per users')
    print(f'Using minVar and Beta = {beta}')

    #for i in range(1):
    for i in range(nb_users):

        print(i, end=' ')
        #print('')

        seed = np.random.randint(2**16-1)

        pcarpFaces = PCARP_mean_faces(n, seed)

        faces = pcarpFaces.faces
        minVar = pcarpFaces.minVar
        maxVar = pcarpFaces.maxVar

        user = UserLattice()

        if varUsed == "min":
            user.new_user(faces[i], minVar[i], beta)
        elif varUsed == "max":
            user.new_user(faces[i], maxVar[i], beta)
        elif varUsed == "mid":
            user.new_user(faces[i], (maxVar[i]+minVar[i])/2, beta)


        #for j in range(1):
        for j in range(nb_users):
            feat = None
            if varUsed == "min":
                feat = AWGN_output_generator(faces[j], minVar[j])
            elif varUsed == "max":
                feat = AWGN_output_generator(faces[j], maxVar[j])
            elif varUsed == "mid":
                feat = AWGN_output_generator(faces[j], (maxVar[i]+minVar[i])/2)


            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(input_numbers):
                feature = feat.newOutput()
                auth = user.authenticate( feature )
                #auth = user.auth_mess_diff( feature )
                #auth = user.authenticate( faces[j] )
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

    with open(output_directory+'/LatResults/AWGN/'+testname+'.txt', 'a') as file:
        file.write(f'{n}, {input_numbers}, {beta}, {truePos}, {falseNeg}, {falsePos}, {trueNeg}, {seed}\n')

    print(f'{n}, {input_numbers}, {beta}, {truePos}, {falseNeg}, {falsePos}, {trueNeg}, \t\t {seed}\n')





def test_loop_pcarp( n_list, beta_list, test_amount=20, seed=None):
    for n in n_list:
        for beta in beta_list:
            test_pcarp(n, test_amount, beta, seed)
    print('Done!')



def test_loop_pcarp_diffExt( n_list, beta_list, test_amount=20, varType="mid"):
    for n in n_list:
        for beta in beta_list:
            test_pcarp_diffExt(n, test_amount, beta, varType)
    print('Done!')







##AWGN with Sutcu

class UserSutcu:

    def __init__(self):
        pass

    def new_user(self, global_codebook, mean_vector, variance, beta=1):

        #self.shiftSeed = np.random.seed()
        self.shiftSeed = 0

        midpoint = mean_vector
        ranges = variance*np.ones(global_codebook.number_of_components)

        self.midpoint = midpoint
        self.ranges = ranges

        #print( f'min : \n {min_vec} \n max: \n {max_vec} \n midpoint: \n {midpoint} \n ranges: \n {ranges}')

        self.user_codebook = FE.UserCodebook(global_codebook, beta, midpoint.tolist(), ranges.tolist())

        h_bound = 2500
        self.helper_data = np.random.uniform(-h_bound, h_bound, global_codebook.number_of_components).tolist()

        #FOR TESTING PURPOSES
        self.correct_codeword = self.user_codebook.quantize(midpoint.tolist(), self.shiftSeed)

    def authenticate( self, features ):

        codeword = self.user_codebook.quantize(features.tolist(), self.shiftSeed)

        return (codeword == self.correct_codeword)

    def authenticate_ROC( self, features ):

        lam = self.user_codebook.lam

        #print("Features : ", features )

        generator = np.random.RandomState(self.shiftSeed)
        shift = []
        for i in range(len(features)):
            shift.append( generator.randint( -lam, lam ) * self.user_codebook.steps[i] )

        max_score = 0

        for i in range(len(features)):
            dist = abs( (features[i] + shift[i]) - self.correct_codeword[i] )
            score = dist / self.user_codebook.steps[i]

            #if(i == 0):
                #print(f'Code Dist = {self.user_codebook.steps[i]} \n ImageDist = {dist} \n Score = {score}')
                #print(f'Score = {score}')

            if (score > max_score):
                max_score = score

            #print(score, max_score)

        #print(' ')

        return max_score



def test_Sutcu_PCARP_roc(n, input_numbers, beta=1, seeding=None, varUsed="mid", tplist=None):

    if seeding==None:
        seed = np.random.randint(2**16-1)
    else:
        seed = seeding

    #testname = "Lattice_PCARP_AWGN_minVar"
    #testname = "Lattice_PCARP_AWGN_maxVar"
    #testname = "Lattice_PCARP_AWGN_midVar"

    testname=""
    if varUsed == "min":
        testname = "Sutcu_PCARP_AWGN_minVar"
    elif varUsed == "max":
        testname = "Sutcu_PCARP_AWGN_maxVar"
    elif varUsed == "mid":
        testname = "Sutcu_PCARP_AWGN_midVar"

    pcarpFaces = PCARP_mean_faces(n, seed)

    faces = pcarpFaces.faces
    minVar = pcarpFaces.minVar
    maxVar = pcarpFaces.maxVar

    nb_users = len(faces)

    codebook_name = "Codebooks/user_codebook_c" + str(n)
    global_codebook = FE.GlobalCodebook(n, 3)
    file = f'{output_directory}/{codebook_name}.joblib'
    if os.path.exists(file):
        global_codebook.load_codebook(codebook_name)
    else:
        global_codebook.save_codebook(codebook_name)

    testname2 = testname + f'beta_{beta}_ROC_'

    timestring = FE.get_date_string()
    testname_date = testname2 +' '+ timestring

    obs_list = []

    for i in range(nb_users):
        user = UserSutcu()

        if varUsed == "min":
            user.new_user(global_codebook, faces[i], minVar[i], beta)
        elif varUsed == "max":
            user.new_user(global_codebook, faces[i], maxVar[i], beta)
        elif varUsed == "mid":
            user.new_user(global_codebook, faces[i], (maxVar[i]+minVar[i])/2, beta)


        for j in range(nb_users):
            feat = None
            if varUsed == "min":
                feat = AWGN_output_generator(faces[j], minVar[j])
            elif varUsed == "max":
                feat = AWGN_output_generator(faces[j], maxVar[j])
            elif varUsed == "mid":
                feat = AWGN_output_generator(faces[j], (maxVar[i]+minVar[i])/2)

            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(input_numbers):
                feature = feat.newOutput()
                score = user.authenticate_ROC(feature)

                obs_list.append( (score, truthval, i, j, k) )
                #print( score, truthval, i, j, k )

    obs_list.sort(key=itemgetter(0))

    total_size = len(obs_list)
    total_true = nb_users * input_numbers

    total_false = total_size - total_true

    tpr = 0
    fpr = 0

    current_true = 0
    minDistToEER = 1.0
    eer_index = 0

    fpr_at_min = 0

    with open(output_directory+'/LatResults/AWGN/SutcuROCs/'+testname_date+'.txt', 'a') as file:

        file.write('Testing Improvements on Sutcu AWGN PCARP \n')
        file.write('Score, True_match, userID, testID.testImageID\n')

        for obs in obs_list:
            file.write( FE.observation_to_string(obs) )

            current_true += obs[1]
            eer_index += 1
            tpr = current_true / total_true
            fpr = (eer_index - current_true) / total_false

            distToEER = abs(1.0 - tpr - fpr)
            if (distToEER < minDistToEER):
                fpr_at_min = fpr
                minDistToEER = distToEER

            #print(f'tpr: {tpr}, fpr: {fpr}, minDist: {minDistToEER}')

    result_string = f'Test {testname} returns an EER of \n {fpr_at_min}'

    print( result_string )

    if tplist != None:
        print('\nPrinting Confusion Matrix Values')

        testname3 = testname+' '+timestring

        for tp in tplist:
            curTP = 0
            curTN = 0
            curFP = 0
            curFN = 0
            score = 0

            for obs in obs_list:

                if(curTP < tp):
                    if(obs[1] == 1):
                        curTP += 1
                    else:
                        curFP += 1
                    if curTP == tp:
                        score = obs[0]
                else:
                    if(obs[1] == 1):
                        curFN += 1
                    else:
                        curTN += 1

            res_string = f'{n}, {beta}, {curTP}, {curFN}, {curFP}, {curTN}, {seed}'
            with open(output_directory+'/LatResults/AWGN/SutcuConf/'+testname3+'.txt', 'a') as file:
                file.write(f'{n}, {input_numbers}, {beta}, {curTP}, {curFN}, {curFP}, {curTN}, {seed}\n')
            print(res_string)


tplist0 = [1, 2, 3, 4, 5, 37, 38, 39, 40, 41, 42, 43, 100, 101, 102, 103, 104, 105, 150, 151, 152, 153, 154, 155, 159, 160, 161, 170, 175, 180, 185]






