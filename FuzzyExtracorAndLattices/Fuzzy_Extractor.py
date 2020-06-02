'''
This file contains the improved Sutcu fuzzy extractor.

For Testing only. Some parts are not made for actual implementation:
This implementation saves the secret for testing purposes.
Written by Renaud Brien
'''

import os
#import random
import sys
import time, datetime
import cv2
import math
#import dlib
import glob
import numpy as np
#import openface
from sklearn.decomposition import PCA
#from sklearn.externals import joblib
import joblib
from operator import itemgetter


training_faces_path = 'C:/Coding/ARFaces/ARFacesTrain' #path to images for training PCA
testing_faces_path = 'C:/Coding/ARFaces/ARFacesTest'   #path to images for testing extractor
output_directory = 'C:/Coding/PythonFuzzyOutputs'

#for looping through sets of 1 or 2 test images
testim_1 = [[0], [1], [2], [3], [4], [5], [6]]
testim_2 = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]

#training_images = directory_loader(training_faces_path)
#user_im = loader_test_images(testing_faces_path, test_index=[5])

VERBOSE = False


## User and Server side classes for the Fuzzy Extractor
class GlobalCodebook:
    
    def __init__(self, number_of_components, step, steps_array=[]):
        self.number_of_components = number_of_components
        if len(steps_array) == number_of_components:
            self.generate_codebook_array(steps_array, number_of_components)
        else:
            self.generate_codebook(step, number_of_components)
        
    
    def generate_codebook(self,step, number_of_components):
        
        steps = []
        for i in range(number_of_components):
            steps.append(step)
            
        start_point = []
        for i in range(number_of_components):
            start_point.append(np.random.uniform(0,steps[i]))
        
        self.starting_point = start_point
        self.steps = steps
        return True
        
    def generate_codebook_array(self,steps_array, number_of_components):
        
        steps = []
        
        if len(steps_array) != number_of_components:
            return False
        else:
            steps = steps_array
            
        start_point = []
        for i in range(number_of_components):
            start_point.append(np.random.uniform(0,steps[i]))
        
        self.starting_point = start_point
        self.steps = steps
        return True
        
    def set_startpoints( self, startpoints ):
        self.starting_point = startpoints
        return True
        
    def decode_lowerbound(self, input_array):
        if len(input_array) == self.number_of_components:
            message = []
            for i in range(len(input_array)):
                message.append( math.floor((input_array[i] - self.starting_point[i])/self.steps[i]) )
            return message
        else:
            return False
    
    def quantize(self, input_array):
        if len(input_array) == self.number_of_components:
            message = self.decode_lowerbound(input_array)
            codeword = []
            for i in range(len(message)):
                codeword.append( message[i]*self.steps[i] + self.starting_point[i] )
            return codeword
        else:
            return False
        
    def save_codebook(self, name="Codebooks/global_codebook"):
        file = output_directory+'/'+name+'.joblib'
        joblib.dump(self, file)
        #if VERBOSE: print('Saved global_codebook to',file,'.')
        
    def load_codebook(self, name='Codebooks/global_codebook'):
        file = output_directory+'/'+name+'.joblib'
        if os.path.exists(file):
            # Load it with joblib
            #if VERBOSE: print('Loading', file)
            gcbObj = joblib.load(file)
            self.number_of_components = gcbObj.number_of_components
            self.starting_point = gcbObj.starting_point
            self.steps = gcbObj.steps
            return True
        else:
            return False
            

class UserCodebook:
    
    def __init__(self, global_codebook, beta, mid_point_vector, radius_vector, lam=1024):
        self.lam = 1024
        self.generate_codebook(global_codebook, beta, mid_point_vector, radius_vector)
        pass
        
    
    def generate_codebook(self,global_codebook, beta, mid_point_vector, radius_vector):
        
        gc_point = global_codebook.quantize(mid_point_vector)
        self.number_of_components = global_codebook.number_of_components
        
        steps = []
        for i in range(len(radius_vector)):
            steps.append( math.ceil( 2*(beta*radius_vector[i] / global_codebook.steps[i]) + 1 )*global_codebook.steps[i] )
            
        start_point = []
        for i in range(self.number_of_components):
            x = math.floor( (gc_point[i] - global_codebook.starting_point[i]) / steps[i] )*steps[i]
            start_point.append( gc_point[i] - x )
        
        self.starting_point = start_point
        self.steps = steps
        return True
        
    def decode(self, input_array):
        if len(input_array) == self.number_of_components:
            message = []
            for i in range(len(input_array)):
                message.append(round((input_array[i] - self.starting_point[i])/self.steps[i]) )
            return message
        else:
            return False
    
    def quantize(self, input_array, shift_seed=None):
        if len(input_array) == self.number_of_components:
            message = self.decode(input_array)
            if shift_seed != None:
                generator = np.random.RandomState(shift_seed)
                for i in range(len(message)):
                    message[i] += generator.randint( -self.lam, self.lam )
            codeword = []
            for i in range(len(message)):
                codeword.append( message[i]*self.steps[i] + self.starting_point[i] )
            return codeword
        else:
            return False
        
    def save_codebook(self, name="Codebooks/user_codebook"):
        file = output_directory+'/'+name+'.joblib'
        joblib.dump(self, file)
        #if VERBOSE: print('Saved global_codebook to',file,'.')
        
    def load_codebook(self, name='Codebooks/user_codebook'):
        file = output_directory+'/'+name+'.joblib'
        if os.path.exists(file):
            # Load it with joblib
            #if VERBOSE: print('Loading', file)
            gcbObj = joblib.load(file)
            self.number_of_components = gcbObj.number_of_components
            self.starting_point = gcbObj.starting_point
            self.steps = gcbObj.steps
            return True
        else:
            return False
            

class UserData:
    
    def __init__(self):
        pass
        
    def new_user(self, training_images, global_codebook, main_extractor, beta=3, noRP=False ):
        self.matSeed = np.random.seed()
        self.trSeed = np.random.seed()
        #self.shiftSeed = np.random.seed()
        self.shiftSeed = 0
        
        if noRP:
            self.extractor = main_extractor
        else:
            self.extractor = FeRP_Ext( main_extractor, global_codebook.number_of_components, 2500, self.matSeed, self.trSeed)
        
        features = self.extractor.extract_features(training_images)
        
        min_vec = np.amin(features, 0)
        max_vec = np.amax(features, 0)
        ranges = (max_vec - min_vec)/2
        midpoint = (max_vec + min_vec)/2
        
        self.midpoint = midpoint
        self.ranges = ranges
        
        #print( f'min : \n {min_vec} \n max: \n {max_vec} \n midpoint: \n {midpoint} \n ranges: \n {ranges}')
        
        self.user_codebook = UserCodebook(global_codebook, beta, midpoint.tolist(), ranges.tolist())
        
        h_bound = 2500
        self.helper_data = np.random.uniform(-h_bound, h_bound, global_codebook.number_of_components).tolist()
        
        #FOR TESTING PURPOSES
        self.correct_codeword = self.user_codebook.quantize(midpoint.tolist(), self.shiftSeed)
        
    def authenticate( self, image ):
        
        features = self.extractor.extract_features([image])
        
        codeword = self.user_codebook.quantize(features.tolist(), self.shiftSeed)
        
        return (codeword == self.correct_codeword)
        
    def authenticate_ROC( self, image ):
        
        lam = self.user_codebook.lam
        
        features = self.extractor.extract_features([image])[0]
        
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
    
        

##Feature Extractors
class PCA_FeatureExt_sklearn:
    
    def __init__(self):
        pass
    
    
    def train_PCA(self, images, number_of_features=200):
        training_set = np.ndarray(shape=(len(images), images[0].shape[0]*images[0].shape[1]))
        for i, image in enumerate(images):
            training_set[i] = image.flatten()
        pca = PCA(number_of_features)
        if VERBOSE: print(f'Training PCA with {number_of_features} features, and with {len(training_set)} items in the training set.')
        pca.fit(training_set)
        
        self.extractor = pca
        self.nb_of_features = number_of_features
        return True
    
    def load_PCA(self,name='PCA/pca'):
        file = f'{output_directory}/{name}.joblib'
        if os.path.exists(file):
            # Load it with joblib
            if VERBOSE: print('Loading', file)
            pcaObj = joblib.load(file)
            self.extractor = pcaObj.extractor
            self.nb_of_features = pcaObj.nb_of_features
            return True
        else: # No model
            message = f'Error in get_PCA_model(name={repr(name)}), no such file: {file}\n'
            with open(f'{output_directory}/errors/errors.txt', 'a') as log_file:
                log_file.write(message)
            print(message)
            return False
    
    def save_PCA(self, name="PCA/pca"):
        file = f'{output_directory}/{name}.joblib'
        joblib.dump(self, file)
        if VERBOSE: print(f'Saved PCA to {file}.')
        
    def extract_features(self, images):
        training_images = np.empty(shape=(len(images), images[0].shape[0]*images[0].shape[1]))
        for i, image in enumerate(images):
            training_images[i] = image.flatten().reshape(1,-1)
        if VERBOSE: print('Flattened training_images: (Number, Length):', training_images.shape)
        feature_vectors = self.extractor.transform(training_images)
        if VERBOSE: print('feature_vectors (Number, Length):', feature_vectors.shape)
        return feature_vectors

class PCA_ext:
    
    def __init__(self):
        pass
        
    def train_PCA(self, images, number_of_features=200):
        training_set = np.ndarray(shape=(len(images), images[0].size))
        for i, image in enumerate(images):
            training_set[i] = image.flatten()
        
        self.mean, self.pcaMatrix = cv2.PCACompute(training_set, mean=None, maxComponents = number_of_features)
        if VERBOSE: print(f'Training PCA with {number_of_features} features, and with {len(training_set)} items in the training set.')
        self.nb_of_features = number_of_features
        return True
    
    def save_PCA(self, name="PCA/pca"):
        file = f'{output_directory}/{name}.joblib'
        joblib.dump(self, file)
        if VERBOSE: print(f'Saved PCA to {file}.')
    
    def load_PCA(self,name='PCA/pca'):
        file = f'{output_directory}/{name}.joblib'
        if os.path.exists(file):
            # Load it with joblib
            if VERBOSE: print('Loading', file)
            pcaObj = joblib.load(file)
            self.mean = pcaObj.mean
            self.pcaMatrix = pcaObj.pcaMatrix
            self.nb_of_features = pcaObj.nb_of_features
            return True
        else: # No model
            message = f'Error in get_PCA_model(name={repr(name)}), no such file: {file}\n'
            with open(f'{output_directory}/errors/errors.txt', 'a') as log_file:
                log_file.write(message)
            print(message)
            return False
        
    def extract_features(self, images):
        training_images = np.empty(shape=(len(images), images[0].size))
        for i, image in enumerate(images):
            training_images[i] = image.flatten().reshape(1,-1) - self.mean
        feature_vectors = np.dot(self.pcaMatrix, training_images.transpose()).transpose()
        
        return feature_vectors
    

class RP_ext:
    
    def __init__(self, input_size, nb_of_features, threshold=2500, matSeed=None, trSeed=None):
        self.new_extractor(input_size, nb_of_features, threshold, matSeed, trSeed)
        pass
    
    def new_extractor(self, input_size, nb_of_features, threshold=2500, matSeed=None, trSeed=None):
        self.input_size = input_size
        self.nb_of_features = nb_of_features
        self.threshold = threshold
        
        self.scaling_factor = np.sqrt( input_size/nb_of_features )
        
        #Generating Matrix
        if matSeed == None:
            self.matSeed = np.random.seed()
        else:
            self.matSeed = matSeed
            
        mat_generator = np.random.RandomState(self.matSeed)
        
        self.rpMat = mat_generator.normal(0, np.sqrt(1/input_size ) , (nb_of_features, input_size))
        
        #Generating Translation Vector
        if trSeed == None:
            self.trSeed = np.random.seed()
        else:
            self.trSeed = trSeed
            
        tr_generator = np.random.RandomState(self.matSeed)
        
        self.trVec = mat_generator.uniform(2*threshold, 10*threshold, (1, input_size))
        
    def get_seeds(self):
        return (self.matSeed, self.trSeed)
        
    def extract_features(self, images):
        training_images = np.empty(shape=(len(images), self.input_size))
        for i, image in enumerate(images):
            training_images[i] = image.flatten().reshape(1,-1) + self.trVec
        feature_vectors = np.dot(self.rpMat, training_images.transpose()).transpose() * self.scaling_factor
        
        return feature_vectors
        
class FeRP_Ext:
    
    def __init__(self, first_extractor, nb_of_features, threshold=2500, matSeed=None, trSeed=None):
        
        self.first_extractor = first_extractor
        rp_input_size = first_extractor.nb_of_features
        
        self.rp_extractor = RP_ext(rp_input_size, nb_of_features, threshold, matSeed, trSeed)
        
    def extract_features(self, images):
        features_temp = self.first_extractor.extract_features(images)
        features_as_list = []
        for i in range(features_temp.shape[0]):
            features_as_list.append(features_temp[i])
        
        final_features = self.rp_extractor.extract_features(features_as_list)
        
        return final_features
        
        
    
    
##Loading and Testing functions
#Image Loading Functions
def directory_loader(location='', extention='.png', recursive=True):
    image_paths = glob.glob(location+'/**/*'+extention, recursive=recursive)
    return list_loader(image_paths)


def list_loader(image_paths, flag=cv2.IMREAD_GRAYSCALE):
    images = []
    for path in image_paths:
        if path[-4:] == '.pgm':
            images.append(cv2.imread(path, -1))
        else:
            images.append(cv2.imread(path, flag))
    return images
    
def list_to_matrix( input_list ):
    output_matrix = None
    for im in input_list:
        try:
            output_matrix = np.vstack( (output_matrix, im.flatten().reshape(1,-1)) )
        except:
            output_matrix = im.flatten().reshape(1,-1)
    return output_matrix
    

def loader_test_images(location='', extention='.png', test_index=[], recursive=True):
    user_paths = glob.glob(location+'/**', recursive=False)
    user_images = []
    for path in user_paths:
        user_im_path = glob.glob(path+'/*'+extention, recursive=recursive)
        loaded_list = list_loader(user_im_path)
        user_images.append(loaded_list)
    
    enrollment_sets = []
    testing_sets = []
    
    for user in user_images:
        enroll = []
        testing = []
        for i in range(len(user)):
            if i in test_index:
                testing.append(user[i])
            else:
                enroll.append(user[i])
        enrollment_sets.append(enroll)
        testing_sets.append(testing)
    
    return (enrollment_sets, testing_sets)

def test_setup():
    user_im = loader_test_images(testing_faces_path, test_index=[5])
    
    nb_comp = 200
    pca_name = 'PCA/pca' + str(nb_comp)
    
    pcaExt = PCA_ext()
    
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, nb_comp)
        pcaExt.save_PCA(pca_name)
    
def observation_to_string( tuple ):
    #tuple is (score, true_match, userID, testID, testImageID)
    ob_string = f'{tuple[0]}, {tuple[1]}, {tuple[2]}, {tuple[3]}.{tuple[4]}\n'
    return ob_string
    
def get_date_string():
    now = datetime.datetime.now().timetuple()
    s = f'{now[0]}-{now[1]}-{now[2]} {now[3]}-{now[4]}-{now[5]}'
    return s
        
    
def test_pca_roc( pca_features_nb, test_indices, saveEER=False ):
    user_im = loader_test_images(testing_faces_path, test_index=test_indices)
    
    enrollment_sets = user_im[0]
    testing_sets = user_im[1]
    
    number_of_users = len(enrollment_sets)
    
    pca_name = 'PCA/pca' + str(pca_features_nb)
    
    pcaExt = PCA_ext()
    
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, pca_features_nb)
        pcaExt.save_PCA(pca_name)
    
    codebook_name = "Codebooks/user_codebook_c" + str(pca_features_nb)
    global_codebook = GlobalCodebook(pca_features_nb, 3)
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        global_codebook.load_codebook(codebook_name)
    else:
        global_codebook.save_codebook(codebook_name)
    
    #global_codebook.set_startpoints( [0,0,0,0,0,0,0,0,0,0] )
    
    testname = f'PCA_ROC_c{pca_features_nb}_testImages_'
    for index in test_indices:
        testname += str(index)
    
    testname_date = testname +' '+ get_date_string()
    
    obs_list = []
        
    for i in range(number_of_users):
        user = UserData()
        user.new_user(enrollment_sets[i], global_codebook, pcaExt, noRP=True)
        
        #if (i == 0):
        #    print(f'User {i} has user codebook : \n {user.user_codebook.starting_point} \n {user.user_codebook.steps}')
        
        for j in range(number_of_users):
            testim = testing_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(len(testim)):
                score = user.authenticate_ROC(testim[k])
                
                obs_list.append( (score, truthval, i, j, k) )
                #print( score, truthval, i, j, k )
                
    obs_list.sort(key=itemgetter(0))
    
    total_size = len(obs_list)
    total_true = number_of_users * len(test_indices)
    
    #total_true = 26*6
    total_false = total_size - total_true
    
    tpr = 0
    fpr = 0
    
    current_true = 0
    minDistToEER = 1.0
    eer_index = 0
    
    fpr_at_min = 0
                    
    with open(output_directory+'/ROCs/'+testname_date+'.txt', 'a') as file:
        
        file.write('Testing Improvements on Sutcu : PCA, no RP \n')
        file.write('Score, True_match, userID, testID.testImageID\n')
        
        for obs in obs_list:
            file.write( observation_to_string(obs) )
            
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
    
    eer_string = f'PCA, {pca_features_nb}, {test_indices}, {fpr_at_min} \n'
    
    if saveEER :
        with open(output_directory+'/SavedEER_pca_2.txt', 'a') as file:
            file.write( eer_string )
    
    
    print( result_string)
    
def test_pcaRP_roc( pca_features_nb, test_indices, saveEER=False ):
    user_im = loader_test_images(testing_faces_path, test_index=test_indices)
    
    enrollment_sets = user_im[0]
    testing_sets = user_im[1]
    
    number_of_users = len(enrollment_sets)
    
    pca_name = 'PCA/pca' + str(pca_features_nb)
    
    pcaExt = PCA_ext()
    
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, pca_features_nb)
        pcaExt.save_PCA(pca_name)
        
    final_nb_components = int(pca_features_nb/2)
    
    codebook_name = "Codebooks/user_codebook_c" + str(final_nb_components)
    global_codebook = GlobalCodebook(final_nb_components, 3)
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        global_codebook.load_codebook(codebook_name)
    else:
        global_codebook.save_codebook(codebook_name)
    
    #global_codebook.set_startpoints( [0,0,0,0,0,0,0,0,0,0] )
    
    testname = f'PCARP_ROC_c{final_nb_components}_testImages_'
    for index in test_indices:
        testname += str(index)
    
    testname_date = testname +' '+ get_date_string()
    
    obs_list = []
        
    for i in range(number_of_users):
        user = UserData()
        user.new_user(enrollment_sets[i], global_codebook, pcaExt, noRP=False)
        
        #if (i == 0):
        #    print(f'User {i} has user codebook : \n {user.user_codebook.starting_point} \n {user.user_codebook.steps}')
        
        for j in range(number_of_users):
            testim = testing_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(len(testim)):
                score = user.authenticate_ROC(testim[k])
                
                obs_list.append( (score, truthval, i, j, k) )
                #print( score, truthval, i, j, k )
                
    obs_list.sort(key=itemgetter(0))
    
    total_size = len(obs_list)
    total_true = number_of_users * len(test_indices)
    
    #total_true = 26*6
    total_false = total_size - total_true
    
    tpr = 0
    fpr = 0
    
    current_true = 0
    minDistToEER = 1.0
    eer_index = 0
    
    fpr_at_min = 0
                    
    with open(output_directory+'/ROCs/'+testname_date+'.txt', 'a') as file:
        
        file.write('Testing Improvements on Sutcu : PCARP \n')
        file.write('Score, True_match, userID, testID.testImageID\n')
        
        for obs in obs_list:
            file.write( observation_to_string(obs) )
            
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
    
    eer_string = f'PCARP, {final_nb_components}, {test_indices}, {fpr_at_min} \n'
    
    if saveEER :
        with open(output_directory+'/SavedEER.txt', 'a') as file:
            file.write( eer_string )
    
    print( result_string)
    

def test_RP_RP_roc( pca_features_nb, test_indices, saveEER=False ):
    #training_images = directory_loader(training_faces_path)
    user_im = loader_test_images(testing_faces_path, test_index=test_indices)
    
    enrollment_sets = user_im[0]
    testing_sets = user_im[1]
    
    number_of_users = len(enrollment_sets)
        
    final_nb_components = int(pca_features_nb/2)
    
    codebook_name = "Codebooks/user_codebook_c" + str(final_nb_components)
    global_codebook = GlobalCodebook(final_nb_components, 0.3)
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        global_codebook.load_codebook(codebook_name)
    else:
        global_codebook.save_codebook(codebook_name)
    
    testname = f'RpRP_ROC_c{final_nb_components}_testImages_'
    for index in test_indices:
        testname += str(index)
    
    testname_date = testname +' '+ get_date_string()
    
    rpMain = RP_ext(user_im[0][0][0].size, nb_components_pca)
    
    obs_list = []
        
    for i in range(number_of_users):
        user = UserData()
        user.new_user(enrollment_sets[i], global_codebook, rpMain, noRP=False)
        
        #if (i == 0):
        #    print(f'User {i} has user codebook : \n {user.user_codebook.starting_point} \n {user.user_codebook.steps}')
        
        for j in range(number_of_users):
            testim = testing_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(len(testim)):
                score = user.authenticate_ROC(testim[k])
                
                obs_list.append( (score, truthval, i, j, k) )
                #print( score, truthval, i, j, k )
                
    obs_list.sort(key=itemgetter(0))
    
    total_size = len(obs_list)
    total_true = number_of_users * len(test_indices)
    
    #total_true = 26*6
    total_false = total_size - total_true
    
    tpr = 0
    fpr = 0
    
    current_true = 0
    minDistToEER = 1.0
    eer_index = 0
    
    fpr_at_min = 0
                    
    with open(output_directory+'/ROCs/'+testname_date+'.txt', 'a') as file:
        
        file.write('Testing Improvements on Sutcu : RP-RP \n')
        file.write('Score, True_match, userID, testID.testImageID\n')
        
        for obs in obs_list:
            file.write( observation_to_string(obs) )
            
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
    
    eer_string = f'RPRP, {final_nb_components}, {test_indices}, {fpr_at_min} \n'
    
    if saveEER :
        with open(output_directory+'/SavedEER.txt', 'a') as file:
            file.write( eer_string )
    
    print( result_string)


def small_loop_test():
    for indices in [ [0] ]:#, [1], [2], [3], [4], [5], [6], [3,4],[4,5], [5,6] ]:
        for i in [10, 20, 30, 40, 50, 75, 100, 200]: #For PCA
            test_pca_roc( i, indices, False)
            
        #for i in [20, 30, 40, 50, 75, 100]: #For PCARP
        #    for k in range(5):
        #        #test_pcaRP_roc( 2*i, indices, True)
        #        test_RP_RP_roc( 2*i, indices, True)
    print('\nDone!')
    


#Testing fucntions to get a confusion matrix to compare with LDLC fuzzy extractors

def test_pcaRP_confusion_matrix( final_nb_components, test_indices, min_tp, max_tp, saveEER=False ):
    user_im = loader_test_images(testing_faces_path, test_index=test_indices)
    
    enrollment_sets = user_im[0]
    testing_sets = user_im[1]
    
    number_of_users = len(enrollment_sets)
    
    pca_features_nb = int(final_nb_components*2)
    #pca_features_nb = int(final_nb_components)
    
    pca_name = 'PCA/pca' + str(pca_features_nb)
    """
    pcaExt = PCA_ext()
    
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, pca_features_nb)
        pcaExt.save_PCA(pca_name)
    """
    #RPRP
    pcaExt = RP_ext(user_im[0][0][0].size, pca_features_nb)
    #"""
    #final_nb_components = int(pca_features_nb/2)
    
    codebook_name = "Codebooks/user_codebook_c" + str(final_nb_components)
    global_codebook = GlobalCodebook(final_nb_components, 0.5)
    """
    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        global_codebook.load_codebook(codebook_name)
    else:
        global_codebook.save_codebook(codebook_name)
    """
    #global_codebook.set_startpoints( [0,0,0,0,0,0,0,0,0,0] )
    
    testname = f'RPRP_ConfusionMatrix_c{final_nb_components}_testImages_'
    for index in test_indices:
        testname += str(index)
    
    testname_date = testname +' '+ get_date_string()
    
    obs_list = []
        
    for i in range(number_of_users):
        user = UserData()
        user.new_user(enrollment_sets[i], global_codebook, pcaExt, noRP=False)
        
        #if (i == 0):
        #    print(f'User {i} has user codebook : \n {user.user_codebook.starting_point} \n {user.user_codebook.steps}')
        
        for j in range(number_of_users):
            testim = testing_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1
            for k in range(len(testim)):
                score = user.authenticate_ROC(testim[k])
                
                obs_list.append( (score, truthval, i, j, k) )
                #print( score, truthval, i, j, k )
                
    obs_list.sort(key=itemgetter(0))
    
    total_size = len(obs_list)
    total_true = number_of_users * len(test_indices)
    
    #total_true = 26*6
    total_false = total_size - total_true
    
    print(f'Test {testname}:')
    
    for tp in range(min_tp, max_tp+1):
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
            
        res_string = f'{curTP}, {curFN}, {curFP}, {curTN}, {score}'
        print(res_string)
    
    #print('\nDone!')



def test_ranges( final_nb_components, test_indices=[5], fuzz="PCA", noRP=False):
    user_im = loader_test_images(testing_faces_path, test_index=test_indices)
    
    enrollment_sets = user_im[0]
    testing_sets = user_im[1]
    
    number_of_users = len(enrollment_sets)
    
    if noRP:
        pca_features_nb = final_nb_components
    else:
        pca_features_nb = int(final_nb_components*2)
    
    mainExt = None
    
    if fuzz=="PCA":
        pca_name = 'PCA/pca' + str(pca_features_nb)
        
        mainExt = PCA_ext()
        
        file = f'{output_directory}/{pca_name}.joblib'
        if os.path.exists(file):
            mainExt.load_PCA(pca_name)
        else:
            training_images = directory_loader(training_faces_path)
            mainExt.train_PCA(training_images, pca_features_nb)
            mainExt.save_PCA(pca_name)
    
    elif fuzz=="RP":
        mainExt = RP_ext(user_im[0][0][0].size, pca_features_nb)

    global_codebook = GlobalCodebook(final_nb_components, 3)

    minRanges = None
    
    for i in range(number_of_users):
        user = UserData()
        user.new_user(enrollment_sets[i], global_codebook, mainExt, noRP=False)

        if (i==0):
            minRanges = user.ranges
        else:
            minRanges = np.amin( ( minRanges,user.ranges ),0 )
    
    smallest_delta = np.amin(minRanges)
    
    rp=None
    if (not noRP):
        rp = "RP"
    print(minRanges)
    print( f'For {fuzz}{rp} with n={final_nb_components} the min delta is: \n{smallest_delta}')



def test_pca_roc_loop(complist, saveEER=True):
    
    for comp in complist:
        for indices in testim_1:
            test_pca_roc(comp, indices, saveEER)
            
        for indices in testim_2:
            test_pca_roc(comp, indices, saveEER)
    
    print("Done!")
        
    







