"""
Contains the original Scheme by Sutcu et Al. for testing purposes.
Written by Renaud Brien.
"""

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
import Fuzzy_Extractor as FE

training_faces_path = 'C:/Coding/ARFaces/ARFacesTrain'    #path to images for training PCA
testing_faces_path = 'C:/Coding/ARFaces/ARFacesTest'      #path to images for testing extractor
output_directory = 'C:/Coding/PythonFuzzyOutputs'

complist = [20, 30, 40, 50, 75, 100]
testim_1 = [[0], [1], [2], [3], [4], [5], [6]]


def test_sutcu(number_of_components, test_indices, alpha=0.15, beta=1, theta=1):
    n = number_of_components
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)

    pca_name = 'PCA/pca' + str(n)
    pcaExt = FE.PCA_ext()

    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = FE.directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, n)
        pcaExt.save_PCA(pca_name)

    enrollment_sets = user_im[0]
    testing_sets = user_im[1]

    number_of_users = len(enrollment_sets)

    #Seed list for randomization matrix
    seeds = []
    for i in range(number_of_users):
        seeds.append(np.random.seed())

    #Getting Templates for all users
    user_midpoint = np.ndarray(shape=(number_of_users, n))
    user_ranges = np.ndarray(shape=(number_of_users, n))
    user_min = np.ndarray(shape=(number_of_users, n))
    user_max = np.ndarray(shape=(number_of_users, n))

    for i in range(number_of_users):
        gen = np.random.RandomState(seeds[i])
        #r_mat = gen.uniform(-theta, theta, (n,n)) #Original Randomization
        r_mat = np.eye(n) #For Testing with no randomization
        pca_features = pcaExt.extract_features(enrollment_sets[i])

        features = np.dot(r_mat, pca_features.transpose()).transpose()

        min_vec = np.amin(features, 0)
        max_vec = np.amax(features, 0)
        ranges = (max_vec - min_vec)/2
        midpoint = (max_vec + min_vec)/2

        user_midpoint[i] = midpoint.flatten()
        user_ranges[i] = ranges.flatten()
        user_min[i] = min_vec.flatten()
        user_max[i] = max_vec.flatten()

    #Generating Global Codebook:
    global_codebook = []
    min_ranges = np.amin(user_ranges, 0)*alpha
    global_min = np.amin(user_min, 0)
    global_max = np.amax(user_max, 0)

    gen2 = np.random.RandomState(0)
    for i in range(n):
        ri = gen2.uniform(0,10)

        comp_book = []
        current_val = global_min[i] - ri
        comp_book.append(current_val)

        while( current_val < global_max[i] ):
            current_val += min_ranges[i]
            comp_book.append(current_val)

        global_codebook.append(comp_book)

    #User Codebooks
    user_codebooks = []
    correct_codewords = []
    for i in range(number_of_users):
        uc_i = []

        q = quantize(user_midpoint[i], global_codebook)

        for j in range(n):
            step_num = int( (q[j]-global_codebook[j][0])/(global_codebook[j][1] -global_codebook[j][0]) )
            d_ij = int( math.ceil(beta*user_ranges[i][j]/min_ranges[j]) )
            steps = 2*d_ij + 1
            uc_ij = []

            #print(f'User {i}, component {j}: \n step_num = {step_num} \n d_ij = {d_ij} \n steps = {steps}')

            start = step_num

            while start > steps:
                start -= steps

            index = start
            cg_len = len(global_codebook[j])

            #print(f'start = {start} \n')

            while index < cg_len :
                uc_ij.append(global_codebook[j][index])
                index += steps

            uc_i.append(uc_ij)
        user_codebooks.append(uc_i)

        #Getting Correct Codewords
        correct = quantize(user_midpoint[i], uc_i)
        correct_codewords.append(correct)

    #Running EER Tests
    obs_list = []

    for i in range(number_of_users):

        correct = correct_codewords[i]
        codebook = user_codebooks[i]

        gen = np.random.RandomState(seeds[i])
        #r_mat = gen.uniform(-theta, theta, (n,n))
        r_mat = np.eye(n)

        for j in range(number_of_users):
            testim = testing_sets[j]
            #testim = enrollment_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1
            pca_features = pcaExt.extract_features(testim)
            features = np.dot(r_mat, pca_features.transpose()).transpose()
            for k in range(len(testim)):
                currFeatures = features[k]
                max_score = 0
                q = quantize(currFeatures, codebook)

                matched = 0
                if q == correct:
                    matched = 1

                for index in range(n):
                    dist = abs( q[index]-correct[index] )
                    #print(q[index], correct[index], dist)
                    dist = abs( (currFeatures[index] - correct[index] ) )
                    if len(codebook[index])==1:
                        score = 0
                    else:
                        score = dist / (codebook[index][1] - codebook[index][0])
                    if score > max_score:
                        max_score = score

                    #print(q[index], correct[index], score)

                #if max_score == 0:
                    #print(q, '\n', correct, '\n', q == correct)
                obs_list.append( (max_score, truthval, matched, i, j, k) )


    obs_list.sort(key=itemgetter(0))

    total_size = len(obs_list)
    total_true = number_of_users * len(test_indices)
    #total_true = number_of_users *(7 - len(test_indices))

    #total_true = 26*6
    total_false = total_size - total_true

    tpr = 0
    fpr = 0

    current_true = 0
    minDistToEER = 1.0
    eer_index = 0

    fpr_at_min = 0

    for obs in obs_list:

        current_true += obs[1]
        eer_index += 1
        tpr = current_true / total_true
        fpr = (eer_index - current_true) / total_false

        distToEER = abs(1.0 - tpr - fpr)
        if (distToEER < minDistToEER):
            fpr_at_min = fpr
            minDistToEER = distToEER

        #print(f'tpr: {tpr}, fpr: {fpr}, minDist: {minDistToEER}')

    testname = f'Sutcu_ROC_c{n}_testImages_'
    for index in test_indices:
        testname += str(index)

    result_string = f'Test {testname} returns an EER of \n {fpr_at_min}'

    #return global_codebook, user_codebooks, correct_codewords
    print(result_string)
    print(obs_list[0])

    eer_string = f'PCA, {n}, {alpha}, {test_indices}, {fpr_at_min} \n'

    if False :
        with open(output_directory+'/SavedEER_yagiz.txt', 'a') as file:
            file.write( eer_string )

    print(eer_string)
    return obs_list



def quantize(vector, codebook):
    quantized_value = []
    for i in range(len(codebook)):
        vecVal = vector[i]
        c_i = codebook[i]
        if vecVal < c_i[0]:
            quantized_value.append(c_i[0])
        elif vecVal > c_i[len(c_i)-1]:
            quantized_value.append(c_i[len(c_i)-1])
        else:
            code_dist = (c_i[1] - c_i[0]) #/2
            steps = int( math.floor( (vecVal-c_i[0])/code_dist ) )
            d1 = abs(vecVal - c_i[steps])
            d2 = abs(vecVal - c_i[steps+1])

            if d1 < d2 :
                quantized_value.append(c_i[steps])
            else:
                quantized_value.append(c_i[steps+1])

            #found = False
            #while not found:
            #    d = abs(vecVal - c_i[index])
            #    if d < code_dist:
            #        found = True
            #    else:
            #        index += 1
            #quantized_value.append(c_i[index])
    return quantized_value

"""
Section to test EER for a classical PCA measure
"""
#Nearest in Class EER
def test_pca_NiC(n, test_indices, rp=False):
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)

    if rp:
        pca_comp = n*2
    else:
        pca_comp = n

    pca_name = 'PCA/pca' + str(pca_comp)
    pcaExt = FE.PCA_ext()

    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = FE.directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, pca_comp)
        pcaExt.save_PCA(pca_name)

    enrollment_sets = user_im[0]
    testing_sets = user_im[1]

    number_of_users = len(enrollment_sets)

    #Seed list for randomization matrix
    seeds = []
    for i in range(number_of_users):
        seeds.append( (np.random.seed(),np.random.seed()) )

    user_features = []

    for i in range(number_of_users):
        features=None
        if rp:
            extractor = FE.FeRP_Ext(pcaExt, n, matSeed=seeds[i][0], trSeed=seeds[i][1])
            features = extractor.extract_features(enrollment_sets[i])
        else:
            features = pcaExt.extract_features(enrollment_sets[i])
        user_features.append(features)


    obs_list = []
    for i in range(number_of_users):
        feat_set = user_features[i]
        #print(feat_set.shape)
        for j in range(number_of_users):
            testim = testing_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1

            features=None
            if rp:
                extractor = FE.FeRP_Ext(pcaExt, n, matSeed=seeds[i][0], trSeed=seeds[i][1])
                features = extractor.extract_features(testim)
            else:
                features = pcaExt.extract_features(testim)

            for k in range(len(testim)):
                currFeatures = features[k]

                min_dist = np.linalg.norm(feat_set[0] - currFeatures)
                for f in range(1,feat_set.shape[0]):
                    c_dist = np.linalg.norm(feat_set[f] - currFeatures)
                    if c_dist < min_dist:
                        min_dist = c_dist

                obs_list.append( (min_dist, truthval, i, j, k) )

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

    for obs in obs_list:

        current_true += obs[1]
        eer_index += 1
        tpr = current_true / total_true
        fpr = (eer_index - current_true) / total_false

        distToEER = abs(1.0 - tpr - fpr)
        if (distToEER < minDistToEER):
            fpr_at_min = fpr
            minDistToEER = distToEER

        #print(f'tpr: {tpr}, fpr: {fpr}, minDist: {minDistToEER}')

    testname = f'PCA_NiC_ROC_c{n}_testImages_'
    if rp:
        testname = f'PCARP_NiC_ROC_c{n}_testImages_'

    for index in test_indices:
        testname += str(index)

    result_string = f'Test {testname} returns an EER of \n {fpr_at_min}'

    #return global_codebook, user_codebooks, correct_codewords
    print(result_string)

    return obs_list


def test_classical_pca_MidPoint(n, test_indices, rp=False):
    user_im = FE.loader_test_images(testing_faces_path, test_index=test_indices)

    if rp:
        pca_comp = n*2
    else:
        pca_comp = n

    pca_name = 'PCA/pca' + str(pca_comp)
    pcaExt = FE.PCA_ext()

    file = f'{output_directory}/{pca_name}.joblib'
    if os.path.exists(file):
        pcaExt.load_PCA(pca_name)
    else:
        training_images = FE.directory_loader(training_faces_path)
        pcaExt.train_PCA(training_images, pca_comp)
        pcaExt.save_PCA(pca_name)

    enrollment_sets = user_im[0]
    testing_sets = user_im[1]

    number_of_users = len(enrollment_sets)

    #Seed list for randomization matrix
    seeds = []
    for i in range(number_of_users):
        seeds.append( (np.random.seed(),np.random.seed()) )

    user_midpoints = []

    for i in range(number_of_users):
        features=None
        if rp:
            extractor = FE.FeRP_Ext(pcaExt, n, matSeed=seeds[i][0], trSeed=seeds[i][1])
            features = extractor.extract_features(enrollment_sets[i])
        else:
            features = pcaExt.extract_features(enrollment_sets[i])

        min_vec = np.amin(features, 0)
        max_vec = np.amax(features, 0)
        midpoint = (max_vec + min_vec)/2

        user_midpoints.append(midpoint)


    obs_list = []
    for i in range(number_of_users):
        midpoint = user_midpoints[i]
        #print(feat_set.shape)
        for j in range(number_of_users):
            testim = testing_sets[j]
            truthval = 0
            if (j == i):
                truthval = 1

            features=None
            if rp:
                extractor = FE.FeRP_Ext(pcaExt, n, matSeed=seeds[i][0], trSeed=seeds[i][1])
                features = extractor.extract_features(testim)
            else:
                features = pcaExt.extract_features(testim)

            for k in range(len(testim)):
                currFeatures = features[k]

                dist = np.linalg.norm(midpoint - currFeatures)

                obs_list.append( (dist, truthval, i, j, k) )

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

    for obs in obs_list:

        current_true += obs[1]
        eer_index += 1
        tpr = current_true / total_true
        fpr = (eer_index - current_true) / total_false

        distToEER = abs(1.0 - tpr - fpr)
        if (distToEER < minDistToEER):
            fpr_at_min = fpr
            minDistToEER = distToEER

        #print(f'tpr: {tpr}, fpr: {fpr}, minDist: {minDistToEER}')

    testname = f'PCA_Classical_Midpoint_ROC_c{n}_testImages_'
    if rp:
        testname = f'PCARP_Classical_Midpoint_ROC_c{n}_testImages_'
    for index in test_indices:
        testname += str(index)

    result_string = f'Test {testname} returns an EER of \n {fpr_at_min}'

    #return global_codebook, user_codebooks, correct_codewords
    print(result_string)

    return obs_list


















