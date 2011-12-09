import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys
import math
import timeit
import copy
import time
import struct
import scipy.stats.mstats as stats
import ConfigParser
import os
import getopt
import h5py
import random as rnd
import cPickle as pickle
import operator

from em import *

def get_song_dict():
    fileList = []
    rootdir = '/disk1/home_user/egonina/asp/MSD/MillionSongSubset/data/'
    for root, subFolders, files in os.walk(rootdir):
        for file in files:
            fileList.append(os.path.join(root,file))

    file_tag_dict = {}
    for file in fileList:
        print file

        f = h5py.File(file, 'r')
        mbtags = f['musicbrainz']['artist_mbtags']
        list = []
        for t in mbtags:
            list.append(t)
        tags = f['metadata']['artist_terms']
        tag_freq = f['metadata']['artist_terms_freq']
        tags_dict = {}
        for t in range(len(tags)):
            tags_dict[tags[t]] = tag_freq[t]

        file_id = str(f['analysis']['songs']['track_id'][0])
        file_tag_dict[file_id] = {}
        file_tag_dict[file_id]['artist_mbtags'] = list
        file_tag_dict[file_id]['artist_terms'] = tags_dict
        file_tag_dict[file_id]['artist_name'] = str(f['metadata']['songs']['artist_name'][0])
        file_tag_dict[file_id]['title'] = str(f['metadata']['songs']['title'][0])
        file_tag_dict[file_id]['segments_timbre'] = np.array(f['analysis']['segments_timbre'], dtype=np.float32)
        file_tag_dict[file_id]['duration'] = float(f['analysis']['songs']['duration'][0])
        file_tag_dict[file_id]['tempo'] = float(f['analysis']['songs']['tempo'][0])
        file_tag_dict[file_id]['time_signature'] = float(f['analysis']['songs']['time_signature'][0])
        file_tag_dict[file_id]['segments_start'] = np.array(f['analysis']['segments_start'], dtype=np.float32)
        f.close()

    p = open("/disk1/home_user/egonina/asp/MSD/all_file_dict_dump.pkl", "wb")
    pickle.dump(file_tag_dict, p, True)
    p.close()
    return file_tag_dict



def count_songs_by_tag(tags_file_name, output_file_name, fileDict):

    tags_file = open(tags_file_name, 'r')
    tag_dict = {}
    for tag in tags_file:
        tag = tag[:len(tag)-1] #delete end-of-line characater
        tag_dict[tag] = 0

        #---------- READ FILES -----------
        start = time.time()
                
        for file in fileDict.keys():

            tags = fileDict[file]
            if tag in tags:
                tag_dict[tag]+=1

        total = time.time() - start
        print "songs with keyword [" + tag + "]: "+ str(tag_dict[tag])
        print "total time: ", total
        
    tag_out = open(output_file_name, 'w')

    for tag in tag_dict.keys():
        tag_out.write(tag+"\t"+str(tag_dict[tag])+"\n")

    tag_out.close()


if __name__ == '__main__':

    total_start_time = time.time()

    freq_threshold = 0.8
    M = 32
    category_tag = "metal"
    rnd.seed(42)
    
    print "Reading Files"
    #song_dict = get_song_dict()
    st = time.time()

    # assume the dictionary has been already read in and pickled
    p = open("/disk1/home_user/egonina/asp/MSD/all_file_dict_dump.pkl", "rb")
    song_dict = pickle.load(p)
    p.close()
    print "--- File Reading:\t", time.time() - st, " -----"
    
    st = time.time()

    # collect songs
    songs_with_tag = {}
    songs_without_tag = {}
    song_with_tag_count = 0
    song_without_tag_count = 0
    for song in song_dict.keys():
        if category_tag in song_dict[song]['artist_terms'].keys(): #the song's tag list contains the tag we're looking for
            if song_dict[song]['artist_terms'][category_tag] > freq_threshold:
                songs_with_tag[song] = song_dict[song]
                song_with_tag_count += 1
        else:
            songs_without_tag[song] = song_dict[song]
            song_without_tag_count += 1

    print "--- Collecting songs for the tag time:\t", time.time() - st, " ----- "
    print "INFO: songs with tag count:", song_with_tag_count
    print "INFO: songs without tag count: ", song_without_tag_count

    st = time.time()
    
    # get indices for various sets of songs
    all_positive_indices = range(song_with_tag_count-1)
    all_negative_indices = range(song_without_tag_count-1)
    all_indices = range(len(song_dict.keys()))

    #split songs with tag into training/testing sets (70/30)
    training_sample_indices = np.array(rnd.sample(all_positive_indices, int(song_with_tag_count*0.7)))
    testing_sample_indices = np.delete(all_positive_indices, training_sample_indices)
    negative_sample_indices = all_negative_indices

    print "INFO: number of training indices:", len(training_sample_indices)
    print "INFO: testing indices:", len(testing_sample_indices)
    print "INFO: negative testing indices:", len(negative_sample_indices)
    
    # get song keys for the:
    # - 70% of total songs for training
    # - 30% of total songs for testing
    # - (total songs - songs with tag) for negative testing
    # - 30% of all song features for UBM model
    song_keys = np.array(songs_with_tag.keys())
    song_neg_keys = np.array(songs_without_tag.keys())
    all_song_keys = np.array(song_dict.keys())

    # get the corresponding song keys for each of the sets
    training_song_keys = song_keys[training_sample_indices]
    testing_song_keys = song_keys[testing_sample_indices]
    negative_song_keys = song_neg_keys[negative_sample_indices]
                                   
    # collect features for positive GMM training
    first_song = True
    for song in training_song_keys:
        feats = songs_with_tag[song]['segments_timbre']            
        
        if first_song:
            total_features = feats
            first_song = False
        else:
            total_features = np.concatenate((total_features, feats))
        
    print "--- Collecting training features time:\t", time.time() - st, " ----- "
    print "INFO: total features: ", total_features.shape

    # collect features for UBM training
    st = time.time()
    p = open("/disk1/home_user/egonina/asp/MSD/ubm_features_all.pkl", "rb")
    total_ubm_features = np.array(pickle.load(p))
    p.close()

    # train the UBM on 30% of the total features from all songs
    training_ubm_features = np.array(rnd.sample(total_ubm_features, int(len(total_ubm_features)*0.3)))

    print "--- Collecting ubm features time:\t", time.time() - st, " -----"
    print "INFO: total ubm features: ", total_ubm_features.shape, " 30%: ", training_ubm_features.shape

    # train UBM on features
    D = total_ubm_features.shape[1]
    ubm = GMM(M,D)
    
    train_st = time.time()
    ubm.train(training_ubm_features)
    train_total = time.time() - train_st
    print "--- UBM training time:\t", train_total, " -----"
    
    # train positive GMM on features
    D = total_features.shape[1]
    gmm = GMM(M, D, means=np.array(ubm.components.means), covars=np.array(ubm.components.covars), weights=np.array(ubm.components.weights))
    
    train_st = time.time()
    gmm.train(total_features)
    train_total = time.time() - train_st
    print "--- GMM training time:\t", train_total, " -----"

    print "--- Testing Labeled Examples ---"

    # testing the labeled test files
    test_st = time.time()
    labeled_songs = {}
    unlabeled_songs = {}

    for test_song in testing_song_keys:
        test_feats = songs_with_tag[test_song]['segments_timbre']
        all_lklds = gmm.score(test_feats)
        all_ubm_lklds = ubm.score(test_feats)
        
        avg_lkld = np.average(all_lklds)
        avg_ubm_lkld = np.average(all_ubm_lklds)
        sum_lkld = np.sum(all_lklds)
    
        labeled_songs[str(songs_with_tag[test_song]['artist_name']+ " - "+songs_with_tag[test_song]['title'])] = (avg_lkld, avg_ubm_lkld, avg_lkld - avg_ubm_lkld)
    
    print "--- Testing Unlabeled Examples ---"
    test_st = time.time()

    # testing the unlabeled test files
    for test_song in negative_song_keys:
        test_feats = songs_without_tag[test_song]['segments_timbre']

        all_lklds = gmm.score(test_feats)
        all_ubm_lklds = ubm.score(test_feats)
        avg_lkld = np.average(all_lklds)
        avg_ubm_lkld = np.average(all_ubm_lklds)
        sum_lkld = np.sum(all_lklds)
        
        unlabeled_songs[str(songs_without_tag[test_song]['artist_name'] + " - " + songs_without_tag[test_song]['title'])] = (avg_lkld, avg_ubm_lkld, avg_lkld - avg_ubm_lkld)

    test_total = time.time() - test_st
    print "--- Total testing time:\t", test_total, " -----"

    #print out top 20 labeled suggestions and unlabeled recommendations
    print "======================================================================"
    print "=================== TOP 20 LABELED SAMPLES ==========================="
    print "======================================================================"
    sorted_lab_samples = sorted(labeled_songs.iteritems(), key=lambda k: k[1][2], reverse=True)
    for p in range(20):
        print sorted_lab_samples[p]
        
    print "======================================================================"
    print "=================== TOP 20 UNLABELED SAMPLES ========================="
    print "======================================================================"
    sorted_unlab_samples = sorted(unlabeled_songs.iteritems(), key=lambda k: k[1][2], reverse=True)
    for p in range(20):
        print sorted_unlab_samples[p]


    print "-------------- DONE ---------------"
    print "--- Total time: ", time.time() - total_start_time, " ---"
    print "-----------------------------------"

        

