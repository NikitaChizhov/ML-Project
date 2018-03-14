import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data as td

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

import tqdm
import time

class ToxicTextsDataset(td.Dataset):
    def __init__(self, data_path='train.csv', 
                       n_train_batches=16000, 
                       n_test_batches=4000,
                       n_valid_batches=1600,
                       separate_test_and_valid=True,
                       test_size=0.2,
                       valid_size=0.1,
                       batch_size=10, 
                       vocab_size=2000,
                       mode='train',
                       random_seed=None,
                       verbose=0,
                       use_cuda = True):
        """
        INPUT:
            n_train_batches - int, number of batches to be drawn from data for training
            n_test_batches -  int, number of batches to be drawn from data for testing
            n_valid_batches -  int, number of batches to be drawn from data for validation
            separate_test_and_valid - bool, wherever to draw training, testing and validation 
                                      from all data or from separated parts of data (a chance 
                                      of intersection between training, testing and validation 
                                      data if False)
            test_size - float from [0, 1], a portion of initial data reserved for creating 
                        dataset for testing. Not aplicable if separate_test_and_valid=False
            valid_size - float from [0, 1], a portion of initial data reserved for creating 
                         dataset for validation. Not aplicable if separate_test_and_valid=False
            batch_size - int, number of samples in one minibatch
            vocab_size - int, number of unique tokens to save and embed. Saved [vocab_size] 
                         most frequently encountered tokens, all others will be encoded as 
                         UNKNOWN token
            mode = string, one from ['train', 'test', 'valid']. Determinedes from which dataset 
                    will be returned sample on ToxicTextsDataset[i]
            verbose - int, 0 for no printed info, 1 for minimum info, 2 for maximum info
            
        """
        super(ToxicTextsDataset, self).__init__()
        
        self.n_train_batches = n_train_batches
        self.n_test_batches = n_test_batches
        self.n_valid_batches = n_valid_batches
        self.separate_test_and_valid = separate_test_and_valid
        self.test_size = test_size
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.mode = mode
        self.verbose = verbose
        self.use_cuda = use_cuda
        
        if(random_seed != None):
            np.random.seed(random_seed)
        
        if(verbose): print('Downloading data from ' + data_path + '... ', end='')
        # read csv file
        df = pd.read_csv(data_path)
        if(verbose): print('Completed')
        
        # separate text from class labels
        X = np.array(df.iloc[:, 1])
        y = np.array(df.iloc[:, 2:])
        
        if(verbose): print('Generating vocabulary... ', end='')
        # generating vocabulary of tokens
        self.CreateTokenVocab(X, y)
        if(verbose): print('Completed')
        
        if(separate_test_and_valid == True):
            # split data for
            X_train, X, y_train, y = train_test_split(X, y, test_size=valid_size + test_size)
            
            if(verbose): print('Creating train dataset... ', end='')
            self.train_dataset = self.CreateBalancedDataset(X_train, y_train, n_train_batches)
            if(verbose): print('Completed')
            
            if(test_size != 0 and valid_size != 0):
                X_test, X_valid, y_test, y_valid = train_test_split(X, y, 
                                                    test_size=valid_size/(test_size+valid_size))
                
                if(verbose): print('Creating test dataset... ', end='')
                self.test_dataset = self.CreateBalancedDataset(X_test, y_test, n_test_batches)
                if(verbose): print('Completed')
                if(verbose): print('Creating validation dataset... ', end='')
                self.valid_dataset = self.CreateBalancedDataset(X_valid, y_valid, n_valid_batches)
                if(verbose): print('Completed')
                    
            elif(test_size == 0):
                X_valid = X
                y_valid = y
                
                if(verbose): print('Creating validation dataset... ', end='')
                self.valid_dataset = self.CreateBalancedDataset(X_valid, y_valid, n_valid_batches)
                if(verbose): print('Completed')
                
                self.test_dataset = []              
                    
            elif(valid_size == 0):
                X_test = X
                y_test = y
                
                if(verbose): print('Creating test dataset... ', end='')
                self.test_dataset = self.CreateBalancedDataset(X_test, y_test, n_test_batches)
                if(verbose): print('Completed')
                
                self.valid_dataset = []            
                
        elif(separate_test_and_valid == False):
            
            if(verbose): print('Creating train dataset... ', end='')
            self.train_dataset = self.CreateBalancedDataset(X, y, n_train_batches)
            if(verbose): print('Completed')
            
            if(verbose): print('Creating test dataset... ', end='')
            self.test_dataset = self.CreateBalancedDataset(X, y, n_test_batches)
            if(verbose): print('Completed')
            
            if(verbose): print('Creating validation dataset... ', end='')
            self.valid_dataset = self.CreateBalancedDataset(X, y, n_valid_batches)
            if(verbose): print('Completed')
                    
        
    def encode(self, text):
        """ function that splits text into tokens and returns a list of encodings for 
            each token 
                INPUT: text - python string
                OUTPUT: codes - list of integers, 
                        cl_features - list of floats (character level features)
        """
        tokens = self.Smart_Split(text)
        codes = []
        cl_features = self.ComputeCharacterLevelFeatures(text)
        for token in tokens:
            if(self.word_to_id.get(token) != None):
                codes.append(self.word_to_id[token])
            else:
                codes.append(self.vocab_size - 1) # UNKNOWN token
        return codes, cl_features
    
    def ComputeCharacterLevelFeatures(self, text):
        """This function computes a character level features 
           INPUT: text - a python string
           OUTPUT: cl_features - a list of floats
               
               cl_features[0] - lenght of text
               cl_features[1] - mean of lenghts of all tokens in text
               cl_features[2] - ratio of capital letters in text
               cl_features[3] - ratio of non-letter symbols in text
        """
        text_len = float(len(text))
        
        cl_features = [
            text_len,
            np.mean([len(token) for token in self.Smart_Split(text)]),
            len(re.findall(r'[A-Z]', text)) / text_len,
            (1. - len(re.findall(r'[a-zA-Z]', text)) / text_len)
        ]
        
        return cl_features
    
    def CreateBalancedDataset(self, X, y, n_batches):
        """This functions returns a balanced dataset (a list of batched samples with 
           corresponding labels). Produced dataset is drawn with repetition from initial data, 
           and therefore can contain duplicates Depending on n_batches, it will do either 
           undersampling, oversampling or combination of both
        
          INPUT: X - one dimensional np.array of shappe (n_samples, ) with unparsed text 
                     as elements
                 y - two dimensional np.array of shape (n_samples, n_labels) with 
                     classification labels (label != 0 is assumed to be "interesting" )
                 n_batches - integer, number of batches in dataset (so the number of samples 
                             in dataset is equal to n_batches * batch_size = len(dataset) * batch_size)
          OUTPUT:
                  dataset - list of dictionaries where dataset[i]['input'] is a i-th batch 
                            of inputs and dataset[i]['labels'] - corresponding batch of labels"""
        dataset = []
        masks = self.MakeMasks(y)
        n_subbatches = n_batches // len(masks)
        
        if(self.verbose >= 2): print('\n')
        
        for mask in masks:
            if(self.verbose >= 2): print('\tApplying mask: ' + mask['name'] + '... ', end='')
            dataset += self.CreateDatasetFromXY(X[mask['mask']], y[mask['mask']], n_subbatches)
            if(self.verbose >= 2): print('Completed')
                
        return shuffle(dataset)
    
    def CreateDatasetFromXY(self, X, y, n_batches):
        """
        This functions constructs and returns a dataset (a list of batched samples 
        with corresponding labels). 
        
          INPUT: X - one dimensional np.array of shappe (n_samples, ) with unparsed 
                     text as elements
                 y - two dimensional np.array of shape (n_samples, n_labels) with 
                     classification labels
                 n_batches - integer, number of batches in dataset (so the number 
                             of samples in dataset is equal to n_batches * batch_size = 
                             len(dataset) * batch_size)
          OUTPUT:
                  dataset - list of dictionaries where dataset[i]['input'] is a i-th 
                            batch of inputs and dataset[i]['labels'] - corresponding 
                            batch of labels
        
        """
        # we sort our samples on the lenght of the text (in the number of tokens) and 
        # place texts of the same lenght in the same position in this dictionary. 
        # This can be also viewed as a hash-table
        Len_table = dict()
        for i in range(len(X)):
            codes, cl_features = self.encode(X[i])
            if(Len_table.get(len(codes)) != None):
                Len_table[len(codes)].append((codes, cl_features, y[i]))
            else: 
                Len_table[len(codes)] = [(codes, cl_features, y[i])]
        
        # we have different number of samples of different lenght. There is a lot more 
        # samples of lenght ~10-50 tokens and much smaller number of samples of lenght 
        # 100+ tokens. Now we will get a distribution of number of samples:
        dist = np.array([[i, len(Len_table[i])] for i in Len_table.keys()])
        # here dist[i, 0] is some lenght of sample we encountered in dataset
        # and dist[i, 1] is a number of samples of that lenght 
        
        p = dist[:, 1] / np.sum(dist[:, 1])
        
        # we will construct actual dataset, randomly drawing samples from that distribution:
        dataset = []
        for _ in range(n_batches):
            i = np.random.choice(dist[:, 0], p=p)
            sample_indices = np.random.randint(0, len(Len_table[i]), self.batch_size)
            # it took me some time to figure out correct transformation from mess of 
            # lists and numpy array to torch tensor :)
            if(self.use_cuda):
                batch = {'input':Variable(torch.LongTensor(
                    np.array(np.array(Len_table[i])[sample_indices][:, 0].tolist())), 
                    requires_grad=False).cuda(),
                         'cl_features':Variable(torch.FloatTensor(
                    np.array(np.array(Len_table[i])[sample_indices][:, 1].tolist())), 
                    requires_grad=False).cuda(),
                         'labels':Variable(torch.FloatTensor(
                    np.array(np.array(Len_table[i])[sample_indices][:, 2].tolist())), 
                    requires_grad=False).cuda()}
            else:
                batch = {'input':Variable(torch.LongTensor(
                    np.array(np.array(Len_table[i])[sample_indices][:, 0].tolist())), 
                    requires_grad=False),
                         'cl_features':Variable(torch.FloatTensor(
                    np.array(np.array(Len_table[i])[sample_indices][:, 1].tolist())), 
                    requires_grad=False),
                         'labels':Variable(torch.FloatTensor(
                    np.array(np.array(Len_table[i])[sample_indices][:, 2].tolist())), 
                    requires_grad=False)}
                
            dataset.append(batch)        
        
        return dataset
    
    def CreateTokenVocab(self, X, y):
        '''This function generates a word_to_id dictionary we use for encoding text
        
            INPUT: X - one dimensional np.array of shappe (n_samples, ) with unparsed 
                       text as elements
                   y - two dimensional np.array of shape (n_samples, n_labels) with 
                       classification labels (label != 0 is assumed to be "interesting" - 
                       we prioretize tokens encoundered in examples with at least one label = 1)
        
        '''
        token_freq = dict()

        # firstly we exctract all tokens we see in positivly labeled samples
        X_relevant = X[np.sum(y, axis=1) > 0] 
        X_relevant += shuffle(X[np.sum(y, axis=1) == 0])[:len(X_relevant)] 
        # we add random portion of "all-negative" data of equal size 
         
        for text in X_relevant:
            tokens = self.Smart_Split(text)

            for token in tokens:
                if(token_freq.get(token) == None):
                    token_freq[token] = 1
                else: token_freq[token] += 1

        tokens = sorted(token_freq, key=token_freq.get)[::-1]

        # secondly, we assign id's to the most frequently encountered tokens in positivly 
        # classified samples
        self.word_to_id = dict()
        for i in range(self.vocab_size - 1):
            self.word_to_id[tokens[i]] = i

        # finally, we would like to find very similar tokens and assign to them the 
        # same id (those are mainly misspells and parsing 
        # innacuracies. For example 'training', 'traning', 'trainnin', 'training"' and so on)
        vec = TfidfVectorizer()
        vec_tokens = vec.fit_transform(tokens)
        same_tokens = ((vec_tokens * vec_tokens.T) > 0.99)
        rows, cols = same_tokens.nonzero()

        for token_pair in zip(rows, cols):
            if(token_pair[0] > self.vocab_size):
                break
            if(token_pair[0] <= token_pair[1]):
                continue
            else:
                self.word_to_id[tokens[token_pair[1]]] = token_pair[0]
    
    def Smart_Split(self, text):
        """Parsing function 
            INPUT: text - python string with any text
            OUTPUT: list of strings, containing tokens
        """
        out = text.strip().lower().replace('\n', ' ')
        out = out.replace(',', ' , ').replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ')
        out = out.replace(')', ' ) ').replace('(', ' ( ').replace(':', ' : ').replace(';', ' ; ')
        return out.split()

    def MakeMasks(self, y):
        """this function makes masks (bool np.arrays of length y). Each mask is 
        cunstructed so that X[mask] is a part of data grouped by some combination 
        of labels (for example - all data with al labels = 0, or all data with
        first class label = 1 and all other equal to 0, or all data with all 
        labels equal to 1)
            INPUT: y - np.array of shape [n_samples, n_classes]
            OUTPUT: masks - list of bool np.arrays of length y
        """
        
        def not_i_col(y, i):
            """Utility function that returns all columns of y, except i-th"""
            mask = np.array([True, True, True, True, True, True])
            mask[i] = False
            return y[:, mask]

        # mask for data with label_excluded_i = 1 and all others = 0
        # important: there is no data for label_1 = 1 and all others equal to 0, 
        # so skipping that mask
        mask1 = []
        for excluded_i in range(6):
            mask1.append(np.logical_and(y[:, excluded_i] == 1, 
                                        np.sum(not_i_col(y, excluded_i), axis=1) == 0))

        # masks for 2, 3, 4, 5 and 6 labels respectivly equal to 1 (here we do not care, 
        # which label (i.e. label_1, label_2, ...) 
        # is equal to 1, just that there is exactly n=2,3,.. labels equal to 1)
        mask2 = np.sum(y, axis=1) == 2
        mask3 = np.sum(y, axis=1) == 3
        mask4 = np.sum(y, axis=1) == 4
        mask5 = np.sum(y, axis=1) == 5
        mask6 = np.sum(y, axis=1) == 6

        mask0 = (np.sum(y, axis=1) == 0)

        # let's save all masks in one list:
        masks = [{'mask':mask0, 'name':'all-negative data'}, 
                 {'mask':mask1[0], 'name':'only fisrt class labeled positive'},
                 {'mask':mask1[2], 'name':'only third class labeled positive'},
                 {'mask':mask1[3], 'name':'only fourth class labeled positive'},
                 {'mask':mask1[4], 'name':'only fifth class labeled positive'},
                 {'mask':mask1[5], 'name':'only sixth class labeled positive'},
                 {'mask':mask2, 'name':'exactly two positive labels'},
                 {'mask':mask3, 'name':'exactly three positive labels'},
                 {'mask':mask4, 'name':'exactly four positive labels'},
                 {'mask':mask5, 'name':'exactly five positive labels'},
                 {'mask':mask6, 'name':'all-positive data'}]
            
        if(self.verbose >= 2): print('\n\tMasks created (a reminder - no data for "only second class labeled positive")', end='')
        
        return masks
    
    def __getitem__(self, i):
        if(self.mode == 'train'):
            return self.train_dataset[i]
        elif(self.mode == 'test'):
            return self.test_dataset[i]
        elif(self.mode == 'valid'):
            return self.valid_dataset[i]
    
    def __len__(self):
        if(self.mode == 'train'):
            return len(self.train_dataset)
        elif(self.mode == 'test'):
            return len(self.test_dataset)
        elif(self.mode == 'valid'):
            return len(self.valid_dataset)

    def shuffle(self):
        """shuffles dataset, corresponding to current mode"""
        if(self.mode == 'train'):
            self.train_dataset = shuffle(self.train_dataset)
        elif(self.mode == 'test'):
            self.test_dataset = shuffle(self.test_dataset)
        elif(self.mode == 'valid'):
            self.valid_dataset = shuffle(self.valid_dataset)
        