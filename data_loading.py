import numpy as np
import pandas as pd
import string
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder


class AGnews(Dataset):
    def __init__(self, dataset,supergroups=False):
        self.x_data=dataset[1]+dataset[2]
        y_data=np.array(dataset[0])
        for i in range(len(y_data)):
            y_data[i] -= 1
        self.y_data=y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        #this is where the encoding happens
        x_data=self.encoded(index)
        y_data=self.y_data[index]
        return x_data,y_data

    def encoded(self,index):
        alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list('’') + list('\n')
        encoded_data=torch.zeros(70,1014)
        chars=self.x_data[index]
        for index, char in enumerate(chars[::-1]):
            if char in alphabet and index<1014:
                encoded_data[alphabet.index(char)][index]=1
        return encoded_data

class Newsgroups(Dataset):
    def __init__(self, dataset,supergroups=False):
        self.x_data = pd.Series(dataset.data)
        self.y_data = pd.Series(dataset.target)
        self.y_map={0:5,1:0,2:0,3:0,4:0,5:0,6:2,7:1,8:1,9:1,10:1,11:3,12:3,13:3,14:3,15:5,16:4,17:4,18:4,19:5}
        self.supergroups=supergroups

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        #this is where the encoding happens
        x_data=self.encoded(index)
        y_data=self.y_data[index]
        if self.supergroups:
            y_data=self.y_map[self.y_data[index]] #supergroups
        return x_data,y_data

    def encoded(self,index):
        alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list('’') + list('\n')
        encoded_data=torch.zeros(70,1014)
        chars=self.x_data[index]
        for index, char in enumerate(chars[::-1]):
            if char in alphabet and index<1014:
                encoded_data[alphabet.index(char)][index]=1
        return encoded_data

class I2b2(Dataset):
    def __init__(self, dataset,supergroups=False):
        self.x_data=dataset['TEXT']
        y_data=np.array(dataset['SMOKING/_STATUS'])
        self.y_data=y_data


    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        #this is where the encoding happens
        x_data=self.encoded(index)
        y_data=self.y_data[index]
        return x_data,y_data

    def encoded(self,index):
        alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list('’') + list('\n')
        encoded_data=torch.zeros(70,1014)
        chars=self.x_data[index]
        for index, char in enumerate(chars[::-1]):
            if char in alphabet and index<1014:
                encoded_data[alphabet.index(char)][index]=1
        return encoded_data

class Sentiment(Dataset):
    def __init__(self, dataset, supergroups=False):
        self.x_data=dataset[5]
        self.y_data=dataset[0]
        self.y_map={0:0,2:1,4:2}

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        #this is where the encoding happens
        x_data=self.encoded(index)
        y_data=self.y_map[self.y_data[index]]
        return x_data,y_data

    def encoded(self,index):
        alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list('’') + list('\n')
        encoded_data=torch.zeros(70,1014)
        chars=self.x_data[index]
        for index, char in enumerate(chars[::-1]):
            if char in alphabet and index<1014:
                encoded_data[alphabet.index(char)][index]=1
        return encoded_data

def load_data(dataset='<dataset>',transformation=None,n_train=None,n_test=None):


    if dataset=='20Newsgroups':
        train = fetch_20newsgroups(subset='train')
        test = fetch_20newsgroups(subset='test')
        if transformation=='supergroups':
            trainset = Newsgroups(train,supergroups=True)
            train_loader = DataLoader(dataset=trainset, batch_size=128, num_workers=0, drop_last=False)
            testset = Newsgroups(test,supergroups=True)
            test_loader = DataLoader(dataset=testset, batch_size=128, num_workers=0, drop_last=False)
        else:
            trainset = Newsgroups(train)
            train_loader = DataLoader(dataset=trainset, batch_size=128, num_workers=0, drop_last=False)
            testset = Newsgroups(test)
            test_loader = DataLoader(dataset=testset, batch_size=128, num_workers=0, drop_last=False)

        return train_loader, test_loader

    elif dataset=='i2b2':
        le=LabelEncoder()
        train=pd.read_csv('./smoker_train.csv')
        train['SMOKING/_STATUS']=le.fit_transform(train['SMOKING/_STATUS'])
        test=pd.read_csv('./smoker_test_labeled.csv')
        test['SMOKING/_STATUS'] = le.transform(test['SMOKING/_STATUS'])
        trainset=I2b2(train)
        train_loader = DataLoader(dataset=trainset, batch_size=128, num_workers=0, drop_last=False)
        testset=I2b2(test)
        test_loader = DataLoader(dataset=testset, batch_size=128, num_workers=0, drop_last=False)

        return train_loader,test_loader
    elif dataset=='sentiment140':
        train=pd.read_csv('sentiment140_training.csv',header=None,encoding='latin-1') #1600000x6
        test=pd.read_csv('sentiment140_testing.csv',header=None) #498x6

        trainset=Sentiment(train)
        train_loader = DataLoader(dataset=trainset, batch_size=128, num_workers=0, drop_last=False,shuffle=True)
        testset=Sentiment(test)
        test_loader = DataLoader(dataset=testset, batch_size=128, num_workers=0, drop_last=False,shuffle=True)

        return train_loader,test_loader
    else : #if dataset=='AGNews': AGNews default
        train = pd.read_csv('./AGnews_train.csv',header=None)  # 120000x3
        test = pd.read_csv('./AGnews_test.csv',header=None)  # 7600x3
        trainset = AGnews(train)
        train_loader = DataLoader(dataset=trainset, batch_size=128, num_workers=0, drop_last=False)
        testset = AGnews(test)
        test_loader = DataLoader(dataset=testset, batch_size=128, num_workers=0, drop_last=False)

        return train_loader, test_loader

