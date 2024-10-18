 

from collections import defaultdict
import pandas as pd
import numpy as np


from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression, Ridge, Lasso, LassoCV, ElasticNet, ElasticNetCV, HuberRegressor, LassoLars, BayesianRidge,SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler,Normalizer,MaxAbsScaler,RobustScaler,QuantileTransformer,PowerTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist



class FeatureEng():

    def __init__(self,path):
        
        self.label = defaultdict(LabelEncoder)
        self.train = pd.read_csv('{}/train.csv'.format(path), sep = ',').drop(columns= 'PassengerId')
        self.test = pd.read_csv('{}/test.csv'.format(path), sep = ',').drop(columns= 'PassengerId')
        self.X = pd.concat([self.train.drop(columns= 'Survived'),self.test], axis = 0).drop(columns=['Pclass','Age','SibSp','Parch','Ticket','Fare'])

    @staticmethod
    def Ticket_Char(X):
        num = []
        for value in X['Ticket']:
            if str(value) == 'nan':
                num.append(np.nan)
            else:
                num_tick = ''.join([char for char in str(value) if char.isdigit()])
                if num_tick == '':
                    num.append(np.nan)
                else:
                    num.append(float(num_tick))
        X = X.drop(columns = 'Ticket')
        X['Ticket'] = num
        return X

    @staticmethod
    def Cabins_Char(X):
        cabin_char = []
        for value in X['Cabin']:
            if str(value) == 'nan':
                cabin_char.append(np.nan)
            else:
                cabin = ''.join([char for char in str(value) if not char.isdigit()])
                cabin_char.append(cabin)
        X = X.drop(columns = 'Cabin')
        X['Cabin'] = cabin_char
        return X

    @staticmethod
    def Title(X):
        title = []
        words = []
        for sentence in X['Name']:
            sentence = sentence.lower()
            try:
                start = sentence.index( ',' ) + 2
                end = sentence.index( '.', start )
                titulo = sentence[start:end] 
                title.append(titulo)
            except ValueError:
                print('Title Not Found')
            stop = stopwords.words('english') + title
            toke = RegexpTokenizer(r'\w+')
            tokens = toke.tokenize(sentence)
            words.append([word for word in tokens if word not in stop])
        return title,words

    @staticmethod
    def Names(X,title_words,fit_):
        
        compress = np.concatenate(title_words[1])
        freq = FreqDist(compress)

        label_sum =[]
        label_comb = []
        for sentence in title_words[1]:
            index = "".join([str(freq[word]) for word in sentence])
            label_sum.append(sum([freq[word] for word in sentence]))
            label_comb.append(int(index))

#       #  AVALIAR A CORRELACAO DAS LABELS PELAS COLUNAS  #
#           forte relacao label_sums * Pclass 

        #label_combs = []
        label_sums = []
        for i in range(len(title_words[0])):
        
            #label_combs.append( float(str(fit[i]) + '.' + str(label_comb[i]) ))
            label_sums.append(float(str(fit_[i]) + '.' + str(label_sum[i])))
        #self.X['label_sum'] = label_sum
        X['label_sums'] = label_sums
        X = X.drop(columns='Name')
        return X

    @staticmethod
    def grid_models(rss,xy):
        estimator = rss[0]
        style = rss[1]
        scaler = rss[2]

        y = xy[1]
        X = xy[0]
        MSE = []
        MAE = []
        r2 = []
        for frame in range(len(X)):
            X_ = X[frame]
            y_ = y
            num_items = len(X_)
            num_train = round(num_items * 0.8)
            num_val = num_items - num_train
            x_Train,x_Test,y_Train,y_Test = train_test_split(X_,y_,train_size = num_train, test_size= num_val,random_state=50,stratify=y_)

            clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
            clf.fit(x_Train, y_Train)
            y_pred = clf.predict(x_Test)
            MSE.append(mean_squared_error(y_Test,y_pred))
            MAE.append(mean_absolute_error(y_Test,y_pred))
            r2.append(r2_score(y_Test,y_pred))
        param = pd.DataFrame([estimator,style,scaler],index=['ESTIMATOR','STYLE','SCALER']).T
        scores = pd.DataFrame([MSE,MSE,r2], index=['MSE','MAE','R2']).T
        grid = pd.concat([param,scores], axis = 1)
        return grid
    
    def labels(self):
        print('==========CODIFICANDO LABELS CATEGORICOS==========')
        title = FeatureEng.Title(X = self.X)
        self.label['Title'] = LabelEncoder().fit(title[0])
        
        self.X = self.X.drop(columns= 'Name')
        
        self.X = FeatureEng.Cabins_Char(X = self.X)
        
        self.X.apply(lambda x: self.label[x.name].fit(x))

    def Encoder(self,data):
        print('==========TRATANDO DATA==========')
        title_words = FeatureEng.Title(data)
        fit_train = self.label['Title'].transform(title_words[0])
        data['Title'] = fit_train

        data = FeatureEng.Names(X= data,title_words= title_words,fit_ = fit_train)
        data = FeatureEng.Cabins_Char(data)
        data = FeatureEng.Ticket_Char(data)

        num = [col for col in data.columns if data[col].dtypes == 'O']
        categorical = data[num]
        
        mask_nan = categorical.isnull()
        trans = categorical.apply(lambda x: self.label[x.name].transform(x))
        trans = trans.where(~mask_nan,categorical).astype({"Embarked" : float, "Cabin" : float})
        
        noncategorical = [col for col in data.columns if col not in num]
        data = pd.concat([trans,data[noncategorical]], axis = 1)

        
        return data
    