
import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time

class EBLS():
    def __init__(self):
        
        self.beta11 = []
        self.distMaxAndMin = []
        self.minOfEachWindow = []
        self.beta = []
        self.wh = []
        self.beta2 = []
        self.parameter = 0 
        self.correct_count = 0

        self.data_counter = 0
        self.t2_cal = 0
        self.scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self.batch_size = 0

        self.dot_product = []
        self.L2_prev = []
        self.weights = []
        self.max = []
        self.maxOfEachWindow = []
        self.t2_trans = []
        self.tx2 = []
        self.acc = 0
        self.acc_eachstep = []
        
        self.batch_acc_ensemble = []
        self.batch_prec_ensemble = []
        self.batch_recall_ensemble = []
        self.batch_subset_acc_ensemble = []
        
        self.s = 0
        self.C = 0
        self.N1 = 0
        self.N2 = 0
        self.N3 = 0
        
        
    def show_accuracy_ensemble_multi_label(self,predictLabel,Label, batch_size):
       
        numerator = np.sum(np.logical_and(Label, predictLabel), axis = 1)
        denominator = np.sum(np.logical_or(Label, predictLabel), axis = 1)
        instance_accuracy = numerator/denominator
        wrong_index = []
        avg_accuracy = np.mean(instance_accuracy)
        acc_tmp = avg_accuracy

        self.data_counter += batch_size
        
        batch_acc = acc_tmp
        
        
        batch_acc = batch_acc * 100
        #self.batch_acc_ensemble = []
        self.batch_acc_ensemble.append(batch_acc)
        tmp_all_batch = self.batch_acc_ensemble.copy()
        all_acc = np.mean(tmp_all_batch)
        
        return batch_acc,all_acc, self.batch_acc_ensemble , wrong_index
    def show_f1_ex_ensemble_multi_label(self,prec,recall):
        numirator = 2 * prec * recall
        denominator = prec + recall 
        f1 = numirator / denominator
        return f1
    def show_prec_ex_ensemble_multi_label(self,predictLabel,Label, batch_size):
        
        self.data_counter += batch_size
        
        precision_num = np.sum(np.logical_and(Label, predictLabel), axis = 1)
    
        # Total number of pred true labels
        precision_den = np.sum(predictLabel, axis = 1)
        
        # precision averaged over all training examples
        avg_precision = np.mean(precision_num/precision_den)

        batch_acc = avg_precision
        
        batch_acc = batch_acc * 100
        
        self.batch_prec_ensemble.append(batch_acc)
        tmp_all_batch = self.batch_prec_ensemble.copy()
        all_acc = np.mean(tmp_all_batch)
        
        return all_acc, batch_acc#self.batch_prec_ensemble 
    
    def show_subset_accuracy(self,predictLabel,Label, batch_size):
       
        num = np.sum(np.all(predictLabel == Label ,axis = 1))
        den = batch_size
        
        tmp = num/den
        tmp = np.all(Label == predictLabel, axis = 1).mean()
        avg_subse_acc = tmp
        
        self.data_counter += batch_size
 
        batch_acc = avg_subse_acc
        
        batch_acc = batch_acc * 100
        
        self.batch_subset_acc_ensemble.append(batch_acc)
        tmp_all_batch = self.batch_subset_acc_ensemble.copy()
        all_acc = np.mean(tmp_all_batch)
       
        return all_acc, batch_acc#self.batch_subset_acc_ensemble 
    
    
    def show_recall_ex_ensemble_multi_label(self,predictLabel,Label, batch_size):
        
        recall_num = np.sum(np.logical_and(Label, predictLabel), axis = 1)
        
        recall_den = np.sum(Label, axis = 1)
        
        tmp = recall_num/recall_den
        nans = np.where(np.isnan(tmp))
        tmp[nans] = 0
        avg_recall = np.mean(tmp)
    
        self.data_counter += batch_size
        
        batch_acc = avg_recall
        
        batch_acc = batch_acc * 100
        
        self.batch_recall_ensemble.append(batch_acc)
        tmp_all_batch = self.batch_recall_ensemble.copy()
        all_acc = np.mean(tmp_all_batch)
        
        return all_acc, batch_acc#self.batch_recall_ensemble 
    
    def show_accuracy_ensemble(self,predictLabel,Label, batch_size):
        wrong_index = []
        self.data_counter += batch_size
        label_1 = np.zeros(Label.shape[0])
    
        label_1 = Label.argmax(axis = 1)

        predlabel = predictLabel.argmax(axis = 1)
        
        choices = []
        cc = 0
        firstclass_c = 0
        secondclass_c = 0
        
        indexes = []
        for j in list(range(Label.shape[0])):
            
            if(label_1[j] != predlabel[j]):
                wrong_index.append(j)
                choices.append(0)  
                indexes.append(j)
                
            elif (label_1[j] == predlabel[j]):
                if(label_1[j] == 0 ):
                    firstclass_c += 1
                if(label_1[j] == 1):
                    secondclass_c += 1
                choices.append(1)
                self.correct_count += 1
                cc += 1      
        batch_acc = choices.count(1)/batch_size
        batch_acc = batch_acc * 100
        #print(batch_acc)
        self.batch_acc_ensemble = []
        self.batch_acc_ensemble.append(batch_acc)
        
        return indexes,(round(self.correct_count/self.data_counter,4)) , self.batch_acc_ensemble , wrong_index
    def show_accuracy(self,predictLabel,Label):
        
        label_1 = np.zeros(Label.shape[0])
        
        label_1 = Label.argmax(axis = 1)
        predlabel = predictLabel.argmax(axis = 1)
        
        
        choices = []
        cc = 0
        firstclass_c = 0
        secondclass_c = 0
        
        for j in list(range(Label.shape[0])):
            if(label_1[j] != predlabel[j]):
                choices.append(0)  
                
            if label_1[j] == predlabel[j]:
                
                if(label_1[j] == 0):
                    firstclass_c += 1
                if(label_1[j] == 1):
                    secondclass_c += 1
                choices.append(1)
                #self.correct_count += 1
                cc += 1      
        
        batch_acc = choices.count(1)/self.batch_size
        choices = np.array(choices)
       
        return 0 , batch_acc
    
    def tansig(self,x):
        return (2/(1+np.exp(-2*x)))-1

    def sigmoid(self,data):
        return 1.0/(1+np.exp(-data))

    def linear(data):
        return data

    def tanh(self,data):
        return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))

    def relu(self,data):
        return np.maximum(data,0)

    def pinv2(self,A,row,reg):
        return np.mat(reg*np.eye(A.shape[1])+A).I.dot(row)
    def pinv(self,A,reg):
        return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)

    def shrinkage(self,a,b):
        z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
        return z
    
    def sparse_bls_2(self,A,b,dot_product,L2_prev):
        
        lam = 0.00001
        itrs = 2
        AABB = A.T.dot(A)
       
        AABB = AABB + dot_product
        
        m = A.shape[1]
        n = b.shape[1]
        x1 = np.zeros([m,n])
        wk = x1
        ok = x1
        uk = x1
        
        L1 = np.mat(AABB + np.eye(A.shape[1])).I
        
        L2_1 = A.T.dot(b) + L2_prev
        #L2_1 = normalize(L2_1)
        L2 = L1.dot(L2_1)
        
        for i in range(itrs):
            ck = L2 + np.dot(L1,(ok - uk))
            
            ok = self.shrinkage(ck + uk, lam)
            uk = uk + ck - ok
            wk = ok
        
        return wk,AABB,L2_1
    
    def BLS_online_init(self ,use_prep, train_x,s,C,N1,N2,N3):
        
        
        self.s = s
        self.C = C
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.batch_size = len(train_x)
    
        if(use_prep):
         
            train_x = preprocessing.scale(train_x,axis = 1)
        

        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0],1])])
    
        y = np.zeros([train_x.shape[0],N2*N1]);
        
        self.weights = []
        i = 0
        for i in range(N2):
            
            
            we = 2 * random.randn(H1.shape[1],N1)-1   
                      
            self.weights.append(we)
            
            
            A1 = H1.dot(we)
            
            self.scaler2.partial_fit(A1)
            
            A1 = self.scaler2.transform(A1)
            
          
            new_dot_product = np.zeros([N1,N1])
            new_L2_prev = np.zeros([N1,H1.shape[1]])
            self.dot_product.append(new_dot_product)
            self.L2_prev.append(new_L2_prev)
            
            beta1,new_dot_product,new_L2_prev = self.sparse_bls_2(A1,H1,new_dot_product,new_L2_prev)
            
            beta1 = beta1.T
            
            self.beta11.append(beta1)
            
            T1 = H1.dot(beta1)

            self.minOfEachWindow.append(np.zeros([1,T1.shape[1]]))
            self.maxOfEachWindow.append(np.zeros([1,T1.shape[1]]))
            self.distMaxAndMin.append( T1.max(axis = 0) - T1.min(axis = 0))
            T1 = (T1 - self.minOfEachWindow[i])/self.distMaxAndMin[i]
           
            y[:,N1*i:N1*(i+1)] = T1
        
        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0],1])])
       
        if N1*N2>=N3 :
            random.seed(67797325)
            wh = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
        else:
            random.seed(67797325)
            wh = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
        self.wh.append(wh)
        
        
        T2 = H2.dot(self.wh)
        
        T2 = T2.reshape(T2.shape[0], -1 )
        #print(s)
        
        self.parameter = s/np.max(T2)
        
        T2 = self.tanh(T2 * self.parameter);
        #T2 = T2 * 0
        T3 = np.hstack([y,T2])
        
        
        return T3
        
    def test_first_phase(self,use_prep, test_x):
        self.batch_size = len(test_x)
        ymin = 0
        
        if(use_prep):
            test_x = preprocessing.scale(test_x,axis = 1)
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
        #HH1 = test_x
        yy1=np.zeros([test_x.shape[0],self.N2*self.N1]);
        i = 0
        for i in range(self.N2):
            beta1 = self.beta11[i]
            TT1 = HH1.dot(beta1)
            
            TT1 = (1 - ymin)*(TT1 - self.minOfEachWindow[i])/self.distMaxAndMin[i] - ymin
            yy1[:,self.N1*i:self.N1*(i+1)]= TT1
        
        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])]);
        TT2 = self.tanh(HH2.dot(self.wh[0]) * self.parameter)
        TT2 = TT2.reshape(TT2.shape[0] , -1)
        #TT2 = TT2 * 0 
        TT3 = np.hstack([yy1,TT2])
        
        return TT3
    
    def test_second_phase(self, TT3,test_y):
       
        
        self.data_counter = self.data_counter + self.batch_size
        self.batch_size = len(test_y)
       
        x = TT3.dot( self.beta2)
        x = np.array(x)
        TestingAccuracy ,batch_acc = self.show_accuracy(x,test_y)
       
        
        self.acc = TestingAccuracy * 100
        batch_acc = batch_acc * 100
       
        self.acc_eachstep.append(self.acc)
        return  self.acc_eachstep, batch_acc ,x
    
    def just_test_second_phase(self, TT3):
        x = TT3.dot( self.beta2)
        x = np.array(x)
        
        return  x
    
    def update_first_layer(self,use_prep,train_x):
        self.batch_size = len(train_x)
        
        train_xx = train_x
        
        if(use_prep):
            train_xx = preprocessing.scale(train_xx,axis = 1)
        Hx1 = np.hstack([train_xx, 0.1 * np.ones([train_xx.shape[0],1])])
        
        yx = np.zeros([train_xx.shape[0],self.N1*self.N2])
        
        i = 0
        for i in range(self.N2):

                
            A1 = Hx1.dot(self.weights[i])

            self.scaler2.partial_fit(A1)
            A1 = self.scaler2.transform(A1)

            
            beta1,new_dot_product,new_L2_prev = self.sparse_bls_2(A1,Hx1,self.dot_product[i],self.L2_prev[i])
            self.dot_product[i] = new_dot_product
            self.L2_prev[i] = new_L2_prev
            
            beta1 = beta1.T
            
            self.beta11[i] = (beta1)


            Tx1 = Hx1.dot(beta1)
            tmp_min = Tx1.min(axis = 0)
            
            tmp_min_array = []
            tmp_min_array.append(tmp_min)
            tmp_min_array.append(self.minOfEachWindow[i])
            tmp_min_array = np.array(tmp_min_array)

            self.minOfEachWindow[i] = (tmp_min_array.min(axis = 0))

            tmp_max = Tx1.max(axis = 0)
            tmp_max_array = []
            tmp_max_array.append(tmp_max)
            tmp_max_array.append(self.maxOfEachWindow[i])


            tmp_max_array = np.array(tmp_max_array)
            self.maxOfEachWindow[i] = (tmp_max_array.max(axis = 0))

            self.distMaxAndMin[i] = ( self.maxOfEachWindow[i] - self.minOfEachWindow[i])
            Tx1 = (Tx1 - self.minOfEachWindow[i])/self.distMaxAndMin[i]
            #if(counter_class == 1):
            yx[:,self.N1*i:self.N1*(i+1)] = Tx1

      
        Hx2 = np.hstack([yx, 0.1 * np.ones([yx.shape[0],1])]);
        wh = self.wh[0]
        self.parameter = self.s/np.max(Hx2)
        t2 = self.tanh(Hx2.dot(wh) * self.parameter);
        
        t2 = np.hstack([yx, t2])
        
        return t2
    
    
    def update_second_layer(self,t2,train_y1,C):
        
        self.C = C
        self.batch_size = len(train_y1)
        
        t2 = np.array(t2)
        
        if((self.t2_cal == 0) or ( len(self.t2_trans) == 0)):
            
            self.t2_trans = t2.T.dot(t2)
            self.tx2 = t2.T.dot(train_y1)     
            
        else:
            
            self.t2_trans = self.t2_trans + (t2.T.dot(t2))
            self.tx2 = self.tx2 + (t2.T).dot(train_y1)

        betat = self.pinv2(self.t2_trans,self.tx2,self.C)
        self.beta2 = betat
        
        self.beta2 = np.array(self.beta2)
        
        self.t2_cal = self.t2_cal + 1
        