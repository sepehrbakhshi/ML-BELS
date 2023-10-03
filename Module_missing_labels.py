
import copy
from EBLS import *
import numpy as np
import time
from sklearn.preprocessing import normalize
from numpy.random import default_rng

class Module_missing_labels():
    def __init__(self,label_count,N1,N2,N3,max_learners, data_numbers):
        self.data_numbers = data_numbers     
        self.max_learners = max_learners
        self.N1 = N1       
        self.N2 = N2
        self.N3 = N3  
        self.s = 0.8  #shrink coefficient
        self.C = 2**-30 # Regularization coefficient
        self.learners = []
        self.acc_BLS = EBLS()
        self.acc_BLS_2 = EBLS()
        self.acc_BLS_3 = EBLS()
        self.label_count = label_count
        self.ensemble_accs_labeled = []

        self.initialized = False
        self.beta_1 = []
        self.learners_count = 0
       
        self.preds_w = np.zeros((self.data_numbers,self.label_count))
           
        self.ensemble_accs = []
 
        self.current_learner = 0
        self.worst_list = []
      
        self.restarted_list = []
        self.eleminated = []    
        self.old_besties = []
       
        self.threshold = 50

    def test_then_train(self,test_whole,is_binary,x_list, y_list,x_list_labeled,y_list_labeled,y_list_missing,missing_data,missing_label,missing_label_all):
        test_whole = np.array(test_whole)
        removed_instances_all = np.where(missing_label_all == 2)[0]
        
        removed_instances = np.where(y_list_missing == 2)[0]
        
        removed_instances_all= np.unique(removed_instances_all)
        
        new_TT3 = test_whole.copy()
        
        if(is_binary):
            
            if(len(new_TT3) > 0):
              
                new_TT3 = np.delete(new_TT3, removed_instances, axis=0)
             
            yy = y_list_labeled.copy()
            x_list_labeled = np.delete(x_list_labeled, removed_instances, axis=0 )
            yy = np.delete(yy, removed_instances, axis=0)
            y_list_labeled =yy.copy()
            
        else:
            
            new_TT3 = test_whole.copy()
            y_list_labeled= missing_label_all
            
        p_labeled = []
        ensemble_acc_2 = -1
        ensemble_acc = -1
        tmp_batch_acc = -1

        preds = np.zeros((x_list.shape[0],self.label_count))
        pred_prob = np.zeros((x_list.shape[0],self.label_count))
        pred_prob_w = np.zeros((len(x_list),self.label_count))
        
        if(self.learners_count < self.max_learners ):  
           
            EBLS_tmp = None
            EBLS_tmp = EBLS()        
            if(is_binary == False):
                print("here")
            self.restarted_list = []    
            self.learners.append(EBLS_tmp)
            
            self.current_learner = self.learners_count# + 
            self.learners_count += 1 
            self.restarted_list.append(self.current_learner) 
            
        if(self.learners_count >= self.max_learners or  (len(self.worst_list) > len(self.learners) /2)):
            
            self.restarted_list = []
            if(len(self.worst_list) != 0):
                
                for k in range(0 ,int(((len(self.worst_list))) -1 )):
    
                    l = self.worst_list[k]
                   
                    if(len(self.worst_list) > len(self.learners)/2 and l!=0):
                        
                        self.eleminated.append(self.learners[l])
                    
                    if(l!=0  ):
                        
                        self.learners[l] = 0
                
                learners_tmp_new = []    
  
                for k in range(0 , len(self.learners)):
                    if(self.learners[k] != 0):
                        
                        learners_tmp_new.append(self.learners[k])
                self.learners = learners_tmp_new
                
                ff = 0
                while (len(self.learners) < (self.max_learners ) and ff < len(self.old_besties)):
                    if(self.old_besties[ff] < len(self.eleminated)):
                        tmpppp = copy.deepcopy(self.eleminated[self.old_besties[ff]])
                        
                        self.learners.append(tmpppp)
                        
                        del self.eleminated[self.old_besties[ff]]
    
                        self.restarted_list = []
                        
                    ff += 1
                
                if(len(self.eleminated ) > 100):
                    self.eleminated = self.eleminated[20:100]
                
                self.old_besties = []
                
                self.learners_count = len(self.learners)
        
            
        self.worst_list = []
        
        if(self.initialized == False ):
            
            self.beta_1 = x_list_labeled
            
            for m in range(0 , self.learners_count):      
                
                self.learners[m].update_second_layer(self.beta_1,y_list_labeled,self.C)
            self.initialized = True

        elif(self.initialized == True ):
            
            TT3_w = test_whole#self.BLS.test_first_phase(self.x_list)            
            
            self.preds_w = np.zeros((len(x_list),self.label_count))
            aa = time.time()  
            for m in range(0 , self.learners_count):
                
                if((m not in self.restarted_list) and len(self.learners[m].beta2 ) > 0):
                    
                        if(len(y_list_labeled)!= 0):
                            self.my_accs ,batc_acc ,m_pred = self.learners[m].test_second_phase(new_TT3,y_list_labeled)
                        else:
                            batc_acc =100
                        m_pred_w = []
                     
                        #m_pred_w = m_pred
                        m_pred = self.learners[m].just_test_second_phase(TT3_w)
                        m_pred_w = m_pred
                        if(len(self.ensemble_accs_labeled)!=0):
                            self.threshold = self.ensemble_accs_labeled[len(self.ensemble_accs_labeled)-1]
                        
                        if(batc_acc < self.threshold) :
                            
                            self.worst_list.append(m)
                    
                        
                        prediction_prob = m_pred.copy()       
                        prediction = m_pred.copy()
                        
                        prediction_prob_w = m_pred_w.copy()
                        prediction_w = m_pred_w.copy()
                    
                        prediction_mins = np.zeros_like(prediction)
                        prediction_mins_w = np.zeros_like(prediction_w)
                        
                        prediction_mins[np.arange(len(prediction)), prediction.argmax(1)] = 1
                        prediction_mins_w[np.arange(len(prediction_w)), prediction_w.argmax(1)] = 1
  
                        prediction = 1 * prediction_mins 
                        prediction_w = 1 * prediction_mins_w
                        
                        preds = preds + prediction    
                        pred_prob = pred_prob +prediction_prob
                        
                        self.preds_w = self.preds_w + prediction_w
                        pred_prob_w = pred_prob_w +prediction_prob_w
                    
           
            for jj in range(0 , len(preds)):
                if(preds[jj,0] == preds[jj,1]):
                    preds[jj] = pred_prob[jj]
            for jj in range(0 , len(self.preds_w)):
                if(self.preds_w[jj,0] == self.preds_w[jj,1]):
                    self.preds_w[jj] = pred_prob_w[jj]    
            
            self.old_besties = []
            
            for ll in range(0 , len(self.eleminated)):
                if(len(y_list_labeled) != 0):
                    my_accs_e ,batc_acc_e ,m_pred_e = self.eleminated[ll].test_second_phase(new_TT3,y_list_labeled)
                else:
                    batc_acc_e = 0
                if(is_binary == True):
                    th = 50
                else:
                    th = 30
                if(batc_acc_e > th):
                    
                    self.old_besties.append(ll)
            
            indexes_2 = []
            indexes = []
            if(is_binary):               
                indexes, ensemble_acc , tmp_batch_acc , wrong_index = self.acc_BLS.show_accuracy_ensemble(self.preds_w,y_list,len(x_list))
                preds_copy = self.preds_w.copy()
                preds_copy = np.delete(preds_copy, removed_instances, axis=0)   
                if(len(y_list_labeled)!= 0):
                    indexes_2, ensemble_acc_2 , tmp_batch_acc_2 , wrong_index_2 = self.acc_BLS_2.show_accuracy_ensemble(preds_copy,y_list_labeled,len(x_list_labeled))
                else:
                    ensemble_acc_2= 100
            if( is_binary == False ): #non_binary
                
                self.preds_w = self.preds_w / (len(self.learners)-1)
                
                self.preds_w = normalize(self.preds_w, axis=1, norm='max')
                
                for k in range(0, len(self.preds_w)):
                    iindex = np.argmax(self.preds_w[k])
                    self.preds_w[k,iindex] = 1
                    
                    for l in range(0, self.label_count):
                        if(l != iindex):
                            self.preds_w[k,l] = 0
                
                
                _, ensemble_acc , tmp_batch_acc , wrong_index = self.acc_BLS_3.show_accuracy_ensemble_multi_label(self.preds_w,y_list,len(x_list))
                ensemble_acc_2 = ensemble_acc
                ensemble_acc = ensemble_acc * 100
            ensemble_acc_2 = ensemble_acc_2 * 100
            
            self.ensemble_accs.append(ensemble_acc)
            self.ensemble_accs_labeled.append(ensemble_acc_2)
            
            p_labeled = pred_prob_w.copy()            
          
            if(is_binary and len(indexes_2) > 0):
                #print("oops")
                #time.sleep(2)
                y_tmp = y_list_labeled[indexes_2]
                x_tmp = x_list_labeled[indexes_2]
                
                x_tmp = np.array(x_tmp)
                y_tmp = np.array(y_tmp)
                x_list_labeled = np.vstack((x_list_labeled,x_tmp))
                y_list_labeled = np.vstack((y_list_labeled,y_tmp))
            
            self.beta_1 = x_list_labeled
           
            learners_choice = []
            rng = default_rng()
           
            for zz in range(0 , len(self.learners)):
                second_phase_start = time.time()
                if(len(y_list_labeled)!=0):
                   
                    self.learners[zz].update_second_layer(self.beta_1 ,y_list_labeled,self.C)
                
            self.preds_w = np.zeros((self.data_numbers,self.label_count))
            pred_prob_w = np.zeros((self.data_numbers,self.label_count))
            preds = np.zeros((y_list.shape[0],self.label_count))
            pred_prob = np.zeros((y_list.shape[0],self.label_count))
        
        self.x_list = []
        y_list = []

        return  p_labeled,tmp_batch_acc
    