import copy
from EBLS import *
import numpy as np
import time
from scipy.io import arff
import matplotlib.pyplot as plt
from Module_missing_labels import *
#from Module_missing_labels_2 import *
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
#from skmultiflow.metrics import hamming_score
from sklearn.metrics import hamming_loss
import random
import tracemalloc
import sys
import pandas as pd
import warnings
"""
chunk size for datasets based on their sizes:
dataset size ==>/ chunk size
###### >= 30000  =>>   1000
######15000 - 30000 =>> 500

######7000 - 15000 =>> 250
######1000 - 7000  =>> 100

######250 - 1000  =>>  50
######less than =< 250  =>> 25
"""

warnings.filterwarnings('ignore')
seeds = []
loop_no = 1
np.set_printoptions(threshold = sys.maxsize)
missing_percentage = 0


dataset_name = "1_Enron_new"
df = pd.read_csv(dataset_name+".csv" ,dtype=float)
dd = np.array(df)

label_count_full = 53
labels = dd[:,0:label_count_full]
data= dd[:,label_count_full:]
#chunk_size
chunk_size = 100
sensitivity_analysis = False
ablation = False
tau = 1.5

for mmm in range(0, loop_no):
    # starting the monitoring
    subset_acc_all = []
    f1_example_all = []
    micro_f1_all = []
    macro_f1_all = []
    acc_example_all = []
    hamming_score_result = []
    hamming_loss_full = []
    tracemalloc.start()
    
    
    #200NG: 500 , enrron = 100 , imddb = 1000, mediaamill= 1000,ohsuumed=250
    #Reuuters = 100 , slasshdot = 100, tmmc = 500 #yeasat = 100
     
    #random.seed(1000)
    
    f1_counter = 0
    f1_scikit_all = 0
    f1_scikit_macro_all = 0
    my_flag = False
    my_flag_final = False
    is_tested_final = False
    save = True
    
    max_learners = 3
    
    N1 = 5  #  # of nodes belong to each window
    #limit = 20 #  # of windows -------Feature mapping layer
    N2 = 5#20
    N3 = 1#120 #  # of enhancement nodes -----Enhance layer
    L = 15    #  # of incremental steps
    M1 = 50 #  # of adding enhance nodes
    s = 0.8  #  shrink coefficient
    C = 2**-30 # Regularization coefficient
    
    
    x_list = []
    y_list = []

    
    labels_in_chunk = label_count_full * chunk_size
    
    
    tt = time.time()
    tt2= time.time()
    
    
    
    
    print("loaded...")
    subset_acc_full = []
    ensemble_f1_full = []
    f1_scikit_micro_full = []
    f1_scikit_macro_full = []
    ensemble_acc_full = []
    prec_full = []
    recall_full = []
    
    model_test_train_time_start = time.time() #timer start
    
    labels = labels.astype(float)

    BLS = EBLS()
    BLS_prec = EBLS()
    BLS_recall = EBLS()
    BLS_f1 = EBLS()
    BLS_subset = EBLS()
    BLS_acc= EBLS()
    models = []
    
    subset_acc_full = []
    ensemble_f1_full =[]
    
    f1_scikit_micro_full = []
    f1_scikit_macro_full = []
    ensemble_acc_full = []
    prec_full = []
    recall_full = []
    
    full_labels_model = Module_missing_labels(label_count_full,N1,N2,N3,2, chunk_size)
    
    
    label_count_binary = 2
    for k in range(0, label_count_full):
        a = None
        a = Module_missing_labels(label_count_binary,N1,N2,N3,max_learners, chunk_size)
        models.append(a)
    
    LC_now = 1000
    LC = 0
    tmp_LC = 0
    LC_counter = 0
    
    y_list_not_encoded = []
    
    TT3 = []
    
    initialized = False
    
    
    for k in range(0, int(len(data)/chunk_size)+1):
        first = True
        
        x_list = data[(k*chunk_size): ((k+1)*chunk_size)]
        y_list = labels[(k*chunk_size): ((k+1)*chunk_size)]
        
        if(len(x_list) == 0 or (len(x_list) < chunk_size)):
            break
        
        print('-------------------Online_BLS---------------------------')
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        y_list_not_encoded = np.array(y_list_not_encoded)
        
        
        new_y_list = []
        new_y_list_not_encoded = []
       
        for k in range(0, label_count_full):
            new_y_list.append(np.zeros((1,2)))
            new_y_list_not_encoded.append(np.zeros((1,2)))
        new_y_list = np.ones((label_count_full,len(y_list),1))
        
        new_y_list_2 = np.zeros((label_count_full,len(y_list),1))
        new_y_list = np.concatenate((new_y_list,new_y_list_2), axis = 2)
        
        new_y_list_not_encoded = np.zeros((label_count_full,len(y_list),1))
        
        
        missing_label_positions = random.sample(range(0,labels_in_chunk), int(labels_in_chunk * missing_percentage))
       
        missing_label_positions = np.array(missing_label_positions, dtype=int)
        missing_label_positions = np.reshape(missing_label_positions,(1,missing_label_positions.shape[0]))
        
        y_list_missing = y_list.copy()
        new_y_list_mising = new_y_list.copy()
        
        missing_data = (missing_label_positions/label_count_full)
        missing_data = missing_data.astype(int)
        missing_label = (missing_label_positions%label_count_full)
        
        y_list_missing_all = y_list_missing.copy()
        
        new_y_list_mising[missing_label,missing_data,0] = 2
        
        y_list_missing_all_lc = y_list_missing_all.copy()
        y_list_missing_all[missing_data,missing_label] = 0
        y_list_missing_all_lc[missing_data,missing_label]= 2
      
        for k in range(0, len(y_list)):
            #b = np.where(y_list_missing_all_lc[k] == 2)[0]     
            a = np.where(y_list[k] == 1)[0]            
            new_y_list[a,k,0] = 0
            new_y_list[a,k,1] = 1 
            
            new_y_list_not_encoded[a,k,0] = 1
            first = False 
            
        ending = time.time()
        new_y_list = np.array(new_y_list)
        
        new_y_list_not_encoded = np.array(new_y_list_not_encoded)
        new_y_list_not_encoded = new_y_list_not_encoded.astype(int)
        ###############################################
        
        x_list = x_list.reshape(x_list.shape[0],-1)
        y_list = y_list.reshape(y_list.shape[0],-1)        
        ######################################################       
        
        
        if(initialized == False ):
            
            
            beta_1 = BLS.BLS_online_init(True,x_list,s,C,N1,N2,N3)           
         
            beta_1 = np.array(beta_1)
            all_data = []
            all_labeles = []
            
            
            for k in range(0, len(new_y_list_not_encoded)):
                x_data = []
                y_data = []
                
                X_res = beta_1
                y_data = new_y_list[k]
                x_data.append(X_res)
                
                x_data = np.array(x_data)
                y_data = np.array(y_data)
               
                all_data.append(x_data)
                all_labeles.append(y_data)
            
            for k in range(0, len(all_labeles)):
                all_labeles[k] = all_labeles[k].reshape((1,-1,2))
            
            for kk in range(0, len(models)):
                if(len(all_data[kk]) > 0):
                    is_binary = True
                    all_data_k = all_data[kk].reshape((all_data[kk].shape[0]*all_data[kk].shape[1],-1))
    
                    all_labeles_k = all_labeles[kk].reshape((all_labeles[kk].shape[0]*all_labeles[kk].shape[1],-1))
                    
                    m_pred,batc_acc  = models[kk].test_then_train(TT3,is_binary,beta_1,new_y_list[kk],all_data_k,all_labeles_k,new_y_list_mising[kk],missing_data,missing_label,y_list_missing_all)
                    
                    
            m_pred_all11,batc_acc11  = full_labels_model.test_then_train(TT3,False,beta_1,y_list,beta_1,y_list,new_y_list_mising[kk],missing_data,missing_label,y_list_missing_all)
             
            initialized = True
           
        elif(initialized == True ):
            
            TT3 = BLS.test_first_phase(True,x_list)         

            beta_1 = BLS.update_first_layer(True,x_list)
            
            #####################################################
            ############## Undersampling ########################
            beta_1 = np.array(beta_1)
            all_data = []
            all_labeles = []
            data_length = []
            for k in range(0, len(new_y_list_not_encoded)):
                x_data = []
                y_data = []
                X_res = beta_1
                y_data = new_y_list[k]
                
                x_data.append(X_res)
                
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                
                all_data.append(x_data)
                all_labeles.append(y_data)
            
                data_length.append(len(x_data[0]))
                
            for k in range(0, len(all_labeles)):
                all_labeles[k] = all_labeles[k].reshape((1,-1,2))
            #####################################################
            #####################################################
            
            testing_count = 0
            preds_binary = []
            batch_accs_all = []
            
            preds_full = []
            batch_accs_all_full = []
            def final_learner_3(TT3,beta_1,y_list,new_probs,decision_list,full_labels_model):
                
                preds_binary_2 = []
        
                m_pred_all,batc_acc  = full_labels_model.test_then_train(TT3,False,beta_1,y_list,beta_1,y_list,new_y_list_mising[kk],missing_data,missing_label,y_list_missing_all)

                m_pred_all = np.array(m_pred_all)                
                preds_binary_2 = m_pred_all
                if(len(preds_binary_2) > 0):
                    preds_binary_2 = preprocessing.minmax_scale(preds_binary_2, feature_range=(0, 1), axis=1, copy=True)

                return preds_binary_2
            
            aa = time.time()
            
            for kk in range(0, len(models)):
                    
                    testing_count += 1
                    all_data_k = all_data[kk].reshape((all_data[kk].shape[0]*all_data[kk].shape[1],-1))
                    
                    all_labeles_k = all_labeles[kk].reshape((all_labeles[kk].shape[0]*all_labeles[kk].shape[1],-1))
                    
                    m_pred,batc_acc  = models[kk].test_then_train(TT3,True,beta_1,new_y_list[kk],all_data_k,all_labeles_k,new_y_list_mising[kk],missing_data,missing_label,y_list_missing_all)
                    
                    preds_binary.append(m_pred)
                    batch_accs_all.append(batc_acc)
            aaa = time.time() - aa
            print("time_fast = ", aaa)
            preds_binary = np.array(preds_binary)
            
            
            new_probs = np.zeros((preds_binary.shape[0],preds_binary.shape[1]))
            preds_probs = np.zeros((preds_binary.shape[0],preds_binary.shape[1]))
            
            decision_list = []
            
            final_outcome = final_learner_3(TT3,beta_1,y_list,new_probs,decision_list,full_labels_model)
            
            counterrrr = 0
            
            final_outcome = final_outcome.T
            
            #1.5
            if(len(final_outcome) > 0 and LC_now >=tau) :
                
                preds_binary[:,:,0] = (abs(preds_binary[:,:,0]) * (1-final_outcome[:,:]))
                preds_binary[:,:,1] = (abs(preds_binary[:,:,1]) * (final_outcome[:,:]))
            
            tmp_column_one = np.where((preds_binary[:,:,0]) < (preds_binary[:,:,1]))
            tmp_column_zero = np.where((preds_binary[:,:,0]) > (preds_binary[:,:,1]))
            
            
            new_probs[tmp_column_one[0],tmp_column_one[1]] = 1
            preds_probs = preds_binary[:,:,1]
            bb = time.time()
            
            
            new_probs = np.transpose(new_probs) 
            preds_probs = np.transpose(preds_probs) 
            
            reserved_probes = new_probs.copy()
            predicts = []
            
            decision_list = []
            
            for k in range(0, len(new_probs)):
                result = np.all(new_probs[k] == 0)     
                if(result):
                    a = np.argmax(preds_probs[k], axis=0)
                    new_probs[k,a] = 1
             
            final_probs = new_probs

            f1_counter += 1
            f1_scikit_micro = f1_score(y_list, final_probs, average="micro")
            f1_scikit_macro = f1_score(y_list, final_probs, average="macro")
            
 
            tmp_batch, ensemble_acc , tmp_batch_acc , wrong_index = BLS_acc.show_accuracy_ensemble_multi_label(final_probs,y_list,chunk_size)
            ensemble_prec , tmp_batch_prec = BLS_prec.show_prec_ex_ensemble_multi_label(final_probs,y_list,chunk_size)
            ensemble_recall , tmp_batch_recall = BLS_recall.show_recall_ex_ensemble_multi_label(final_probs,y_list,chunk_size)

            ensemble_f1 = BLS_f1.show_f1_ex_ensemble_multi_label(ensemble_prec,ensemble_recall)
            
            #subset_acc,subset_acc_batch = BLS_subset.show_subset_accuracy(final_probs,y_list,chunk_size)
            #subset_acc_all.append(subset_acc_batch)
            acc_example_all.append(tmp_batch)     
            micro_f1_all.append(f1_scikit_micro) 
            macro_f1_all.append(f1_scikit_macro)
            f1_example_all.append(ensemble_f1)
            hamming_score_result = 1-hamming_loss(y_list,final_probs)
            
            
            hamming_loss_full.append(hamming_score_result)
            
            #print("subset_acc: ", np.mean(subset_acc_batch))
            print("example_based f1: ", np.mean(f1_example_all)/100)      
            print("micro_f1: ",np.mean(micro_f1_all) )
            print("macro_f1: " , np.mean(macro_f1_all) )
            print("example_based acc: ",np.mean(acc_example_all)/100)
            print("hamming score: ", np.mean(hamming_loss_full))

            print("length: ", len(acc_example_all), len(y_list))
           
        
        LC_counter += len(y_list_missing_all_lc)
        tmp_tmp = 0
        for k in range(0, len(y_list_missing_all_lc)):
            tmp = np.count_nonzero(y_list_missing_all_lc[k] == 1)
            tmp_LC += tmp
            tmp_tmp += tmp
        LC =  tmp_LC / LC_counter
        LD = (tmp_LC/label_count_full) / LC_counter
        
        print(LC_counter)
    
        LC_now = tmp_tmp/len(y_list_missing_all_lc)
        if(missing_percentage > 0):
            LC = (1/(1-missing_percentage) * LC)
            LC_now = (1/(1-missing_percentage) * LC_now) 
        #LC_now = 0
        print("LC: ", LC)
        print("LD: " , LD)
        #print("now: ", LC_now)
        
        x_list = []
        y_list = []
        y_list_not_encoded = []

    model_test_train_time_end = time.time()
    total_time = model_test_train_time_end - model_test_train_time_start 
    
    
    if(save == True):
   
        ln = str(mmm)
        mp = str(int(missing_percentage*100))
        #np.save("subset_acc_"+dataset_name, subset_acc_all)
        np.save("examplef1_"+dataset_name, f1_example_all)
        np.save("microf1_"+dataset_name, micro_f1_all)
        np.save("macrof1_"+dataset_name, macro_f1_all)
        #np.save("results_multi_label_new/"+dataset_name+"/precision_"+dataset_name, ensemble_prec)
        #np.save("results_multi_label_new/"+dataset_name+"/recall_"+dataset_name, ensemble_recall)
        np.save("accs_"+dataset_name, acc_example_all)
        np.save("time_"+dataset_name, total_time)
        np.save("memory_"+dataset_name, tracemalloc.get_traced_memory())
        np.save("hammingscore"+dataset_name, hamming_loss_full)
        
        print("save ==  true")
        
    
    
    print("Total runtime: ",total_time)
    # displaying the memory
    print(tracemalloc.get_traced_memory())
     
    # stopping the library
    tracemalloc.stop()
  