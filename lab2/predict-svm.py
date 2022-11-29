#!/usr/bin/env python3

import sys
import numpy as np
from scipy.sparse import csr_matrix
import pickle


if __name__ == '__main__':

    # Load dictionary
    with open('models/SVM/feat_to_id.pickle', 'rb') as handle:
        feat_to_id = pickle.load(handle)
    
    max_id, key_max_id = max([(feat_to_id[key], key) for key in feat_to_id], key=lambda x: x[0])
    n_features = max_id + 1
    
    # Load model
    with open('models/SVM/model_SVM.pickle', 'rb') as handle:
        model = pickle.load(handle)
    
    row_sp = np.array([], dtype=int)
    col_sp = np.array([], dtype=int)
    data_sp = np.array([], dtype=int)
    current_instance = 0
    
    args_pred = []
    # Read instances from STDIN, and classify them
    for line in sys.stdin:
        
        fields = line.strip('\n').split("\t")
        (sid,e1,e2) = fields[0:3]
        args_pred.append((sid, e1, e2))
        
        last_feat = False
        ids_feats_to_add = []
        for feat in fields[4:]:
            if feat in feat_to_id:
                ids_feats_to_add.append(feat_to_id[feat])
                if feat == key_max_id:
                    last_feat = True
        
        data_row_sp = [1]*len(ids_feats_to_add)
        
        if not last_feat:
            ids_feats_to_add.append(max_id)
            data_row_sp.append(0)
        
        col_sp = np.concatenate((col_sp, ids_feats_to_add))
        row_sp = np.concatenate((row_sp, [current_instance]*len(ids_feats_to_add)))
        data_sp = np.concatenate((data_sp, data_row_sp))
        current_instance += 1
        
    data_sp = data_sp.astype(int)
    row_sp = row_sp.astype(int)
    col_sp = col_sp.astype(int)
    
    M_test = csr_matrix((data_sp, (row_sp, col_sp)))
    predictions = model.predict(M_test)
    
    for ind, prediction in enumerate(predictions):
        if prediction != "null" :      
            sid, e1, e2 = args_pred[ind]      
            print(sid,e1,e2,prediction,sep="|")
        
