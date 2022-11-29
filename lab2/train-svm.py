#!/usr/bin/env python3

import numpy as np
from sklearn import svm
import pickle


if __name__ == '__main__':

    # Load dictionary
    with open('models/SVM/feat_to_id.pickle', 'rb') as handle:
        feat_to_id = pickle.load(handle)
    
    # Load data
    with open('models/SVM/M_SVM.pickle', 'rb') as handle:
        M_SVM = pickle.load(handle)
    with open('models/SVM/y_SVM.pickle', 'rb') as handle:
        y_SVM = pickle.load(handle)

    n_features = np.max([feat_to_id[key] for key in feat_to_id])

    # load ME model
    # 'linear' C = 0.05
    
    # Parameters:
    # kernels = ['linear', 'rbf']
    # C = [0.05, 0.5, 5]
    model = svm.SVC(kernel='rbf', C=5)
    
    # Train
    model.fit(M_SVM, y_SVM)
    
    with open('models/SVM/model_SVM.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
