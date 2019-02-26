"""
Datasets 
"""
import rep.preprocessing as p
import gin

@gin.configurable
def rep_blood_expression(x_train_h5, y_train_h5, x_valid_h5, y_valid_h5):
    
    x_train = p.readh5(x_train_h5)
    y_train = p.readh5(y_train_h5)
    x_valid = p.readh5(x_valid_h5)
    y_valid = p.readh5(y_valid_h5)
    
    # avoid zero entries
    x_train[0,:] = x_train[0,:] + 0.001
    y_train[0,:] = y_train[0,:] + 0.001
    x_valid[0,:] = x_valid[0,:] + 0.001
    y_valid[0,:] = y_valid[0,:] + 0.001
    
    return (x_train, y_train), (x_valid, y_valid)
    
    