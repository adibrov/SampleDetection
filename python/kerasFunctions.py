

# def my_binary_crossentropy(weights =(1., 1.)):
#     def _func(y_true, y_pred):
#         return -(weights[0] * K.mean((1-y_true)*K.log((1-y_pred)+K.epsilon())) +
#                  weights[1] * K.mean(y_true*K.log(y_pred+K.epsilon())))
#     return _func
import keras.backend as K


def my_binary_crossentropy(weights =(1., 1.)):
    def _func(y_true, y_pred):
        return -(weights[0] * K.mean((1-y_true)*K.log((1-y_pred)+K.epsilon())) +
                 weights[1] * K.mean(y_true*K.log(y_pred+K.epsilon())))
    return _func

def my_binary_crossentropy_mod(weights =(1., 1.)):
    def _func(y_true, y_pred):
        return -(weights[0] * K.mean(K.cast(K.greater(0.25, y_true),dtype='float32')*K.log((1-y_pred)+K.epsilon())) +
                 weights[1] * K.mean(K.cast(K.greater(y_true,0.25),dtype='float32')*K.log(y_pred +K.epsilon())))
    return _func


############ accuracies #############
def acc1(y_true, y_pred):
   
   
    nom = K.mean(K.cast(K.cast(K.equal(K.round(y_pred),y_true), dtype='float32')*K.cast(K.equal(y_true,1),dtype='float32'),dtype='float32'))
    denom = K.mean(y_true)
#     nom = K.cast(nom, dtype='float32')
#     denom =  K.cast(denom, dtype='float32')
    return nom/denom
#     return K.shape(y_true)[0]


def acc0(y_true, y_pred):
    
    nom = K.mean(K.cast(K.cast(K.equal(K.round(y_pred),y_true), dtype='float32')*K.cast(K.equal(y_true,0),dtype='float32'),dtype='float32'))
    denom = K.mean(K.cast(K.equal(y_true,0),dtype='float32'))
#     nom = K.cast(nom, dtype='float32')
#     denom =  K.cast(denom, dtype='float32')
    return nom/denom

def acc1_mod(y_true, y_pred):
   
   
    nom = K.mean(K.cast(K.cast(K.greater(0.05,K.abs(y_pred-y_true)), dtype='float32')*K.cast(K.greater(y_true,0.2),dtype='float32'),dtype='float32'))
    denom = K.mean(K.cast(K.greater(y_true,0.2),dtype='float32'))
#     nom = K.cast(nom, dtype='float32')
#     denom =  K.cast(denom, dtype='float32')
    return nom/denom
#     return K.shape(y_true)[0]



def acc0_mod(y_true, y_pred):
    
    nom = K.mean(K.cast(K.cast(K.equal(K.round(y_pred),y_true), dtype='float32')*K.cast(K.equal(y_true,0),dtype='float32'),dtype='float32'))
    denom = K.mean(K.cast(K.equal(y_true,0),dtype='float32'))
#     nom = K.cast(nom, dtype='float32')
#     denom =  K.cast(denom, dtype='float32')
    return nom/denom
