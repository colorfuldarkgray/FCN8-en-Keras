#!/usr/bin/env python
# coding: utf-8

## Import usual libraries
#import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as be 
be.clear_session()
import keras, sys, time, warnings
warnings.filterwarnings("ignore")
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import pandas as pd 
import cv2, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import random
from contextlib import redirect_stdout
  
print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__))#; del keras
#print("tensorflow version {}".format(tf.__version__))

def FCN8(nClasses, input_height, input_width, vgg_n, tl):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
	## vgg debe ser entero igual a 16 o 19
	## tl=1 para activar transfer learning y cargar pesos codificador vgg entrenados con Imagenet para clasificación
    assert input_height%32 == 0
    assert input_width%32 == 0
    
    tn_enc=True # True or False (train vgg encoder or not)
    tn_dec=True

    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 3)) ## Assume 224,224,3
    
    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING, trainable=tn_enc)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    if vgg_n==19: x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    if vgg_n==19: x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING, trainable=tn_enc )(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    if vgg_n==19: x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', data_format=IMAGE_ORDERING, trainable=tn_enc )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)

    #x = Flatten(name='flatten')(x)
    #x = Dense(4096, activation='relu', name='fc1')(x)
    # <--> o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data
    
    #x = Dense(4096, activation='relu', name='fc2')(x)
    # <--> o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)   
    # assuming that the input_height = input_width = 224 as in VGG data
    
    #x = Dense(1000 , activation='softmax', name='predictions')(x)
    # <--> o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data
    
    if tl==1:
        vgg  = Model(  img_input , pool5  )
        vgg.load_weights('vgg'+str(vgg_n)+'.h5') ## loading VGG weights for the encoder parts of FCN8
    
    n = 4096
    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING, trainable=tn_dec))(pool5)
    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING, trainable=tn_dec))(o)
    
    
    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING, trainable=tn_dec )(conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING, trainable=tn_dec))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING, trainable=tn_dec ))(pool411)
    
    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING, trainable=tn_dec))(pool3)
        
    o = Add(name="add")([pool411_2, pool311, conv7_4 ])
    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING, trainable=tn_dec )(o)
    o = (Activation('softmax'))(o)
    
    model = Model(img_input, o)
    #if stage>0:
    #    model.load_weights(fpath+fid[:-1]+str(stage-1)+'.h5') ## loading  weights from previous training

    return model

def getImageArr( path , width , height ):
        img = cv2.imread(path, 1)
        img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
        return img

def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ))
    img = img[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)
    ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
    return seg_labels


def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)
    file1 = open(fpath+"performance_"+fid+".txt","w") 
    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c)) 
        IoU = TP/float(TP + FP + FN)
        str="class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}\n".format(c,TP,FP,FN,IoU)
        print(str)
        file1.write(str)
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    str="_________________\n"
    file1.write(str)
    print(str)
    str="Mean IoU: {:4.3f}\n".format(mIoU)
    file1.write(str)
    print(str)
    file1.close()


# Visualize the model performance
def give_color_to_seg_img(seg,n_classes):
    '''
    seg : (input_width,input_height,3)
    '''
    
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros((seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc=(seg==c)
        seg_img[:,:,0]+=(segc*(colors[c][0]))
        seg_img[:,:,1]+=(segc*(colors[c][1]))
        seg_img[:,:,2]+=(segc*(colors[c][2]))

    return(seg_img)


#conf=[['c',0,16],['c',1,16],['c',2,16],['c',0,19],['c',1,19],['c',2,19],['nc',0,16],['nc',1,16],['nc',2,16],['nc',0,19],['nc',1,19],['c',2,19]]
#conf=[['c',16,'tl'],['nc',16,'tl'],['c',19,'tl'],['nc',19,'tl'],['c',16,'f0'],['nc',16,'f0'],['c',19,'f0'],['nc',19,'f0']]
conf=[['c',16,'tl'],['c',19,'tl'],['c',16,'f0'],['c',19,'f0']]
## tl es transfer learning
## f0 es desde cero
## 
for i in range(3,4):
    modo=conf[i][0]
#    tn_stage=conf[i][1]
    vgg_number=conf[i][1] # 16 o 19
    tipo=conf[i][2]       # tl o f0
    if tipo=='tl':tlb=1
    else:tlb=0
    if modo=='c': 
        fpath='centrado/'
    if modo=='nc': 
        fpath='no_centrado/'
    sz=224

    fid='fcn_vgg'+str(vgg_number)+'_'+modo+'_'+tipo
    dir_img = fpath+'n_img/'
    dir_seg = fpath+'n_lbl/'
    
    n_classes=36

    model = FCN8(nClasses     = n_classes,  
                 input_height = sz, 
                 input_width  = sz,
                 vgg_n=vgg_number,
                 tl=tlb)

    with open(fid+'_summary.txt', 'a') as f:
        with redirect_stdout(f):
            model.summary()

#    file = open(fid+'_summary.txt', 'w')
#    sys.stdout = file
#    model.summary()
#    file.close()
#    sys.stdout = save_stdout
    
    input_height , input_width = sz , sz
    output_height , output_width = sz , sz

    images = os.listdir(dir_img)
    images.sort()
    segmentations  = os.listdir(dir_seg)
    segmentations.sort()
        
    X = []
    Y = []
    for im , seg in zip(images,segmentations) :
        X.append( getImageArr(dir_img + im , input_width , input_height )  )
        Y.append( getSegmentationArr( dir_seg + seg , n_classes , output_width , output_height )  )

    X, Y = np.array(X) , np.array(Y)
    print(X.shape,Y.shape)
    
    train_rate = 0.75
    index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)
    index_test  = list(set(range(X.shape[0])) - set(index_train))
    X, Y = shuffle(X,Y)
    X_train, y_train = X[index_train],Y[index_train]
    X_test, y_test = X[index_test],Y[index_test]
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    #lrsch=LearningRateScheduler(schedule, verbose=0)
    mck=ModelCheckpoint(fpath+fid+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #es=EarlyStopping(monitor='val_loss', min_delta=0, patience=90, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    sgd = optimizers.SGD(lr=0.01, decay=5**(-4), momentum=0.9, nesterov=True)
    #adadelta=optimizers.Adadelta()
    #adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #adag=optimizers.Adagrad(lr=0.01)
    #nadam=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    hist1 = model.fit(X_train,y_train,
                      validation_data=(X_test,y_test),
                      batch_size=32,epochs=900,verbose=2,
                      callbacks=[mck])

    #model.save(fpath+fid+".h5")

    # Plot the change in loss over epochs
    plt.figure()
    for key in ['loss', 'val_loss']:
        plt.plot(hist1.history[key],label=key)
    plt.legend()
    plt.savefig(fpath+fid+'loss_vs_epochs_'+'.png')

    # Calculate intersection over union for each segmentation class
    y_pred = model.predict(X_test)
    y_predi = np.argmax(y_pred, axis=3)
    y_testi = np.argmax(y_test, axis=3)
    print(y_testi.shape,y_predi.shape)
    
    IoU(y_testi,y_predi)
    shape = (224,224)
    # n_classes= 2

    fig = plt.figure(figsize=(10,30))
    for i in range(10):
        img_is  = (X_test[i] + 1)*(255.0/2)
        seg = y_predi[i]
        segtest = y_testi[i]

        ax = fig.add_subplot(10,3,3*i+1)
        ax.imshow(img_is/255.0)
        ax.set_title("original")
        
        ax = fig.add_subplot(10,3,3*i+2)
        ax.imshow(give_color_to_seg_img(seg,n_classes))
        ax.set_title("predicted class")
        
        ax = fig.add_subplot(10,3,3*i+3)
        ax.imshow(give_color_to_seg_img(segtest,n_classes))
        ax.set_title("true class")
    plt.savefig(fpath+fid+'performance_'+'.png')

    del model
    del hist1
    be.clear_session()


