import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import json, pickle
from novel_activations import *
from keras.utils.layer_utils import count_params


###
### Code for QMF wavelet CNNs is modified from https://github.com/MichauGabriel/DeSpaWN

class HardCodableCNN(tf.keras.layers.Layer):
    def __init__(self, stride=2, dilations=None, enforce_reverse=False, enforce_reverse_alternate=False, padding='SAME', **kwargs):
        self.stride    = stride
        self.dilations = dilations
        if any([x!=1 for x in self.stride]) and self.dilations!=None:
            raise ValueError("Cannot specify both stride and dilations")  
        self.padding   = padding      
        self.enforce_reverse_alternate=enforce_reverse_alternate
        self.enforce_reverse = enforce_reverse
        super(HardCodableCNN, self).__init__(**kwargs)


    def build(self, input_shape):
        self.qmfFlip = tf.reshape(tf.Variable([(-1)**(i+self.enforce_reverse_alternate) for i in range(input_shape[1][0])],
                                              dtype='float32', name='mask', trainable=False),(-1,1,1,1))
        super(HardCodableCNN, self).build(input_shape)

    def call(self, inputs):
        '''
        Inputs should be of shape (None, N, 1, 1), kernel [N,1,1,1]
        '''
        input_data = inputs[0]
        kernel = inputs[1]
        if self.enforce_reverse:
            ra_kernel = tf.reverse(kernel,[0])
        else:
            ra_kernel = kernel
        if self.enforce_reverse_alternate>0:    
            ra_kernel = tf.math.multiply(ra_kernel,self.qmfFlip)
        return tf.nn.conv2d(input_data, ra_kernel, padding=self.padding, strides=self.stride,dilations=self.dilations)
      
    def get_config(self):
        config = super().get_config()
        config.update({
            "stride"    : self.stride,
            "dilations"    : self.dilations,
            "padding"    : self.padding,
            "enforce_reverse_alternate" : self.enforce_reverse_alternate,
            "enforce_reverse" : self.enforce_reverse         
            
        })
        return config

class HardCodableInverseCNN(tf.keras.layers.Layer):
    def __init__(self, enforce_reverse_alternate=False, enforce_reverse=False, stride=(2,1), dilations=None, padding='SAME', **kwargs):
        self.enforce_reverse_alternate = enforce_reverse_alternate
        self.stride    = stride
        self.dilations = dilations
        self.enforce_reverse = enforce_reverse
        if any([x!=1 for x in self.stride]) and self.dilations!=None:
            raise ValueError("Cannot specify both stride and dilations")   
        self.padding   = padding      
        super(HardCodableInverseCNN, self).__init__(**kwargs)


    def build(self, input_shape):
        #print(input_shape, flush=True)
        self.qmfFlip = tf.reshape(tf.Variable([(-1)**(i+self.enforce_reverse_alternate) for i in range(input_shape[1][0])],
                                              dtype='float32', name='mask', trainable=False),(-1,1,1,1))
        super(HardCodableInverseCNN, self).build(input_shape)

    def call(self, inputs):
        '''
        Inputs should be of shape (None, N, 1, 1), kernel (N,1,1,1)
        '''
        input_data = inputs[0]
        
        kernel = inputs[1]


        #target_size = 
        if self.enforce_reverse:
            ra_kernel = tf.reverse(kernel,[0])
        else:
            ra_kernel = kernel
        if self.enforce_reverse_alternate>0:    
            ra_kernel = tf.math.multiply(ra_kernel,self.qmfFlip)
        # input  - (None, 512, 1, 1)
        # filter - (2, 1, 1, 1)
        # target - 3  ----> wrong

        return tf.nn.conv2d_transpose(input_data, ra_kernel, inputs[2], padding=self.padding, strides=self.stride,dilations=self.dilations)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "stride"    : self.stride,
            "dilations"    : self.dilations,
            "padding"    : self.padding,
            "enforce_reverse_alternate" : self.enforce_reverse_alternate,
            "enforce_reverse" : self.enforce_reverse
        })
        return config




class Kernel(tf.keras.layers.Layer):
    def __init__(self, kernelInit=8, trainKern=True, **kwargs):
        self.trainKern  = trainKern
        if isinstance(kernelInit,int):
            self.kernelSize = kernelInit
            self.kernelInit = 'random_normal'
        else:
            self.kernelSize = kernelInit.__len__()
            self.kernelInit = tf.constant_initializer(kernelInit)
        super(Kernel, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(shape       = (self.kernelSize,1,1,1),
                                      initializer = self.kernelInit,
                                      trainable   = self.trainKern, name='kernel')
        super(Kernel, self).build(input_shape)
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernelInit": self.kernelInit,
            "trainKern": self.trainKern,
        })
        return config
    def call(self, inputs):
        return self.kernel
    
class Experiment:
    def __init__(self):
        pass 
    def execute_experiment(self, config_dict, model=None):
        '''
        If a model is passed, a Weavelet Style model will not be generated, that will be used instead

        Components of config_dict:
            DISUSED - val_holdout: amount of data to hold out to verify against after cross validation
            test_holdout: amount of data to hold out to verify against during cross validation
            exp_description: short text description of what this experiment is testing
            exp_name: a unique name for this experiment
            K: number of cross validation folds
            batch_size: batch size
            data_folder: where the MIT dataset is stored
            width: corresponds to the length of an input signal
            input_layer: optional arbritary Keras/tensorflow input layer
            stride: e.g (2,1) - the stride used in the convolution and deconvolutional operations. Second dimension must be 1
            padding: SAME or VALID, as per TF documentation
            dilations: e.g (2,1) Either tuple or list thereof, the dilations to be used at each level Defaults to None, cannot be used in conjunction with stride>1 
            levels: Number of decomposition/recomposition stacked layers to perform
            kernelInitMode: SharedQMF, SeparateQMF, Independent
                 1. QMF filters - each H and L filter, for deconstruction and reconstruction are related
                    a. Shared - Sharess the same filter is learned at all levels 
                    b. Separate - A unique filter is learned at each level
                2. Indepently
                    a. Learn all filters independtly
            initKernels: int or list, or list of int/list for SepQMF and Independent
            trainKernels: True or False
            activationLayer: None, or a custom keras activation layer
            activationLayerParams: a lsit of any arguments to pass to the activation layer
            performReconstruction: True or False, will decide if this is an encoder head or full autoencoder model 
        '''
        self.config = config_dict
        data_folder=config_dict.get("data_folder")
        if data_folder==None: data_folder = "../mit-bih-arrhythmia-database-1.0.0"
        signals, annots = self.load_data(data_folder)
        self.signals = signals 
        self.annots = annots
        
        if model==None:
            inputs, outputs = self.wavelet_cnn(config_dict)
            model = keras.models.Model(inputs, outputs)
        elif isinstance(model, str):
            if model == "RESIDUAL":
                i,o = self.residualCNN(config_dict)
                model = keras.models.Model(i,o)
            elif model == "STANFORD":
                i,o = self.StanfordResidualCNN(config_dict)
                model = keras.models.Model(i,o)
            elif model == "WAVELET_RESID":
                i,o = self.WaveletResidualCNN(config_dict)
                model = keras.models.Model(i,o)
            else:
                raise ValueError("UNKNOWN MODEL '"+model+"'")
        K = config_dict.get("K")
        if K==None: K=1
        
        name = config_dict.get("exp_name")
        if name==None: raise ValueError ("Specify an experiment name")

        exp_description = config_dict.get("exp_description")
        if exp_description==None: raise ValueError ("Specify an experiment description")

        epochs = config_dict.get("epochs")
        if epochs==None: raise ValueError ("Specify a number of training epochs")

        batch_size = config_dict.get("batch_size")
        if batch_size==None: raise ValueError ("Specify a batch_size")
        
        #val_holdout = config_dict.get("val_holdout")
        #if val_holdout==None: raise ValueError ("Specify a validation holdout percentage")
        
     
        self.hist = self.train_model(name, epochs, signals, annots, model, K, batch_size=batch_size, matched_batches=True)

        f = open(f"./Tensorboard/{name}/description.md", "w")
        f.write("# {name}\n")
        f.write(exp_description+'\n# Config')
        f.close()
        with open(f"./Tensorboard/{name}/config.json.pickle", 'wb') as f:
            pickle.dump(config_dict, f)
        with open(f"./Tensorboard/{name}/hist.dict.pickle", 'wb') as f:
            pickle.dump(self.hist, f)
        self.present_results(name, True, f"./Tensorboard/{name}")

    def WaveletResidualCNN(self, config_dict):
        '''
        use 5 or 8 blocks
        with a pooling of 10 or 2
        '''

        
        input_layer = config_dict.get("input_layer")
        if input_layer==None:
            input_layer = keras.layers.Input(shape=(4096,), name='input_Raw')

        kernels = config_dict.get("kernels")
        if kernels==None:
            kernels = 8

        blocks = config_dict.get('blocks')
        if blocks==None: blocks=5
        if blocks not in [1,2,3,4,5,6,8]:
           raise ValueError("Pick 1,2,3,4,5,6 or 8 blocks, things get messy otherwise")
        
        activationLayer = config_dict.get("activationLayer")
        activationLayerParams = config_dict.get("activationLayerParams")

        inputLayer = keras.layers.Reshape((4096,1))(input_layer)
        x1 = tf.keras.layers.Normalization()(inputLayer)

        '''
        y =  keras.layers.Conv1D(64, 16, strides=1, padding="same")(inputLayer)
        y = keras.layers.BatchNormalization()(y)
        x1 = TripleLinearAct(None, True, 0.05)(y)'''

        coeffs = []
        for block in range(blocks):
            conv_x = keras.layers.Conv1D(kernels*(block+1), 16, strides=1, dilation_rate=2, padding='same')(x1)
            bncx = keras.layers.BatchNormalization()(conv_x)
            relu_x = activationLayer(*activationLayerParams)(bncx)
            coeffs.append(relu_x)
            x1 = keras.layers.Dropout(0.2)(relu_x)


        for block in range(blocks):
            bncx = keras.layers.BatchNormalization()(x1)
            recon = keras.layers.Conv1D(1, 16, strides=1, padding='same')(coeffs[-(block+1)])
            recon = keras.layers.BatchNormalization()(recon)
            recon = activationLayer(*activationLayerParams)(recon)
            x1 = keras.layers.Add()([bncx, recon])
            x1 = keras.layers.Conv1DTranspose(kernels*(block+1), 16, strides=1, dilation_rate=2, padding='same')(x1)
            

        x1 = keras.layers.Conv1D(1, 16, strides=1, padding='same')(x1)
        y = activationLayer(*activationLayerParams)(x1)
        y = keras.layers.Flatten()(y)

        out = keras.layers.Activation(keras.activations.sigmoid)(y)
        reconstructed = keras.layers.Reshape((4096,),name='reconstructed')(out)
        
        return input_layer, reconstructed
    

    def StanfordResidualCNN(self, config_dict):
        '''
        use 5 or 8 blocks
        with a pooling of 10 or 2
        '''

        
        input_layer = config_dict.get("input_layer")
        if input_layer==None:
            input_layer = keras.layers.Input(shape=(4096,), name='input_Raw')


        blocks = config_dict.get('blocks')
        if blocks==None: blocks=5
        if blocks not in [1,2,3,4,5,6,8]:
           raise ValueError("Pick 1,2,3,4,5,6 or 8 blocks, things get messy otherwise")
        

        inputLayer = keras.layers.Reshape((4096,1))(input_layer)
        inputLayer = tf.keras.layers.Normalization()(inputLayer)


        y =  keras.layers.Conv1D(64, 16, strides=1, padding="same")(inputLayer)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation(keras.activations.relu)(y)
        
        for block in range(blocks):
            print(block)
            x1 = y
            conv_x = keras.layers.Conv1D(64*(block+1), 16, strides=1, padding='same')(x1)
            bncx = keras.layers.BatchNormalization()(conv_x)
            relu_x = keras.layers.Activation(keras.activations.relu)(bncx)
            dp = keras.layers.Dropout(0.2)(relu_x)
            conv2 = keras.layers.Conv1D(64*(block+1), 16, strides=2, padding='same')(dp)
            x1 = keras.layers.MaxPooling1D(2, padding="same")(x1)
            x1 = keras.layers.Conv1D(64*(block+1), 16, strides=1, padding='same')(x1)
            y=keras.layers.Add()([x1, conv2])
        
        y =  keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation(keras.activations.relu)(y)
        y = keras.layers.Flatten()(y)
        #return input_layer, y
        rsShape = [None,131072,131072,98304,65536,40960,24576 ,None ,8192]
        
        y = keras.layers.Reshape((rsShape[blocks],1))(y)
        y = keras.layers.AveragePooling1D(int(rsShape[blocks]/4096), padding="same")(y)
        
        out = keras.layers.Activation(keras.activations.sigmoid)(y)
        reconstructed = keras.layers.Reshape((4096,),name='reconstructed')(out)
        
        return input_layer, reconstructed

    def residualCNN(self, config_dict):
        '''
        width:
        n_feature_maps
        levels: number of blocks
        input_layer
        convOnResidualConnect: true/false - do we convolve on the residual connection
        initKernels: kernel sizes across the block, of length 3 or 4
        '''
        n_feature_maps = config_dict.get("n_feature_maps")
        if n_feature_maps==None: n_feature_maps=1

        width = config_dict.get('width')
        if width==None:
            width=4096
        nb_classes=width
        
        input_layer = config_dict.get("input_layer")
        if input_layer==None:
            input_layer = keras.layers.Input(shape=(4096,), name='input_Raw')


        blocks = config_dict.get('levels')
        if blocks==None: blocks=4

        convOnResidualConnect = config_dict.get('convOnResidualConnect')
        if convOnResidualConnect==None:
            convOnResidualConnect=False
            
        initKernels = config_dict.get("initKernels")
        if initKernels==None:
            initKernels = [8,5,3]
        elif not convOnResidualConnect and len(initKernels)!=3:
            raise ValueError("Specify 3 inital kernel sizes per block operation")
        elif convOnResidualConnect and len(initKernels)!=4:
            raise ValueError("Specify 4 inital kernel sizes per block operation (incl one for the residual connection)")
        
        inputLayer = tf.keras.layers.Reshape((width,1,1))(input_layer)

        inpBN = config_dict.get("inputBatchNorm")
        if inpBN:
            inputLayer = tf.keras.layers.Normalization()(inputLayer)

        y =  inputLayer

        activationLayer = config_dict.get("activationLayer")
        activationLayerParams = config_dict.get("activationLayerParams")

        
        for block in range(blocks):
            x1 = y
            conv_x = keras.layers.Conv2D(n_feature_maps, initKernels[0], 1, padding='same')(x1)
            #conv_x = keras.layers.Activation('relu')(conv_x)

            if activationLayer!=None:
                conv_x = activationLayer(*activationLayerParams)(conv_x)
            
            conv_y = keras.layers.Conv2D(n_feature_maps, initKernels[1], 1, padding='same')(conv_x)
            #conv_y = keras.layers.Activation('relu')(conv_y)
            if activationLayer!=None:
                conv_y = activationLayer(*activationLayerParams)(conv_y)
            
            conv_z = keras.layers.Conv2D(n_feature_maps, initKernels[2], 1, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)
            
            if convOnResidualConnect:
                shortcut_y = keras.layers.Conv2D(initKernels[3], 1, 1,padding='same')(x1)
            else:
                shortcut_y = x1
            y = keras.layers.Add()([shortcut_y, conv_z])
            #y = keras.layers.Activation('relu')(y)
            if activationLayer!=None:
                y = activationLayer(*activationLayerParams)(y)
       
        full = keras.layers.GlobalAveragePooling2D()(y)
        useSigmoidOnOutput = config_dict.get("useSigmoidOnOutput")
        if useSigmoidOnOutput:
            out = keras.layers.Dense(width,activation='sigmoid')(full)
        else:
            out = keras.layers.Dense(width,activation=None)(full)
        reconstructed = tf.keras.layers.Reshape((width,),name='reconstructed')(out)

        return input_layer, reconstructed


    def wavelet_cnn(self, config_dict):
        '''
        Config dict will describe the parameters of an experiment, and 
        this function will return a wavelet CNN model that fits these 
        definitions, with the following optional parameters

        width: corresponds to the length of an input signal

        input_layer: optional arbritary Keras/tensorflow input layer  

        stride: e.g (2,1) - the stride used in the convolution and deconvolutional operations
        padding:
        dilations

        levels:

        kernelInitMode: SharedQMF, SeparateQMF, Independent

        initKernels: int or list, or list of int/list for sepQMF and Indep
        trainKernels:

        activationLayer: posiby NONE
        activationLayerParams:
        performReconstruction:
        '''

        width = config_dict.get("width")
        if width==None:
            width = 4096
            print("INFO: Defaulting to default width", width)


        input_layer = config_dict.get("input_layer")
        if input_layer==None:
            input_layer = keras.layers.Input(shape=(4096,), name='input_Raw')

        inputLayer = tf.keras.layers.Reshape((width,1,1))(input_layer)

        inpBN = config_dict.get("inputBatchNorm")
        if inpBN:
            inputLayer = tf.keras.layers.BatchNormalization()(inputLayer)

        levels = config_dict.get("levels")
        if levels==None:
            levels = 1
        
        stride = config_dict.get("stride")
        if stride == None:
            stride = 2

        padding = config_dict.get("padding")
        if padding == None:
            padding = 'SAME'

        dilations = config_dict.get("dilations")
        dilation_list = []
        if isinstance(dilations, list):
            if len(dilations)!=levels:
                raise ValueError("Must specify the same number of dilations as levels, or one dilation for all levels")
            for d in dilations: 
                dilation_list.append(d)
        else: 
            for _ in range(levels): 
                dilation_list.append(dilations)

        H_down_filters = []
        L_down_filters = []
        H_up_filters = []
        L_up_filters = []

        '''
        Lets allow for 3 experimental scenarios

        1. QMF filters - each H and L filter, for deconstruction and reconstruction are related
            a. Share the same filter is learned at all levels 
            b. A unique filter is learned at each level
        2. Non-qmf 
            a. Learn all filters independtly

        Option 1a requires a single integer or list as input - the size or value of the initial kernel respectively
        Options 1b and 2a require a list of lists or ints - corresponding to the sizes or inital weights for each layer
            respectively 
        '''

        kernelInitMode = config_dict.get("kernelInitMode")
        initKernels    = config_dict.get("initKernels")
        trainableKerns  = config_dict.get("trainKernels")

        activationLayer = config_dict.get("activationLayer")
        activationLayerParams = config_dict.get("activationLayerParams")
        if trainableKerns==None:
            trainableKerns=True 

        if kernelInitMode=="SharedQMF":
            K = Kernel(initKernels, trainableKerns)(inputLayer)
            for level in range(levels):
                H_down_filters.append(K)
                H_up_filters.append(K)
                L_down_filters.append(K)
                L_up_filters.append(K)
            L_down_filters = L_down_filters[::-1]
            L_up_filters = L_up_filters[::-1]
        elif kernelInitMode=="SeparateQMF":
            for level in range(levels):
                K = Kernel(initKernels[level], trainableKerns)(inputLayer)
                H_down_filters.append(K)
                H_up_filters.append(K)
                L_down_filters.append(K)
                L_up_filters.append(K)
            L_down_filters = L_down_filters[::-1]
            L_up_filters = L_up_filters[::-1]

        elif kernelInitMode=="Independent":
            for level in range(levels):
                K = Kernel(initKernels[level], trainableKerns)(inputLayer)
                H_down_filters.append(K)
                K = Kernel(initKernels[level], trainableKerns)(inputLayer)
                H_up_filters.append(K)
                K = Kernel(initKernels[level], trainableKerns)(inputLayer)
                L_down_filters.append(K)
                K = Kernel(initKernels[level], trainableKerns)(inputLayer)

                L_up_filters.append(K)
            L_down_filters = L_down_filters[::-1]
            L_up_filters = L_up_filters[::-1]
        else: 
            raise ValueError('Choose kernelInitMode as one of "SharedQMF", "SeparateQMF", "Independent"') 
        
        wavelet_coefficients = []
        sizes                = []
        last_step = inputLayer
        
        ###############################
        ####### decomposition  ########
        ###############################
        for level in range(levels):
            # high pass
            # no point in performing the reversse alternate flip 
            # if the kernels aren't shared
            dilations = dilation_list[level]
            sizes.append(tf.shape(last_step))
            coef = HardCodableCNN(enforce_reverse_alternate=0, enforce_reverse=False,\
                stride=stride, dilations=dilations, padding=padding)([last_step, H_down_filters[level]]) 
            if activationLayer!=None:
                if not isinstance(activationLayerParams, list):
                    raise ValueError("must define activationLayerParams list if activationLayer is non-null")
                coef = activationLayer(*activationLayerParams)(coef)
            wavelet_coefficients.append(coef)
            # low pass
            if kernelInitMode=="SharedQMF" or kernelInitMode=="SeparateQMF":
                last_step = HardCodableCNN(enforce_reverse_alternate=1,enforce_reverse=True,\
                    stride=stride, dilations=dilations, padding=padding)([last_step, L_down_filters[level]])
            else:
                last_step = HardCodableCNN(enforce_reverse_alternate=False,enforce_reverse=False,\
                    stride=stride, dilations=dilations, padding=padding)([last_step, L_down_filters[level]])
        
        # apply activation to final coefficient - i.e. embedding
        if activationLayer!=None:
                last_step = activationLayer(*activationLayerParams)(last_step)

        ###############################
        ####### recomposition  ########
        ###############################
        reconstruct = config_dict.get("performReconstruction")
        if reconstruct!=False:
            for level in range(levels):
                dilations = dilation_list[-(level+1)]
                #print(level, (-(level+1), len(wavelet_coefficients), len(H_up_filters), len(sizes)), flush=True)
                if kernelInitMode=="SharedQMF" or kernelInitMode=="SeparateQMF":
                    h = HardCodableInverseCNN(enforce_reverse_alternate=0, enforce_reverse=True,\
                        stride=stride, dilations=dilations, padding=padding)([wavelet_coefficients[-(level+1)], H_up_filters[level],sizes[-(level+1)]])
                else: 
                    h = HardCodableInverseCNN(enforce_reverse_alternate=0, enforce_reverse=False,\
                        stride=stride, dilations=dilations, padding=padding)([wavelet_coefficients[-(level+1)], H_up_filters[level],sizes[-(level+1)]])
                    
                if kernelInitMode=="SharedQMF" or kernelInitMode=="SeparateQMF":
                    last_step = HardCodableInverseCNN(enforce_reverse_alternate=2, enforce_reverse=False,\
                        stride=stride, dilations=dilations, padding=padding)([last_step, L_up_filters[level],sizes[-(level+1)]])
                else:
                    last_step = HardCodableInverseCNN(enforce_reverse_alternate=0, enforce_reverse=False,\
                        stride=stride, dilations=dilations, padding=padding)([last_step, L_up_filters[level],sizes[-(level+1)]])

                last_step = keras.layers.Add()([last_step, h])

        
        useSigmoidOnOutput = config_dict.get("useSigmoidOnOutput")
        if useSigmoidOnOutput:
            reconstructed = tf.keras.layers.Reshape((width,))(last_step)
            reconstructed = tf.keras.layers.Activation(activation='sigmoid',name='reconstructed')(reconstructed)
        else:
            reconstructed = tf.keras.layers.Reshape((width,),name='reconstructed')(last_step)
        
        return (input_layer, wavelet_coefficients+[reconstructed])

    def calculae_steps(self, signals, annots, epochs, target_batch_size=8, input_width=4096):
        N = int(len(signals))
        samples = sum([(len(x)//input_width)//target_batch_size for x in signals])
        return np.floor(samples/target_batch_size)

    def val_batcher(self, signals, annots, epochs, input_width=4096):
        full_train_signals =  signals

        full_train_signals =  signals
        full_train_annots  =  annots
    
        for x, a in zip(full_train_signals, full_train_annots):
            start_a_idx = 0
            full_anot = np.zeros(len(x))
            full_anot[a] = 1.0
            for i in range(input_width, len(x), input_width):
                X=x[i-input_width:i]
                y=full_anot[i-input_width:i]
                yield X, y

    def batcher(self, signals, annots, epochs, target_batch_size=8, input_width=4096):
        full_train_signals =  signals
        full_train_annots  =  annots

        cached = []

        current_batch_x = []
        current_batch_y = []
        for x, a in zip(full_train_signals, full_train_annots):
            start_a_idx = 0
            full_anot = np.zeros(len(x))
            full_anot[a] = 1
            for i in range(input_width, len(x), input_width):
                current_batch_x.append(x[i-input_width:i])
                current_batch_y.append(full_anot[i-input_width:i])

                if len(current_batch_x)==target_batch_size:
                    cached.append(( np.array(current_batch_x), np.array(current_batch_y)))
                    yield np.array(current_batch_x), np.array(current_batch_y)
                    current_batch_x = []
                    current_batch_y = []

            if len(current_batch_x)!=0:
                #yield np.array(current_batch_x),np.array(current_batch_y)
                current_batch_x = []
                current_batch_y = []

        for e in range(1, epochs):
            for x in cached:
                yield x 

    def load_data(self, base_dir="mit-bih-arrhythmia-database-1.0.0"):
        import wfdb
        signals=[]
        annots=[]

        record_file = open(base_dir+"/RECORDS", "r")
        records = [int(x.strip()) for x in record_file.readlines() if x.strip().isnumeric()]

        for i in records:
            signals.append( wfdb.rdsamp(f"{base_dir}/{i}") )
            annots.append( wfdb.rdann(f"{base_dir}/{i}", 'atr') )

        return np.array([s[0][:,0] for s in signals]), np.array([a.sample for a in annots], dtype=object)

    def train_model(self, name, epochs, signals, annots, model, K=1, batch_size=8, matched_batches=True):
        # name of this model for logging purposes

        # train using K fold validation for given K

        # matched_batches forces a constraint that series in the same batch correspond
        # to ordered samples of the same ecg, incorporating the batch_size specified
        # if the number of available samples for a patient is not a multiple of
        # batch_size, the excess may be processed in a smaller batch - samples from different
        # ecgs will never feature in the same batch
        #n_per_fold = ((1-val_holdout)*len(signals))//K
        #folds_idxs = [-1]*int((val_holdout)*len(signals))

        n_per_fold = len(signals)//K
        folds_idxs = []

        for x in range(K):
            folds_idxs+=[x]*n_per_fold
        folds_idxs+=[i for i in range(K)][:(len(signals)%K)]
        folds_idxs=folds_idxs[:len(signals)]
        assert len(folds_idxs)==len(signals)
        folds_idxs=np.array(folds_idxs)
        np.random.shuffle(folds_idxs)

        history = {}
        history['K'] = K
        for fold in range(K):
            foldpadded = '0'*(3-len(str(fold)))+str(fold)
            print("=====================")
            print("===== Fold "+foldpadded+' ======')
            print("=====================")
            chosen_idxs = np.where(folds_idxs!=fold)
            signal_fold = signals[chosen_idxs]
            annot_fold = annots[chosen_idxs]
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),\
            loss={"reconstructed": tf.keras.losses.BinaryCrossentropy()})

            tbcb = keras.callbacks.TensorBoard(log_dir=f'./Tensorboard/{name}/logs-fold-{foldpadded}', histogram_freq=0,  
                    write_graph=True, write_images=True)
            

            if K>1:
                #print(signals[chosen_idxs].shape)
                #print(signals[np.where(folds_idxs!=fold)].shape)
                
                xval, yval = zip(*list(self.val_batcher(signals[np.where(folds_idxs==fold)], annots[np.where(folds_idxs==fold)], epochs, 4096)))
                val_data = (np.array(xval), np.array(yval))
            else:
                val_data=None

            #testX, testY = zip(*list(self.batcher(signals[test_idxs], annots[test_idxs], epochs, target_batch_size=batch_size,  input_width=4096)))
            #print(np.array(testX).shape,np.array(testY).shape)
            steps=self.calculae_steps(signal_fold, annot_fold, epochs, target_batch_size=batch_size, input_width=4096)
            H=model.fit(self.batcher(signal_fold, annot_fold, epochs, target_batch_size=batch_size,  input_width=4096), \
                        epochs=epochs, steps_per_epoch=steps,verbose=0,callbacks=[tbcb],\
                         validation_data = val_data)   
            history[fold] = {'hist':H.history}
            model.save(f"./Tensorboard/{name}/models/{foldpadded}-model")
            history["fold_split"] = folds_idxs
        

        return history
    

    def present_results(self, exp_name, save_results=False, dir=None):
        import matplotlib.pyplot as plt

        if save_results and dir==None:
            raise ValueError("Specify a location to save the plotted results")

        fig = plt.figure(figsize=(12,15))

        
        fig.tight_layout()
         
        ax = fig.add_subplot(421)
        for i in range(self.hist['K']):
            ax.plot(self.hist[i]['hist']['loss'])
            ax.title.set_text('Training Loss')
            ax.set_xlabel("Epoch")
            ax.xaxis.set_label_coords(0.5,0.075)
        ax = fig.add_subplot(423, sharex=ax)    
        for i in range(self.hist['K']):
            ax.plot(self.hist[i]['hist']['val_loss'])
            ax.title.set_text('Validation Loss')
            ax.set_xlabel("Epoch")
            ax.xaxis.set_label_coords(0.5,-0.1)
            
        ax = fig.add_subplot(322)   
        ax.boxplot([[self.hist[i]['hist']['loss'][-1] for i in range(self.hist['K'])],[self.hist[i]['hist']['val_loss'][-1] for i in range(self.hist['K'])]])
        ax.set_xticklabels(["Train", "Val"])
        ax.set_ylabel("Loss")

        
        i = np.random.choice(list(range(self.hist['K'])))
        sigI = np.random.choice(np.where(self.hist["fold_split"]==i)[0])

        internalSigI =  np.random.randint(0, len(self.signals[sigI])-4097)
        testS = np.array(self.signals[sigI][internalSigI:internalSigI+4096])

        foldpadded = '0'*(3-len(str(i)))+str(i)
        mod = keras.models.load_model( f"./Tensorboard/{exp_name}/models/{foldpadded}-model")
        trainable_w      = count_params(mod.trainable_weights)
        untrainable_w    = count_params(mod.non_trainable_weights)
        fig.suptitle(f"{exp_name}: {trainable_w} trainable, {untrainable_w} non-trainable parameters", fontsize=20)

        pred = np.array(mod(testS.reshape(1,4096))[-1]).reshape(-1)
        full_annot = np.zeros(len(self.signals[sigI]))
        full_annot[self.annots[sigI]] = 1.0
        true_annot = full_annot[internalSigI:internalSigI+4096]
        ax = fig.add_subplot(413) 
        ax.title.set_text('Sample Train Prediction') 
        ax.plot(testS, label="Input ECG")
        ax.plot(pred, label="Detected R Peaks")
        ax.scatter(np.where(true_annot>0.1), testS[np.where(true_annot>0.1)], c='r', label="True R Peaks")
        ax.legend()

        ax = fig.add_subplot(414) 
        ax.title.set_text('Sample Validation Prediction') 
        i = np.random.choice(list(range(self.hist['K'])))
        sigI = np.random.choice(np.where(self.hist["fold_split"]!=i)[0])
        internalSigI =  np.random.randint(0, len(self.signals[sigI])-4097)
        testS = np.array(self.signals[sigI][internalSigI:internalSigI+4096])
        pred = np.array(mod(testS.reshape(1,4096))[-1]).reshape(-1)
        full_annot = np.zeros(len(self.signals[sigI]))
        full_annot[self.annots[sigI]] = 1.0
        true_annot = full_annot[internalSigI:internalSigI+4096]
        ax.plot(testS)
        ax.plot(pred)
        ax.scatter(np.where(true_annot>0.1), testS[np.where(true_annot>0.1)], c='r')
        
        if save_results:
            fig.savefig(dir+"/result_plots.png")



if __name__ == "__main__":
    # testing
    signals, annots = load_data()
    config_dict={
         "width": 4096,
        "input_layer": None,  
        "stride": 2,
        "padding": "SAME",
        "dilations" : None,
        "levels": 3,
        "kernelInitMode": "SharedQMF",
        "initKernels": [-0.7071067811865476,0.7071067811865476],
        "trainKernels": False,
        "activationLayer": None,
        "activationLayerParams":None,
        "performReconstruction":True
    }
    ins,outs  = wavelet_cnn(config_dict)
    model = keras.models.Model(ins, outs[-1])
    train_model("trialTestbed", 1, signals, annots, model)