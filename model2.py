import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from scipy import misc
import pdb
from keras.utils import np_utils
from random import randint
from keras import backend as K
import os

nb_filters = 48
batch_size = 128
nb_epoch = 20
maxImg = 50
imgSize = (24,32)
img_rows = imgSize[0]
img_cols = imgSize[1]
eighty = int(maxImg * 0.2)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    print x
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def loadimage(id):
    imgname ="highway/input/in" + "{0:06.0f}".format(id) + ".jpg"
    img = misc.imread(imgname)
    img = misc.imresize(img,imgSize)
    return rgb2gray(img)

def loadgroundtruth(id):
    imgname ="highway/groundtruth/gt" + "{0:06.0f}".format(id) + ".png"
    img = misc.imread(imgname)
    img = misc.imresize(img,imgSize)
    return img

def loadimagesforid(id):
    return np.array(([loadimage(id-2),loadimage(id-1),loadimage(id)]))

def loadallimages(start,max):
    images = []
    for i in range(start + 3,start+max): #in labels, img start at 1
        images.append(loadimagesforid(i))
    return np.asarray(images)

def loadallground(start,max):
    ground = []
    for i in range(start + 3,max):
        ground.append(loadgroundtruth(i))
    return np.asarray(ground)

def getClass((img1,img2,img3)):
    x = (img1 - img2) + (img1 - img2)
    y = np.sum(np.absolute(x))
    #print y
    return 1 if (y/2) > 400 else 0

def processImg(imgs):
    classe = []
    for i in xrange(len(imgs)):
        classe.append(getClass(imgs[i]))
    return classe

def circle_buf((a,b,c), d):
    return (d,a,b)

#img are 240 height by 320 width

#x = np.stack((img1,img2,im3))
#x = x.reshape(240,320,3)
 #3 images stacked


def visualise_first_layer(model, convout1, test_data,p,l):
    path = "output/" + str(p) + "_" + str(l)
    get_layer_output = K.function([model.layers[0].input], [convout1.get_output_at(0)])
    layer_output = get_layer_output([test_data])[0]
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(layer_output.shape[1]):
        misc.imsave(path+'/%d_filter_%d_%d.png' % (p,l,i),layer_output[0,i])


layer_name = "conv1"
filter_index = 2

print("data loaded")
#pdb.set_trace()
model = Sequential()

model.add(Convolution2D(nb_filters, 7, 7,
                        border_mode='valid',
                        input_shape=(3, img_rows, img_cols),name="conv1"))
convout1 = Activation('relu')
model.add(convout1)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(nb_filters, 7, 7))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(nb_filters, 1, 1))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(nb_filters, 1, 1))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(img_rows*img_cols))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
layer_dict = dict([(layer.name, layer) for layer in model.layers])
model.compile(loss="binary_crossentropy", optimizer='adadelta')




print("model loaded")
for p in range(100):
    #nbimage divide by maxImg by batch
    for l in range(1700/maxImg):
        print l
        X_train = loadallimages(l * (maxImg - 2), maxImg)
        #print X_train.shape
        Y_train = np.asarray(processImg(X_train))
        #print Y_train.shape
        if p % 50 == 0 and l %50 == 4 :
            visualise_first_layer(model,convout1,X_train,p,l)
        loss = model.train_on_batch(X_train,Y_train,accuracy=True)
        print loss

score = model.evaluate(X_train[-eighty:], Y_train[-eighty:], show_accuracy=True, verbose=1)
print('Test score:', score)
