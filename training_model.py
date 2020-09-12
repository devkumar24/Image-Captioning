# LOAD LIBRARIES OF PYTHON

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import pickle
import collections
import os
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras.layers import *
from keras.preprocessing.seqeunce import pad_sequences
from keras.layers.merge import add
from keras.models import load_model



# PREPROCESSING OF IMAGES
# PATH OF IMAGES DATA
path = "Dataset/Images/"
img = os.listdir(path)

#LOADING PRE-TRAINED MODEL(USING TRANSFER LEARNING)
model = ResNet50(weights = "imagenet",input_shape = (224,224,3))

#NEW MODEL USING FUNCTIONAL MODEL
model_new = Model(model.input,model.layers[-2].output)

# PREPROCESSING IMAGE 
def preprocess_image(img):
	# LOAD IMAGE USING KERAS
    img = image.load_img(img, target_size=(224,224))
    # CONVERTING IMAGE TO ARRAY
    img = image.img_to_array(img)
    # RESHAPE TO PARTICULAR FORMAT OF IMAGE
    img = img.reshape((1,224,224,3))
    # GIVUNG THE INPUT TO model_new
    img = preprocess_input(img)
    
    return img

# FINDING ENCODING OF IMAGES
def encoding_image(img):
	# PREPROCESSING_IMAGE FUNCTION IS USED
    img = preprocess_image(img)
    # PRDICTING OUTPUT OF IMAGE
    features_of_image = model_new.predict(img)
    # RESHAPING TO DESIRED FORMAT
    features_of_image = features_of_image.reshape(features_of_image.shape[1],)
    
    return features_of_image

# LOADING TRAINING, VALIDATING AND TESTING DATA
f_train = open("E:/Data Science/Projects Data Science/Image Captioning/Dataset/Flickr_TextData/Flickr_8k.trainImages.txt")
training_files = f_train.read()
f_train.close()

f_validation = open("E/Data Science/Projects Data Science/Image Captioning/Dataset/Flickr_TextData/Flickr_8k.devImages.txt")
validation_files = f_validation.read()
f_validation.close()

f_test = open("E:/Data Science/Projects Data Science/Image Captioning/Dataset/Flickr_TextData/Flickr_8k.testImages.txt")
testing_files = f_test.read()
f_test.close()


# LOADING TRAINIG IMAGES NAME
training_files = [file_name[:-4] for file_name in training_files.split("\n")]

# LOADING VALIDATING IMAGES NAME
validation_files = [file_name[:-4] for file_name in validation_files.split("\n")]

# LOADING TESTING IMAGES NAME
testing_files = [file_name[:-4] for file_name in testing_files.split("\n")]


# PREPARING ENCODING OF ALL TRAINING IMAGES
training_image = {}

for num,img_file in enumerate(training_files):
    img_file = path + "{}.jpg".format(img_file)
    file_name = img_file[len(path):]
    training_image[file_name] = encoding_image(img_file)
    
    if num%100 == 0:
        print("Encoded-Images: {}".format(num))

# SAVING ENCODING OF TRAINING IMAGES FOR FUTURE USE
with open("./trained_data/encoding_train_images.pkl","wb") as encoded:
    pickle.dump(training_image,encoded)


# PREPARING ENCODING OF ALL VALIDATING IMAGES
validating_image = {}

for num,img_file in enumerate(validation_files):
    img_file = path + "{}.jpg".format(img_file)
    file_name = img_file[len(path):]
    validating_image[file_name] = encoding_image(img_file)
    
    if num%100 == 0:
        print("Encoded-Images: {}".format(num))

# SAVING ENCODING OF VALIDATING IMAGES FOR FUTURE USE
with open("./trained_data/encoding_validation_images.pkl","wb") as encoded:
    pickle.dump(validating_image,encoded)


# PREPARING ENCODING OF ALL VALIDATING IMAGES
testing_image = {}

for num,img_file in enumerate(testing_files):
    img_file = path + "{}.jpg".format(img_file)
    file_name = img_file[len(path):]
    testing_image[file_name] = encoding_image(img_file)
    
    if num%100 == 0:
        print("Encoded-Images: {}".format(num))

# SAVING ENCODING OF VALIDATING IMAGES FOR FUTURE USE
with open("./trained_data/encoding_test_images.pkl","wb") as encoded:
    pickle.dump(testing_image,encoded)


# ONCE THE ABOVE ENCODING IS BEING MADE OF IMAGES THEN WE LOAD THESE IMAGES BY USING FILES WE CREATED
with open("./trained_data/encoding_train_images.pkl","rb") as encoded_img:
	encoding_train_images = pickle.load(encoded_img)

with open("./trained_data/encoding_test_images.pkl","rb") as encoded_img:
	encoding_test_images = pickle.load(encoded_img)

with open("./trained_data/encoding_validation_images.pkl","rb") as encoded_img:
	encoding_validation_images = pickle.load(encoded_img)


# PREPROCESSING OF TEXT DATA(CAPTIONS)

# LOAD CAPTIONS FILE
path_of_captions= "./Dataset/Flickr_TextData/Flickr8k.token.txt"
#OPEN CAPTIONS FILE
with open(path_of_captions) as file:
    captions = file.read()
    file.close()

# SPLITING OF CAPTIONS 
captions = captions.split("\n")

# MAKE DESCRIPTION OF ALL CAPTIONS
descriptions = {}
for i in captions:
    i = i.split("\t")
    img_name = i[0]
    img_name = img_name.split(".")[0]
    cap = i[1]
    if descriptions.get(img_name) == None:
        descriptions[img_name] = []
    descriptions[img_name].append(cap)


# FUNCTION TO PREPROCESS TEXT AND REMOVE UNWANTED WORDS AND PUNCTUATIONS
def clean_data(text):
    """
    Lower Text
    remove punctuations
    remove words whose len is less than 2
    """
    text = text.lower()
    text = re.sub("[^a-z]+"," ",text)
    text = [t for t in text.split() if len(t)>1]
    text = " ".join(text)
    
    return text


# MAKE CAPTIONS FREE OF UNWANTED WORDS
for key,val in descriptions.items():
    for v in range(len(val)):
        descriptions[key][v] = clean_data(descriptions[key][v])


#write description file
f = open("description_of_captions.txt","w")
f.write(str(descriptions))
f.close()


# CREATE VOCAB OF ALL WORDS PRESENT IN DESCRIPTION FILE
vocab = []
for key,val in descriptions.items():
    for v in val:
        for word in v.split():
            vocab.append(word)


# CREATE VOACB OF ALL UNIQUE WORDS PRESENT IN DESCRIPTIONS
all_vocab = set()
for key,val in descriptions.items():
    for v in val:
        all_vocab.update(v.split())

# CALCULATE FREQUENCY OF EACH WORD AND MAKE DICTIONARY
counter = collections.Counter(vocab)
counter = dict(counter)
# NOW IF FREQ OF WORD IS LESS THAN 10 THEN DISCARD THAT WORD 
THRESHOLD_VALUE = 10
all_vocab = []
for k,v in counter.items():
    if counter[k] > THRESHOLD_VALUE:
        all_vocab.append(k)


# ADD SOS(STARTING OS SEQUENCE) AND EOS(END OF SEQUENCE) TO CAPTIONS
train_descriptions = {}
for t in training_files:
    train_descriptions[t] = []
    for cap in descriptions[t]:
        cap = "SOS " + cap + " EOS"
        train_desc[t].append(cap)


# CREATING WORD_TO_INDEX FILE AND INDEX_TO_WORD FILE 
index_to_word = {}
word_to_index = {}
i = 1
# 0 is used for padding
for voc in all_vocab:
    index_to_word[voc] = i
    word_to_index[i] = a
    i+=1

# ADD SOS AND EOS TO INDEX_TO_WORD AND WORD_TO_INDEX
word_to_index[1846] = "SOS"
word_to_index[1847] = "EOS"
index_to_word["SOS"] = 1846
index_to_word["EOS"] = 1847
# LENGTH OF ALL VOCAB IS 1848 BECAUSE ADD 0 ALSO i.e. USED FOR PADDING
len_vocab = len(word_to_index) + 1


# LENGTH OF CAPTIONS AND CALCULATING MAX LENGTH i.e. 35, AND AFTER PLOTTING THE HISTOGRAM WE GET TO KNOW THAT THERE ARE FEW CAPTIONS THAT HAVE LENGTH OF 35 SO WE DISCARD THAT LENGTH AND CHOOSE 24 AS LENGTH OF CAPTIONS
all_cap_len = []

for key, val in train_descriptions.items():
    for v in val:
        all_cap_len.append(len(v.split()))
        
print(max(all_cap_len))
plt.hist(all_cap_len, bins = 30)


# SAVE WORD_TO_INDEX AND INDEX_TO_WORD FILE FOR FURTHER USE
with open("./trained_data/word_to_index.pkl","wb") as wi:
    pickle.dump(word_to_index,wi)
with open("./trained_data/index_to_word.pkl","wb") as iw:
    pickle.dump(index_to_word,iw)


# CREATING DATASET AS PER WE WANT TO PREDICT THE OUTPUT
def data_generator(train_descriptions, encoding_train_images,index_to_word, num_of_images_per_batch,max_len = 35):
    X1,X2,y = [],[],[]
    n = 0
    while True:
        
        for key, val in train_descriptions.items():
            n+=1
            photo = encoding_train_images[key + ".jpg"]
            for v in val:
                seq = [index_to_word[i] for i in v.split() if i in index_to_word]
                for i in range(1,len(st)):
                    x2_seq = seq[:i]
                    y_seq = seq[i]
                    x2_seq = pad_sequences([x2_seq], max_len,padding="pre",value=0)[0]
                    y_seq = to_categorical([y_seq],num_classes=len_vocab)[0]
                    X1.append(photo)
                    X2.append(x2_seq)
                    y.append(y_seq)
                    
            if n == num_of_images_perbatch:
                yield [[np.array(X1),np.array(X2)],np.array(y)]
                X1,X2,y = [],[],[]
                n = 0


# CREATE WORD EMBEDDINGS 
f = open("./GloVe/glove.6B.50d.txt", encoding='utf8')
embedding_index = {}

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float")
    
    embedding_index[word] = coefs
    
f.close()

#CREATE EMBEDDING OF ALL_VOCAB(1848,50)
def get_embedding_output():
    
    emb_dim = 50
    embedding_output = np.zeros((len_vocab,emb_dim))
    
    for word, idx in word_to_index.items():
        embedding_vector = embedding_index.get(word)
        
        if embedding_vector is not None:
            embedding_output[idx] = embedding_vector
            
    return embedding_output

# MEBEDDING OUTPUT(1848,50)
embedding_output = get_embedding_output()


#MODEL
# IMAGE MODEL
CNN1 = Input(shape = (2048,))
CNN2 = Dropout(0.5) (CNN1)
CNN3 = Dense(256,activation="relu") (CNN2)

# CAPTIONS MODEL
RNN1 = Input(shape=(35,))
RNN2 = Embedding(len_vocab,50,mask_zero=True) (RNN1)
RNN3 = Dropout(0.5) (RNN2)
RNN4 = LSTM(256) (RNN3)

# MAIN MODEL
MLP1 = add([CNN3,RNN4])
MLP2 = Dense(256,activation="relu") (MLP1)
MLP3 = Dense(len_vocab,activation="softmax") (MLP2)

# MERGING BOTH IMAGE AND CAPTION MODEL
model = Model(inputs = [CNN1,RNN1],outputs = MLP3)


# MAKE EMBEDDING LAYER TO TRAINABLE FALSE AND RESET WEIGHTS TO EMBEDDING OUTPUT
model.layers[2].set_weights([embedding_output])
model.layers[2].trainable  =False

# COMPILE MODEL
model.compile(loss="categorical_crossentropy",optimizer = "adam")


# TRAIN MODEL
epochs = 10
num_of_images_per_batch = 4
steps = len(train_descriptions)//num_of_images_per_batch


# MODEL TRAINED FOR 10 EPOCHS AND SAVE AFTER EVERY EPOCH
for i in range(epochs):
    gen = data_generator(train_desc,encoding_train_images,index_to_word,num_of_images_perbatch)
    model.fit_generator(gen,steps,epochs=1,verbose=1)
    model.save("Image-Captioning_weights-{}.h5".format(str(i)))


##########END OF PROJECT##############