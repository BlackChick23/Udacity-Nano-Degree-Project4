import cv2                
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from extract_bottleneck_features import *
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

# define ResNet50 model

# load pretrained Resnet50 model
ResNet50_model = ResNet50(weights='imagenet')

# load face_cascade model
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load my trained model
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
model.add(Dense(133, activation='softmax'))
model.load_weights('saved_models/weights.best.Resnet50.hdf5')


with open("dog_classes.pkl", "rb") as f:
    dog_classes = pickle.load(f) 

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    print("feature:", bottleneck_feature.shape)
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_classes[np.argmax(predicted_vector)]

def predict_dog(img_path):
    # if a dog is detected in the image, return the predicted breed.
    # if a human is detected in the image, return the resembling dog breed.
    if dog_detector(img_path) == True or face_detector(img_path) == True:
        dog_breed = Resnet50_predict_breed(img_path)
        dog_breed = dog_breed
        return dog_breed
    # if neither is detected in the image, provide output that indicates an error.
    else:
        return "Error, Can't classify"
    
if __name__ == "__main__":
    print(predict_dog("cat1.jpg"))