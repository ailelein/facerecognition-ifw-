from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface2.vggface import VGGFace
from keras_vggface2.utils import preprocess_input
import os
import cv2
from keras.models import load_model 
import numpy as np
def extract_face(filename, required_size=(224, 224)):
 
    pixels = pyplot.imread(filename)
    detector = MTCNN()

    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height 
    face = pixels[y1:y2, x1:x2]
    
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
 
def get_embeddings(filenames):
    faces = [extract_face(f) for f in filenames]
    samples = asarray(faces, 'float32')
    
    samples = preprocess_input(samples, version=2)
    #model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    model = load_model('model1.h5')

    #print(model.summary())

    yhat = model.predict(samples)
    #print('----saving...')
    #model.save('model1.h5')
    #print('----saved')
    return yhat


def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

def is_match(known_embedding, candidate_embedding, filename, thresh=0.5):
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('Match with \t\t', os.path.basename(filename))
    else:
        print('NOT a Match with \t', os.path.basename(filename))
 

filenames = ['lfw_funneled/1_mix/' + x for x in os.listdir('lfw_funneled/1_mix')]
filenames.sort()
embeddings = get_embeddings(filenames)
_id = embeddings[0]
#print('Tests')
for i, embedding in enumerate(embeddings[1:]):
    print('{:<30}'.format(os.path.basename(filenames[i+1])), end = ' --> ', flush=True)
    is_match(_id, embedding, filenames[0])
