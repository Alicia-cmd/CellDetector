from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import cv2
import numpy as np
import pandas as pd
import os
import shutil

app = Flask(__name__)
context = ('/etc/apache2/certs/celldetectorfan.crt', '/etc/apache2/certs/celldetectorfan.key')

# Charger le modèle TensorFlow
model = tf.keras.models.load_model('Classif_Model')

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Téléchargez l'image à partir de la requête POST
    #fichier = request.files['image']
    #nom_fichier = fichier.filename
    fichiers = request.files.getlist('image[]')

    # Chemin vers le répertoire où vous souhaitez enregistrer l'image
    destination = 'images'
    
    # Vider le répertoire s'il contient déjà des images
    if os.path.exists(destination):
        shutil.rmtree(destination)
	
    # Vérifiez si le répertoire existe et créez-le s'il n'existe pas
    if not os.path.exists(destination):
        os.makedirs(destination)

    # Parcourir tous les fichiers sélectionnés et les copier dans le répertoire de destination
    for fichier in fichiers:
       nom_fichier = fichier.filename
       fichier.save(os.path.join(destination, nom_fichier))
    

    # Déplacez le fichier vers le répertoire de destination
    #fichier.save(os.path.join(destination, nom_fichier))
    
    #create a dataframe to run the predictions
    test_df = pd.DataFrame({'id':os.listdir(destination)})
    test_df.head()
    # prepare test data (in same way as train data)
    datagen_test = ImageDataGenerator(rescale=1./255.)
    
    test_generator = datagen_test.flow_from_dataframe(
    dataframe=test_df,
    directory=destination,
    x_col='id', 
    y_col=None,
    target_size=(64,64),         # original image = (96, 96) 
    batch_size=1,
    shuffle=False,
    class_mode=None)
    
    print("Prediction des images en cours....")
    # Check model
    predictions = model.predict(test_generator, verbose=1)
    
    #create submission dataframe
    predictions = np.transpose(predictions)[0]
    submission_df = pd.DataFrame()
    submission_df['id'] = test_df['id'].apply(lambda x: x.split('.')[0])
    submission_df['label'] = list(map(lambda x: 0 if x < 0.5 else 1, predictions))
    submission_df.head()
    
    #si le csv existe supprime
    if os.path.exists('resultat.csv'):
    	os.remove('resultat.csv')
    #convert to csv to submit to competition
    print("Generation en cours ....")
    submission_df.to_csv('resultat.csv', index=False)
    
    #if request.form.get('download') == '1':
    return send_from_directory(directory='/var/www/flask-prj1/', path='resultat.csv', as_attachment=True)
    #return redirect(url_for('index'))
    #prediction_result = True
    #return redirect(url_for('index'))

#@app.route('/download')
#def download():
    #return send_from_directory(directory='/var/www/flask-prj1/', filename='resultat.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=443,ssl_context=context)


