  
import json
import logging
import sys
import boto3
import os

import wave
import wget
import zipfile
from vosk import Model, KaldiRecognizer, SetLogLevel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info('Imported all libraries successfuly')

if not os.path.exists('data'):
        os.makedirs('data')

        
print('Beginning file download with wget module')
url = 'http://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip'
wget.download(url)

import zipfile
with zipfile.ZipFile('vosk-model-small-en-us-0.15.zip', 'r') as zip_ref:
    zip_ref.extractall()

def model_fn(model_dir):
    logger.info('In model_fn')
    
    model_name = 'vosk-model-small-en-us-0.15'
    model = Model(model_name)
    
    logger.info(f'Done loading model')
    return model

# data preprocessing
def input_fn(request_body, request_content_type):
    logger.info('In input_fn')
    assert request_content_type=='application/json'
    data = json.loads(request_body)
    logger.info('Done reading Json file')
    logger.info(data)
    return data


# inference
def predict_fn(input_object, model):
    logger.info('In predict_fn')
    if type(input_object) != dict:
        input_object = json.loads(input_object)
    logger.info(input_object)
    audio_path = input_object['audio_path']
    local_path = os.path.join('data', audio_path.split('/')[-1])
    
    s3 = boto3.resource('s3')
    s3.Bucket(input_object['bucket']).download_file(audio_path, local_path)
    logger.info(f'Downloaded file from s3 and start preprocessing file : {local_path}')
    
    wf = wave.open(local_path, "rb")
    os.remove(local_path)
    
    rec = KaldiRecognizer(model, wf.getframerate())
    
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            print(rec.Result())

    res = json.loads(rec.FinalResult())

    return {'transcription':res['text']}


def output_fn(predictions, content_type):
    logger.info('In output_fn')
    assert content_type == 'application/json'
    logger.info(f'Done making json file with type : {content_type}')
    return json.dumps(predictions) , content_type