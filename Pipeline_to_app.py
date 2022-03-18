from flask import Flask, jsonify
from flask import abort
from flask import make_response
from flask import request
from flask import Response
from Server_to_app import Server
import base64
import cv2
import json
import sys
import math



app = Flask(__name__)

tasks = [                                       
    {
        'id': 1,                              
        'titleimage': u'Image_1', 
        'b64image': u'Description_Image_1',
        'methode' : 0,  
        'done': False
    },
    {
        'id': 2,
        'title_image': u'Image_2',
        'b64image': u'Description_Image_2', 
        'methode' : 0,
        'done': False

    }
]






@app.route('/api/v1.0/tasks/mdrv', methods=['POST']) 



    

def send_client():
    server=Server()
    if not request.json:
        return Response(
        "ERREUR! Le body n'est pas au format JSON. Veuillez svp renvoyer un dictionnaire au format JSON",
        status=400,
    ) 

    if not 'b64image' in request.json:
        return Response(
        "ERREUR! Le champ imageB64 n'est pas fournie. Veuillez svp le renseigner pour obtenir un resultat",
        status=400,
    ) 
    Lresult_algo=[]
    JSON = request.get_json()         
    ids = JSON['id']
    methode= JSON['methode']
    description = JSON['b64image']
    description1=server.B64_array(description)
    queryPath=server.Obtention_Querypath(description1)
    Lresult_algo=server.send(queryPath, ids, methode)
    Lresult_b64 = []
    for matrice in Lresult_algo:
        Lresult_b64.append(server.array_B64(matrice))
    Lresult_b64=[str(b64) for b64 in Lresult_b64]
    

    data = {                                                   
            
            }
    c=0

    for elt in Lresult_b64:
        c=c+1
        data[c]= elt
    
    resp=json.dumps(data)
    return resp
    
                       

if __name__ == '__main__':              
    app.run(host=##Your IP adress##)       


