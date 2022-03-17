import base64
from partieAlgoComplete import *
from PIL import Image
import io
import cv2
import base64




class Server:


    def __init__(self):
        self.query=None
        self.port=5000
        self.mdrv1=MDRV()
        self.mdrv2=MDRV()
        self.SIFT=DescripteurSIFT("SIFT")
        self.SURF=DescripteurSURF("SURF")
        self.DB=[]




    def B64_array(self,image):
        a = base64.b64decode(image)
        result = np.frombuffer(a, np.uint8)
        return cv2.imdecode(result, cv2.IMREAD_COLOR)

    def Obtention_Querypath(self, description):
        #matricecv2=Server.B64_array(self,description)
        query_image=Image.fromarray(description)
        query_image.save("test2.png","PNG")
        queryPath=##Query post-processing image path##
        return queryPath



    def array_B64(self, matrice):
        return  base64.b64encode(cv2.imencode('.jpg', matrice)[1])

    def send(self, queryPath, nbVoisins, choixMethode):

        img1 = ##Path of DB images##


        DB=[img1] #List of DB image paths
        self.DB=DB


        dbSURF=Database(self.DB,self.SIFT)
        dbSIFT=Database(self.DB,self.SURF)
        self.mdrv1.addDB(dbSIFT)
        self.mdrv2.addDB(dbSURF)



        return MDRV.sendResult(self.mdrv1, self.mdrv2, queryPath, self.DB, nbVoisins, choixMethode)



