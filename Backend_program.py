#Bibliothèques nécessaires:

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from PIL import Image


#Classes:

class MDRV:

#Cette classe représente le moteur de recherche visuelle. Elle effectue la recherche et l'envoi des images les plus ressemblantes à une image requête.

    def __init__(self): #constructeur
        self.database=None
        self.query=None
        self.server=None

    def addDB(self,database): #instancie une database au constructeur
        self.database=database

    def searchSIFT(self,imgQuery,DB,nbVoisins,nbClusters=4): #renvoie les indices des 5 images de la database les plus ressemblantes à la query, selon le descripteur SIFT
        sift=DescripteurSIFT("SIFT")
        HQ=sift.createImageDesc(imgQuery)
        neighbor = NearestNeighbors(n_neighbors = nbVoisins)
        neighbor.fit(Database.listImagesDesc(self.database,DB,sift))
        dist, result = neighbor.kneighbors([HQ])
        return result, dist

    def searchSURF(self,imgQuery,DB,nbVoisins,nbClusters=4): #renvoie les indices des 5 images de la database les plus ressemblantes à la query, selon le descripteur SURF
        surf=DescripteurSURF("SURF")
        HQ=surf.createImageDesc(imgQuery)
        neighbor = NearestNeighbors(n_neighbors = nbVoisins)
        neighbor.fit(Database.listImagesDesc(self.database,DB,surf))
        dist, result = neighbor.kneighbors([HQ])
        return result, dist

    def compareDescripteur(mdrv,mdrv2,imageQuery,DB,nbVoisins,choixMethode,nbClusters=4): #renvoie les meilleurs résultats issues des recherches utilisant tous les descripteurs
        xSIFT,dSIFT=list(mdrv.searchSIFT(imageQuery,DB,nbVoisins)[0][0]),list(mdrv.searchSIFT(imageQuery,DB,nbVoisins)[1][0])
        if choixMethode==1:
            return xSIFT
        xSURF,dSURF=list(mdrv2.searchSURF(imageQuery,DB,nbVoisins)[0][0]),list(mdrv2.searchSURF(imageQuery,DB,nbVoisins)[1][0])
        if choixMethode==2:
            return xSURF
        xFinal=[]
        while(len(xFinal)<nbVoisins):
            if(min(dSIFT)<min(dSURF)):
                xFinal.append(xSIFT[0])
                xSIFT.remove(xSIFT[0])
                dSIFT.remove(dSIFT[0])
            else:
                xFinal.append(xSURF[0])
                xSURF.remove(xSURF[0])
                dSURF.remove(dSURF[0])
        #compare les distances et renvoie les images associées aux plus petites distances
        return xFinal

    def sendResult(mdrv,mdrv2,imgQuery,DB,nbVoisins,choixMethode,nbClusters=4):  #envoi des résultats au serveur. Fonction à appeler au serveur.
        Lresult=[]
        L=DB[:]
        Lsearch=MDRV.compareDescripteur(mdrv,mdrv2,imgQuery,DB,nbVoisins,choixMethode)
        for x in Lsearch:
            img=Image.open(L[x])    #conversion path-image
            arr=np.array(img)   #conversion image-array
            Lresult.append(arr) #envoi des requêtes sous format array, afin d'être traitées par le serveur
        return Lresult

if __name__ == "__main__":

    query = ##Your query image path##

class Database:

#Cette classe représente la base de données. Elle comprend les méthodes nécessaires à la gestion de la database.

    def __init__(self,listeImagesDB,descripteur): #constructeur
        self.listeImagesDB=listeImagesDB
        self.descripteur=descripteur

    def add(self,imageArray): #ajoute une image de la database
        image=Image.fromarray(imageArray)
        self.listeImagesDB.append(image)

    def setup():    #charge la database
        return self

    def listImagesDesc(self,listeImagesDB,desc): #crée une liste de vecteurs représentant chaque image de la database
        DBtech=[]
        for img in listeImagesDB:
            x=desc.createImageDesc(img)
            DBtech.append(x)
        return DBtech

if __name__ == "__main__":

    img1 = ##Path of DB images##


    DB=[img1] #List of DB image paths



class Descripteur:

#Cette classe est abstraite, et réprésente l'ensemble des descripteurs. Ce polymorphisme qui comprend toutes les moyens de décrire une image à partir d'un vecteur.

    def __init__(self,nomDescripteur):  #constructeur
        self.nomDescripteur=nomDescripteur

    def createImageDesc(imageDesc): #méthode abstraite
        pass



class DescripteurSIFT(Descripteur):

#Classe représentant le descripteur SIFT, classe fille de Descripteur.

    def __init__(self,nomDescripteur):  #constructeur
        Descripteur.__init__(self,nomDescripteur)

    def getKeypoints(self,imageDesc): #obtient les keypoints d'une image, selon le descripteur SIFT
        img = cv2.imread(imageDesc)
        data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,descriptor = sift.detectAndCompute(data,None)
        return kp

    def buildHistogram(descriptor_list, cluster_alg): #construit l'histogramme des fréquences des visuals words
        histogram = np.zeros(len(cluster_alg.cluster_centers_))
        cluster_result =  cluster_alg.predict(descriptor_list)
        for i in cluster_result:
            histogram[i] += 1.0
        return histogram

    def createImageVect(self,imageDesc): #crée une liste de vecteurs approximatifs de keypoints à partir de leurs attributs
        kp=self.getKeypoints(imageDesc)
        X=np.array([[kp[0].size,kp[0].angle,kp[0].pt[0],kp[0].pt[1]]])
        for k in range(1,len(kp)):
            x=np.array([[kp[k].size,kp[k].angle,kp[k].pt[0],kp[k].pt[1]]])
            X=np.concatenate((X,x),axis=0)
        return X

    def createImageDesc(self,imageDesc,nbClusters=4): #partitionne les keypoints et retourne l'histogramme d'une image
        X=self.createImageVect(imageDesc)
        kmeans = cluster.KMeans(n_clusters = nbClusters)
        kmeans.fit(X)
        if (X is not None):
            histogram = DescripteurSIFT.buildHistogram(X, kmeans)
            histogram=histogram/np.linalg.norm(histogram)
        return histogram

class DescripteurSURF(Descripteur):

    def __init__(self,nomDescripteur):  #constructeur
        Descripteur.__init__(self,nomDescripteur)

    def getKeypoints(self,imageDesc): #obtient les keypoints d'une image, selon le descripteur SIFT
        img = cv2.imread(imageDesc,0)
        surf = cv2.xfeatures2d.SURF_create()
        surf.setHessianThreshold(1000)
        kp, des = surf.detectAndCompute(img,None)
        return kp

    def buildHistogram(descriptor_list, cluster_alg): #construit l'histogramme des fréquences des visuals words
        histogram = np.zeros(len(cluster_alg.cluster_centers_))
        cluster_result =  cluster_alg.predict(descriptor_list)
        for i in cluster_result:
            histogram[i] += 1.0
        return histogram

    def createImageVect(self,imageDesc): #crée une liste de vecteurs approximatifs de keypoints à partir de leurs attributs
        kp=self.getKeypoints(imageDesc)
        X=np.array([[kp[0].size,kp[0].angle,kp[0].pt[0],kp[0].pt[1]]])
        for k in range(1,len(kp)):
            x=np.array([[kp[k].size,kp[k].angle,kp[k].pt[0],kp[k].pt[1]]])
            X=np.concatenate((X,x),axis=0)
        return X

    def createImageDesc(self,imageDesc,nbClusters=4): #partitionne les keypoints et retourne l'histogramme d'une image
        X=self.createImageVect(imageDesc)
        kmeans = cluster.KMeans(n_clusters = nbClusters)
        kmeans.fit(X)
        if (X is not None):
            histogram = DescripteurSIFT.buildHistogram(X, kmeans)
            histogram=histogram/np.linalg.norm(histogram)
        return histogram



#Classe Main :

if __name__ == "__main__":

    with open(query, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    a=B64_array(encoded_string)

    queryTest = Image.fromarray(a)
    queryTest.save("test.png")
    queryPath=##Query post-processing image path##

    Sift=DescripteurSIFT("SIFT")
    Surf=DescripteurSURF("SURF")
    dbSURF=Database(DB,Sift)
    dbSIFT=Database(DB,Surf)
    mdrv=MDRV()
    mdrv2=MDRV()
    mdrv.addDB(dbSIFT)
    mdrv2.addDB(dbSURF)
    Lresult=MDRV.sendResult(mdrv,mdrv2,queryPath,DB,5,2)