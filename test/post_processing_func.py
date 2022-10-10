# -*- coding: utf-8 -*-
#from mpl_toolkits.mplot3d import Axes3D   #modifier le 08-09-2021
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy.spatial as spatial
import pandas as pd
import numpy as np
import psutil
import scipy
import sys
import os

################################################ BINNING #########################################################
def binning_array(delta,binary_classes):
    x_min=min(binary_classes[:,0])
    x_max=max(binary_classes[:,0])
    y_min=min(binary_classes[:,1])
    y_max=max(binary_classes[:,1])
    z_min=min(binary_classes[:,2])
    z_max=max(binary_classes[:,2])
    nbr_bin_x=np.ceil((x_max-x_min)/delta)
    nbr_bin_y=np.ceil((y_max-y_min)/delta)
    bin_x = np.linspace(x_min, x_max, nbr_bin_x)
    bin_y = np.linspace(y_min, y_max, nbr_bin_y)
    pos_bin_x = np.digitize(binary_classes[:,0], bin_x)
    pos_bin_y = np.digitize(binary_classes[:,1], bin_y)
    return pos_bin_x,pos_bin_y

################################################ CLUSTERING DBSCAN #########################################################	
def cluster(pcd_list, method_str="dbscan", options={}):
    if method_str == "dbscan":
        method = DBSCAN
    else:
        sys.stderr.write("Error: Unsupported clustering method_str.\n")
        exit(0)
    # Build the estimator with the given options
    estimator = method(**options)
    # Fit the estimator
    os.system("clear")
    os.system("clear")
    print (pcd_list.nbytes)
    mem=psutil.virtual_memory()
    print (mem[1])
    estimator.fit(pcd_list)
    mem=psutil.virtual_memory()
    print (mem[1])
    exit()
    # Get the labels and return the labeled points
    labels = estimator.labels_
    clusters = np.append(pcd_list, np.array([labels]).T, 1)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return clusters, n_clusters_, labels

def run_dbscan(arr, method="dbscan",eps=0.21,bruit=False):
    if method == "dbscan":
        options = {'eps':eps,
               'min_samples':100,
            # 'algorithm':'kd_tree'
            }
    else:
        sys.stderr.write("Error: Unsupported clustering method.\n")
        exit(0)
    clusters, n_clusters , labels= cluster(arr[:,0:2], method, options)
    labl=clusters[:,2]
    clusterss=np.empty((len(clusters),4))
    clusterss[:,0]=clusters[:,0]
    clusterss[:,1]=clusters[:,1]
    clusterss[:,2]=arr[:,2]
    clusterss[:,3]=clusters[:,2]
    if bruit==False:
        return n_clusters, clusterss
    else:
        d=np.where(clusterss[:,3]!=-1)
        clusterss=clusterss[d]
        return n_clusters, clusterss
    
def run_dbscan_noz(arr, method="dbscan",eps=0.3):
    if method == "dbscan":
        options = {'eps':eps #,
               #'min_samples':100,
            # 'algorithm':'kd_tree'
            }
    else:
        sys.stderr.write("Error: Unsupported clustering method.\n")
        exit(0)
    #print(arr.shape)
    clusterss, n_clusters , labels= cluster(arr[:,0:2], method, options)
    labl=clusterss[:,2]
    d=np.where(clusterss[:,2]!=-1)
    clusterss=clusterss[d]
    clu=np.empty((len(clusterss),2))
    clu[:,0]=clusterss[:,0]
    clu[:,1]=clusterss[:,1]
    return clu

################################################ MASKING #########################################################
def masking(binary_classes,bins):
    do_ratio=False
    #prendre les indices de M et de nM
    #print ("Debut", len(binary_classes))
    ind_m=np.squeeze(np.array(np.where(binary_classes[:,4]==0)))
    ind_nm=np.squeeze(np.array(np.where(binary_classes[:,4]==1)))
    out=open("tempo.txt", "w")
    '''for l in np.squeeze(np.array(ind_nm)):
            out.write(str(binary_classes[l,0])+" "+str(binary_classes[l,1])+" "+str(binary_classes[l,2]+15)+"\n")
    for j in np.squeeze(np.array(ind_m)):
            out.write(str(binary_classes[j,0])+" "+str(binary_classes[j,1])+" "+str(binary_classes[j,2])+"\n")
    out.close()'''
    #os.system('wine lasview -i tempo.txt -iparse xyz')
    #Biner
    pos_bin_x,pos_bin_y=binning_array(bins,binary_classes)
    #Creation de tableau stockant len bin sur x,y et les classes et leurs idex respectifs
    l=pos_bin_y.size
    indx=np.arange(l)
    all=np.zeros((l,4))
    all[:,0]=pos_bin_x
    all[:,1]=pos_bin_y
    all[:,2]=binary_classes[:,4]
    all[:,3]=indx
    #prendre les bin x et y des M et nM 
    pos_bin_x_m=pos_bin_x[ind_m]
    pos_bin_x_nm=pos_bin_x[ind_nm]
    #print 'pos_bin_x_m', pos_bin_x_m.shape
    #print 'pos_bin_x_nm', pos_bin_x_nm.shape
    pos_bin_y_m=pos_bin_y[ind_m]
    pos_bin_y_nm=pos_bin_y[ind_nm]
    #print 'pos_bin_y_m', pos_bin_y_m.shape
    #print 'pos_bin_y_nm', pos_bin_y_nm.shape
    c=pos_bin_x_m+pos_bin_y_m*1j	#dictionnairen'ny bin'ny M rehetra
    d=pos_bin_x_nm+pos_bin_y_nm*1j #dictionnairen'ny bin'ny nM rehetra
    e=pos_bin_x+pos_bin_y*1j	##dictionnairen'ny bin'ny points rehetra
    #e jerena ao am c puis d
    diff_m=np.in1d(e,c)
    diff_nm=np.in1d(e,d)
    #Ceux qui sont reste en False sont les valeurs qui n apartiennent pas en meme temps a M et n; cad tsy meme bin
    del_m=np.array(np.where(diff_m==False))
    del_nm=np.array(np.where(diff_nm==False))
    #Ceux qui doivent rester des marquages
    filalmark=binary_classes[del_nm]
    filalmark=np.reshape(filalmark,(filalmark.shape[1],filalmark.shape[2]))
    print ("Marquage solo", len(filalmark))
    #Supprimer ces valeurs
    if len(filalmark)==0:
        do_ratio=True
    if do_ratio:
        del_m=np.reshape(del_m,(del_m.size))

        del_nm=np.reshape(del_nm,(del_nm.size))

        to_del=np.zeros((del_m.size+del_nm.size))
        to_del[0:del_m.size]=del_m
        to_del[del_m.size:]=del_nm

        to_del=np.unique(to_del)
        to_del=np.array(to_del, dtype=int)

        mask=np.ones_like(all,dtype=bool)
        mask[to_del]=False
        all_dd=all[mask,...]
        all_dd=np.reshape(all_dd,(int(len(all_dd)/4),4))


        #Separer les donnees restantes en M et nM	
        ind_m_=np.where(all_dd[:,2]==0)
        ind_nm_=np.where(all_dd[:,2]==1)

        all_dm=all_dd[ind_m_]
        all_dnm=all_dd[ind_nm_]

        #Faire le test d unicite et prendre leur nombre
        uniquem ,unique_inversem, unique_countsm =np.unique(all_dm[:,0:2], return_counts=True,return_inverse=True, axis=0)
        uniquenm, unique_inversenm, unique_countsnm =np.unique(all_dnm[:,0:2], return_counts=True,return_inverse=True, axis=0)

        unique_countsnm=np.array(unique_countsnm, dtype=float)
        unique_countsm=np.array(unique_countsm, dtype=float)

        #rapport
        ratio=np.array(unique_countsnm/unique_countsm, dtype=float)
        ind=np.where(ratio<0.15)

        #e jerena ao am d puis c
        indbool=np.in1d(unique_inversem,ind)
        indf=np.where(indbool==True)
        lmf=all_dm[indf,3]
        lmf=np.reshape(lmf,(lmf.size))

        lmf=np.array(lmf, dtype=int)
        final=binary_classes[lmf]
        '''out=open("test_binmasking4.txt", "w")
        for l in range(len(final)):
            out.write(str(final[l,0])+" "+str(final[l,1])+" "+str(final[l,2])+"\n")
        for l in range(len(filalmark)):
            out.write(str(filalmark[l,0])+" "+str(filalmark[l,1])+" "+str(filalmark[l,2]+10)+"\n")
        out.close()'''
        finalm=np.zeros((len(final)+len(filalmark),5))
        finalm[0:len(final),:]=final
        finalm[len(final):,:]=filalmark
        #print("Marquage miark", len(final))
        return finalm
    else:
        return filalmark
    
################################################ FILTRAGE #########################################################	
def m_filtrage(masked,eps):
    b=False
    method="dbscan"
    print ("debut clustering")
    #sys.exit("Error message")
    if len(masked)>1000000:
        print ("Trop de point")
        return None
    else:
        pass
    #mem=psutil.virtual_memory()
    #if mem[1]<4000000000
        
    n_clusters,clusters =run_dbscan(masked,method , eps , b)
    if n_clusters==0:
        return None
    else:
        pass
    print ("Fin clustering")
    classe_land_mark=np.array(clusters)
    sdt_m=[]
    for j in range(n_clusters):
        #print np.unique(clusters[:,3])
        ind=np.where(clusters[:,3]==j)
        if np.array(ind).size!=0:
            l_m_z=classe_land_mark[ind,2]
            std_u=np.std(l_m_z)
            sdt_m.append(std_u)
    min_std=np.min(sdt_m)
    max_std=np.max(sdt_m)
    mean_std=np.mean(sdt_m)
    median_std=np.median(sdt_m)
    ind_nm=np.where(sdt_m>mean_std)
    ind_nm=np.array(ind_nm)
    ind_nm=np.reshape(ind_nm, (ind_nm.shape[1]))
    ind_cl=[]

    for k in range(len(ind_nm)):
        ind_cl_=np.where(clusters[:,3]==ind_nm[k])
        ind_cl.append(ind_cl_)
    #print ind_cl
    if np.array(ind_cl).size==0:
        return classe_land_mark
    else:
        pass
    ind_cl=np.concatenate(ind_cl, axis=1)
    ind_cl=np.array(ind_cl)
    mask=np.ones_like(classe_land_mark, dtype=bool)
    mask[ind_cl]=False
    classe_land_mark=classe_land_mark[mask,...]
    classe_land_mark=np.reshape(classe_land_mark,(int(len(classe_land_mark)/clusters.shape[1]),clusters.shape[1]))
    return classe_land_mark
################################################ SUPPRESSION BRUIT #########################################################	
def del_noise(filtered,eps):
    method="dbscan"
    n_clusters, clusterss=run_dbscan( filtered, method,eps, True)
    return clusterss

################################################ ALPHA SHAPE, KEYPOINTS #########################################################	
#Si convexe: alpha 50, angle 180
#Si concave: alpha 0.05, angle 165, diff_angl 0.6, fuse: 0.05, #autre config:alpha 0.05, angle 165, diff_angl 0.2, fuse: 0.06
def get_distance(A,B):
    dist = np.linalg.norm(A[0:2]-B[0:2])    #t.r: computed in-line and within (X,Y) plane only
    #dist=np.sqrt(pow(A[:,0]-B[:,0],2)+pow(A[:,1]-B[:,1],2))
    return dist

#calcul de l'angle entre 3 points a,b,c de forme (x,y)
def calc_angle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    ret=np.degrees(angle)
    ret=np.around(ret,  decimals=1)
    #print(ret)
    return ret

#Obtenir le coordonnées d'un point
def get_coords(dist, A,B):
    C=np.zeros((2))
    theta=np.arctan2(B[1]-A[1],B[0]-A[0])
    C[0]=A[0]+dist*np.cos(theta)
    '''else:
        B[0]=A[0]-dist*np.cos(theta)'''
    C[1]=A[1]+dist*np.sin(theta)
    return C, theta	

#Obtenir le points le plus proche de l'axe des x
def get_minimum_point(A,B):	#,D
    if(B[0]>A[0]):
        return A,B
    else:
        return B,A
    
#Distance entre 1 points et les points d'un array
def get_distance_bis(A,B):
    #dist = np.linalg.norm(A[0:2]-B[:,0:2])    #t.r: computed in-line and within (X,Y) plane only
    dist=np.sqrt(pow(A[0]-B[:,0],2)+pow(A[1]-B[:,1],2))
    return dist	

#Distance entre 2 points
def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

#Fusion de points s'ils sont trop proches (> à une distance d), et les remplace par leur moyenne
def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1],points[i][2], points[i][3]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    point[2] += points[j][2]
                    point[3] += points[j][3]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            point[2] /= count
            point[3] /= count
            ret.append((point[0], point[1],point[2], point[3]))
    return ret

#Savoir si un segment est concave ou pas
#Methode: prendre 6 points sur le sgment, et autour d'un rayon de 5cm sans points sur l'un des 6 points choisis==concave
def get_concavity(A,B,in_):
    distance=get_distance(A,B)
    A,B=get_minimum_point(A,B)
    if distance>0.1:
        d_1=distance/7
        d_2=distance*2/7
        d_3=distance*3/7
        d_4=distance*4/7
        d_5=distance*5/7
        d_6=distance*6/7

        pts_1,tetha=get_coords(d_1,A,B)
        pts_2,tetha=get_coords(d_2,A,B)
        pts_3,tetha=get_coords(d_3,A,B)
        pts_4,tetha=get_coords(d_4,A,B)
        pts_5,tetha=get_coords(d_5,A,B)
        pts_6,tetha=get_coords(d_6,A,B)

        #print("keypoints_", len(keypoints_))
        pts_k_1=get_distance_bis(pts_1[0:2],in_)
        pts_k_2=get_distance_bis(pts_2[0:2],in_)
        pts_k_3=get_distance_bis(pts_3[0:2],in_)
        pts_k_4=get_distance_bis(pts_4[0:2],in_)
        pts_k_5=get_distance_bis(pts_5[0:2],in_)
        pts_k_6=get_distance_bis(pts_6[0:2],in_)
        seuil=0.05
        #print("pts_un_k_",pts_un_k_)
        #print("seuil",seuil)
        ind_1=np.array(np.where(pts_k_1<seuil))
        ind_2=np.array(np.where(pts_k_2<seuil))
        ind_3=np.array(np.where(pts_k_3<seuil))
        ind_4=np.array(np.where(pts_k_4<seuil))
        ind_5=np.array(np.where(pts_k_5<seuil))
        ind_6=np.array(np.where(pts_k_6<seuil))

        if ind_1.size==0 or ind_2.size==0 or ind_3.size==0 or ind_4.size==0 or ind_4.size==0 or ind_6.size==0:
            concave=True
        else:
            concave=False
        #print(concave)
        return concave

#Plote les keypoints obtenus
def plot_point(keypoints,in_):
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(in_[:,0], in_[:,1],s=1, c='b')
    n_contour=np.unique(keypoints[:,-1])
    #print(keypoints)
    for k in range(n_contour.size):
        ind=np.where(keypoints[:,-1]==k)
        if np.array(ind).size!=0:
            keypoints_sub=np.array(keypoints[ind])
            #print(keypoints_sub)
            for l in range(len(keypoints_sub)):
                if l==len(keypoints_sub)-1:
                    x0, y0=[keypoints_sub[l,0],keypoints_sub[0,0]], [keypoints_sub[l,1],keypoints_sub[0,1]]#, [edge_points[l,0,2],edge_points[l,1,2]]
                    #print("ty",x0,y0)
                    plt.plot(x0, y0, c='y')	#, marker='o'
                else:
                    x0, y0=[keypoints_sub[l,0],keypoints_sub[l+1,0]], [keypoints_sub[l,1],keypoints_sub[l+1,1]]#, [edge_points[l,0,2],edge_points[l,1,2]]
                    plt.plot(x0, y0, c='y')
    plt.show()

#Post du Post traitement:
#Si on a un angle de 180 °, le point est eliminé
#Si une difference d'angle entre deux angles consécutifs est inférieure à 0.6°, le point actuel est eliminé 
def post_post(test,conc):
    l_=len(test)
    keypoints=[]
    angl=[]
    keypoints.append(test[0])
    angl.append(0)
    if conc==True:
        angl_seuil=180 #Filtrerles points ayant u
    else:
        angl_seuil=180
    for i in range(l_-2):
        if i==l_-2:
            angle=calc_angle(test[i,0:2],test[i+1,0:2],test[0,0:2])
        else:
            angle=calc_angle(test[i,0:2],test[i+1,0:2],test[i+2,0:2])
        if angle<angl_seuil:
            if np.absolute((angle/10)-(angl[-1]/10))>=0.6:
                angl.append(int(angle))
                    #angl=np.array(angl)
                keypoints.append(test[i+1])
    keypoints=np.array(keypoints)
    return keypoints

#Post traitement du convexe hull(CH): voir si on a plusieurs objets en sortie du CH (objet séparé par '#')
#Si nbr objet>1: on creer un array x,y,label_objet(numero objet)
def post_traitement(command,conc):
    #Convex hull ou concave 
    os.system(command)	#0.05, 50 #marquage_5_1,116.565051 #170,0.1
    #Lecture du resultat de convex hull
    test=pd.read_csv("res.txt", header=None, delimiter=",")
    #print("\n\nBordure avant traitement", len(test))
    #Enlever la derniere ligne car c'est une caractère '#'
    #test=test.drop(test.index[len(test)-1])
    test=np.array(test)
    sepa=np.array(np.where(test[:,0]=='#'))
    sepa=np.reshape(sepa, sepa.shape[1])
    #print("sepa",sepa)
    all_keypoints=[]
    for i,sep in enumerate(sepa):
        #print(sep)
        if i==0:
            arr=np.array(test[0:sep], dtype=float)
        else:
            arr=np.array(test[sepa[i-1]+1:sep], dtype=float)
        keyp=post_post(arr,conc)
        label=np.ones_like(keyp[:,0])
        label=label*i
        label=np.reshape(label, (label.size))
        #print("keyp",keyp.shape)
        #print("label",label.shape)
        #keyp_f=np.concatenate((keyp,label),axis=1)
        keyp_f=np.zeros((keyp.shape[0],keyp.shape[1]+1))
        keyp_f[:,:-1]=keyp
        keyp_f[:,-1]=label
        all_keypoints.append(keyp_f)
    all_keypoints=np.concatenate(all_keypoints, axis=0)
    #print(all_keypoints)
    return all_keypoints

#Utilisation de get_concavity qui permet de determiner la concavite de l'objet: True si concave, False sinon
def test_concavite(keypoints):
    #Test de concavité
    concavite=np.empty((len(keypoints)), dtype=bool)
    for i in range(len(keypoints)):
        if i==len(keypoints)-1:
            A=keypoints[i]
            B=keypoints[0]
            concavit=get_concavity(A,B,in_)
            concavite[i]=(concavit)
        else:
            A=keypoints[i]
            B=keypoints[i+1]
            concavit=get_concavity(A,B,in_)
            concavite[i]=(concavit)
    ind_conc=np.array(np.where(concavite==True))
    if ind_conc.size==0:
        return False
        #return keypoints
    else:
        return True	

#Appel de tout les traitements pour obtenir les keypoints
def run_keypoints(array):
    #Ecrire d'abord l'array sur disque car lasboundary travail sur donnée sur disque et ecrit les résultats sur disque également
    to_read="to_keypoints.txt"
    out=open(to_read, "w")
    for l in range(len(array)):
        if array.shape[1]==3:
            out.write(str(array[l,0])+" "+str(array[l,1])+" "+str(array[l,2])+"\n")
        elif array.shape[1]==2:
            out.write(str(array[l,0])+" "+str(array[l,1])+" "+str(0)+"\n")
    out.close()
    '''if os.name == 'nt':
        os.system('lasview -i to_keypoints.txt -iparse xyz')
        #os.system('lasnoise -i to_keypoints.txt remove_noise o lidar_without_noise.las')
        #os.system('lasview -i to_keypoints_1.las -iparse xyz')
    else:
        os.system('wine lasview -i to_keypoints.txt -iparse xyz')
        #os.system('wine lasnoise -i to_keypoints.txt remove_noise o lidar_without_noise.las')
        #os.system('wine lasview -i to_keypoints_1.las -iparse xyz')'''

    #Commande pour convexe(100) et concave hull(1, 0.5)
    '''alpha_dep=50
    alpha_dep_=0.5
    alpha_dep_next=5'''
    alpha_dep=100
    alpha_dep_=10
    alpha_dep_next=1
    #Si on a un systeme windows
    if os.name == 'nt':
        command_convex='lasboundary -i ' +str(to_read)+' -o res.txt -concavity ' +str(alpha_dep)
        command_concave='lasboundary -i ' +str(to_read)+' -o res.txt -concavity ' +str(alpha_dep_)	#0.05
        command_concave_next='lasboundary -i ' +str(to_read)+' -o res.txt -concavity ' +str(alpha_dep_next)	#+' -disjoint -holes'
    #Pour linux
    else:
        command_convex='wine lasboundary -i ' +str(to_read)+' -o res.txt -concavity ' +str(alpha_dep)
        command_concave='wine lasboundary -i ' +str(to_read)+' -o res.txt -concavity ' +str(alpha_dep_)	#0.05
        command_concave_next='wine lasboundary -i ' +str(to_read)+' -o res.txt -concavity ' +str(alpha_dep_next)	#+' -disjoint -holes'

    #Lecture du marquage(un objet)
    in_=np.array(array)
    #print("\nBordure avant traitement", len(array))
    #Creer un array pour tous les keypoints obtenus
    keypoints=post_traitement(command_convex,False)
    
#Test de concavité sur tous les cotés du keypoints et retourne le nombre de segment etant concave
    def test_con(keypoints,nubr):
        if nubr==0:
            concavite=np.empty((len(keypoints)), dtype=bool)
            for i in range(len(keypoints)):
                if i==len(keypoints)-1:
                    A=keypoints[i]
                    B=keypoints[0]
                    concavit=get_concavity(A,B,in_)
                    concavite[i]=(concavit)
                else:
                    A=keypoints[i]
                    B=keypoints[i+1]
                    concavit=get_concavity(A,B,in_)
                    concavite[i]=(concavit)
            ind_conc=np.array(np.where(concavite==True)) 
            return ind_conc
        elif nubr==1:
            n_contour=np.unique(keypoints[:,-1])
            ind_conc_final=[]
            for k in range(n_contour.size):
                ind=np.where(keypoints[:,-1]==k)
                if np.array(ind).size!=0:
                    keypoints_sub=np.array(keypoints[ind])
                    #print(keypoints_sub)
                    concavite=np.empty((len(keypoints_sub)), dtype=bool)
                    for i in range(len(keypoints_sub)):
                        if i==len(keypoints_sub)-1:
                            A=keypoints_sub[i]
                            B=keypoints_sub[0]
                            concavit=get_concavity(A,B,in_)
                            concavite[i]=(concavit)
                        else:
                            A=keypoints_sub[i]
                            B=keypoints_sub[i+1]
                            concavit=get_concavity(A,B,in_)
                            concavite[i]=(concavit)
                    #print(concavite)
                    ind_conc=np.array(np.where(concavite==True))
                    if ind_conc.size!=0:
                        ind_conc=np.reshape(ind_conc, (ind_conc.size))
                        ind_conc_final.append(ind_conc)
                        ind_conc_final=np.concatenate(ind_conc_final)
                        return ind_conc_final
                    else:
                        a=0
            return np.array(ind_conc_final)


    ind_conc=test_con(keypoints,0)
    #Si aucun segment du keypoints n'est concave, on a une forme convexe et le keypoints est retourné
    if ind_conc.size==0:
        #print("\nBordure apres traitement", len(keypoints),"avec concavite", alpha_dep)
        #plot_point(keypoints,in_)
        return keypoints
    #Sinon, on utilise une valeur de alpha plus concave que précédemment
    else:
        keypoints=post_traitement(command_concave_next,True)
        ind_conc=np.array(test_con(keypoints,1))
        #Si le résultats ne présente plus aucune concavité sur l'objet, on a un bon contour pour l'objet, le keypoints est retourné
        if ind_conc.size==0:
            #print("\nBordure apres traitement", len(keypoints),"avec concavite", alpha_dep_next)
            keypoints=np.array(fuse(keypoints, 0.05))
            return keypoints
        #Sinon, on diminue encore alpha et retourne le résultats obtenus
        else:
            keypoints=post_traitement(command_concave,True)
            keypoints=np.array(fuse(keypoints, 0.05))
            #print("\nBordure apres traitement", len(keypoints),"avec concavite", alpha_dep_)
            #plot_point(keypoints,in_)
            return  keypoints
    
################################################ RECTANGULOIDE #########################################################
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)
    return rval

#Calcul l'aire d'un polygone irregulier
def find_area(polygone):
    siz=len(polygone)
    Area=0
    for i in range(siz):
        if i==siz-1:
            side_g=polygone[i]
            side_d=polygone[0]  
        else:
            side_g=polygone[i]
            side_d=polygone[i+1]  
        x_g_h=side_g[0]
        y_g_h=side_g[1]
        x_g_b=side_g[0]
        y_g_b=0

        x_d_h=side_d[0]
        y_d_h=side_d[1]
        x_d_b=side_d[0]
        y_d_b=0
        larg=x_d_b-x_g_b
        Long=(y_g_h+y_d_h)/2
        Ll=larg*Long
        Area+=Ll
    return(Area)

#Ploter les rectangles 
def draw_line(X,bound,P_area,area, A):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(A[:,0],A[:,1],c='g',s=0.5)
    if bound is not None:
        for k in range(len(bound)):
            if k==len(bound)-1:
                x0, y0=[bound[k,0],bound[0,0]], [bound[k,1],bound[0,1]]
                plt.plot(x0, y0, c='r')
            else:
                x0, y0=[bound[k,0],bound[k+1,0]], [bound[k,1],bound[k+1,1]]
                plt.plot(x0, y0, c='r')

    if X is not None:
        for k in range(len(X)):
            if k==len(X)-1:
                x0, y0=[X[k,0],X[0,0]], [X[k,1],X[0,1]]
                plt.plot(x0, y0, c='b')
            else:
                x0, y0=[X[k,0],X[k+1,0]], [X[k,1],X[k+1,1]]
                plt.plot(x0, y0, c='b')
    if P_area is not None and area is not None:
        plt.title('Poly : '+ str(P_area)+ '   Rect : ' +str(area)+'   dif :'+ str(abs(abs(P_area)-abs(area))))
    #mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    #plt.show()
    
    
#Calcul l'aire d'un rectangle
def area_rect(bound):
    bound1=np.ones_like(bound)
    bound1[:,0]=np.roll(bound[:,0], 1)
    bound1[:,1]=np.roll(bound[:,1], 1)
    d=dist = np.sqrt( (bound[:,0] - bound1[:,0])**2 + (bound[:,1] - bound1[:,1])**2 )
    area=d[0]*d[1]
    return area

#TCalcul de l'angle sur les 4 cotes du rectangle
def calc_angle_rect(bound):
    totale=[]
    for i in range(len(bound)):
        if i==0:
            angle=calc_angle(bound[0],bound[1],bound[2])
        elif i==1:
            angle=calc_angle(bound[1],bound[2],bound[3])
        elif i==2:
            angle=calc_angle(bound[2],bound[3],bound[0])
        elif i==3:
            angle=calc_angle(bound[3],bound[0],bound[1])
        totale.append(angle)
    return np.array(totale)

def rectanguloide(all,num_):
    All_final_keypoints=np.empty((0,0,0,0), dtype=object)
    #X_all=pd.read_csv("keypointsVR.txt", header=None, delimiter=' ').values  #TOPCA
    #all=pd.read_csv("TOPCA.txt", header=None, delimiter=' ').values
    keys=np.unique(all[:,3])
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    Totaleangle=np.empty((0))
    for key in keys:
        #print(key)
        #ind_k=np.where(X_all[:,2]==key)
        ind_a=np.where(all[:,3]==int(key))
        z_max=np.max(all[ind_a,2])
        #X=X_all[ind_k,0:2]
        A=all[ind_a,0:2]
        #print(A.shape)
        #X=np.squeeze(X)
        A=np.squeeze(A)
        #print(A.shape)
        #ax.scatter(A[:,0],A[:,1],c='g',s=0.5)
        B=run_dbscan_noz(A)
        if B.size==0:
            print ('Pas de marquage pour l objet ', key, 'sur ', len(keys), 'objet(s)')
            continue
        else:
            pass
        X=run_keypoints(B)
        if B.size!=0:
            #print("X.size",X.size)
            bound=minimum_bounding_rectangle(B)
            Anglee=calc_angle_rect(bound)
            Totaleangle=np.concatenate((Totaleangle,Anglee))
            #Totaleangle=np.concatenate(Totaleangle)
        else:
            continue

        area=area_rect(bound)
        P_area=find_area(X)
        #draw_line(None,bound,P_area,area) 
            #Paraetre convenu le 04/01/2018 d'apres resultat visuel projet3
        ratio=np.round(abs(abs(P_area)-abs(area)), decimals=3)
        #draw_line(None,bound,None,None,A)
        #TEST ANGLE SI 90°
        #print("ANGLE",np.unique(Totaleangle))
        #print("ANGLE",(Totaleangle))
        #print("ANGLE",len(Totaleangle))
        #print("Key",len(keys))
        #ax.text(bound[2,0],bound[2,1],str(ratio),fontsize=8)
        if ratio<2.5:
            Temp=All_final_keypoints
            if All_final_keypoints.size==0:
                All_final_keypoints=np.empty((4,4),dtype=object)
                All_final_keypoints[:,0:2]=bound
                All_final_keypoints[:,2]=z_max*np.ones_like((bound[:,0]))
                numObj=str(num_)+'_'+str(int(key))
                print(type(numObj))
                All_final_keypoints[len(Temp):,3]=np.full(bound[:,0].shape,numObj)
            else:
                All_final_keypoints=np.empty(((len(Temp)+4),4),dtype=object)
                All_final_keypoints[0:len(Temp),:]=Temp
                All_final_keypoints[len(Temp):,0:2]=bound
                All_final_keypoints[len(Temp):,2]=z_max*np.ones_like((bound[:,0]))
                numObj=str(num_)+'_'+str(int(key))
                All_final_keypoints[len(Temp):,3]=np.full(bound[:,0].shape,numObj,dtype=object)
        else:
            Temp=All_final_keypoints
            if All_final_keypoints.size==0:
                All_final_keypoints=np.empty((len(X),4), dtype=object)
                All_final_keypoints[:,0:2]=X[:,0:2]
                All_final_keypoints[:,2]=z_max*np.ones_like((X[:,0]))
                numObj=str(num_)+'_'+str(int(key))
                All_final_keypoints[:,3]=np.full(X[:,0].shape,numObj,dtype=object)
            else:
                All_final_keypoints=np.empty(((len(Temp)+len(X)),4),dtype=object)
                All_final_keypoints[0:len(Temp),:]=Temp
                All_final_keypoints[len(Temp):,0:2]=X[:,0:2]
                All_final_keypoints[len(Temp):,2]=z_max*np.ones_like((X[:,0]))
                numObj=str(num_)+'_'+str(int(key))
                All_final_keypoints[len(Temp):,3]=np.full(X[:,0].shape,numObj,dtype=object)
            #draw_line(None,bound,None,None)
            #ax.text(bound[0,0],bound[0,1],str(ratio),fontsize=6)
    #plt.show()
    return All_final_keypoints

def draw_final_result(All_final_keypoints,object_alls):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') #, projection='3d'
    ax.scatter(object_alls[:,0],object_alls[:,1],c='g', s=0.5)
    ax.scatter(binary_classes[:,0],binary_classes[:,1],c='g', s=0.5)
    objs=np.unique(All_final_keypoints[:,3])
    for obj in objs:
        ind=np.where(All_final_keypoints[:,3]==obj)
        c_obj=All_final_keypoints[ind,0:2]
        c_obj=np.squeeze(c_obj)
        for l in range(len(c_obj)):
            if l==len(c_obj)-1:
                x0, y0=[c_obj[l,0],c_obj[0,0]], [c_obj[l,1],c_obj[0,1]]#, 
                plt.plot(x0, y0, c='r')	#, marker='o'
            else:
                x0, y0=[c_obj[l,0],c_obj[l+1,0]], [c_obj[l,1],c_obj[l+1,1]]#, [edge_points[l,0,2],edge_points[l,1,2]]
                plt.plot(x0, y0, c='r')
    #mng = plt.get_current_fig_manager()
    #mng.window.state('normal')
    plt.show()
        
    
