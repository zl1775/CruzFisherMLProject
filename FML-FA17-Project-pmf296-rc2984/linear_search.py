import numpy as np
import random
from conv_loader2 import conv_loader as cl
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import load_pfm
from PIL import Image
import math

loader = cl("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "wavelet_img/")
loader.load()

training_set = loader.get_training_set_classification(12, 500, 2100, 1000, 2700)

#set the window to predict on.
rstart = 980 #rowstart
cstart = 2650 #column start
rend = 1030 #row end
cend = 2700 #column end
loader.get_test_set_linsearch(12,rstart, cstart, rend, cend)

random.shuffle(training_set)

#Preprocess trainging data
training_labels = np.asarray([tupl[0] for tupl in training_set])
training_samples = np.asarray([tupl[1] for tupl in training_set])
normalize(training_samples, norm='l2', axis=1, copy=False)


#Train classifier
sgd_clf = SGDClassifier()
sgd_clf.fit(training_samples, training_labels)



img3_arr = np.zeros((rend-rstart,cend-cstart))
def getCandidates(r, c):
        candidates = []
        lef = loader.lefts[r-rstart][c-cstart]
        (res, disp) = load_pfm.lookupL(r, c, loader.arr0, loader.arr1)
        if res == 2:
            img3_arr[r-rstart,c-cstart] = disp
        for dis in range(0,300):
            righ = loader.rights[r-rstart][(cend - 1 - c) + dis]
            feature_vector = np.abs(lef - righ)
                #print(disp)
            candidates.append((c, r, dis, feature_vector))

        return candidates

#Calculate the best disparity for each pixel in the test window
best_disps = np.zeros((rend-rstart,cend-cstart))
for r in range(rstart, rend):
    print((r-rstart)/(rend-rstart))
    for c in range(cstart, cend):
        candidates = getCandidates(r,c)
        linsearch_coords = [(tupl[0], tupl[1], tupl[2]) for tupl in candidates]
        linsearch_samples = np.asarray([tupl[3] for tupl in candidates])
        normalize(linsearch_samples, norm='l2', axis=1, copy=False)
        best_disp = -1
        best_conf = float("-inf")
        for i, samp in enumerate(linsearch_samples):
            conf = sgd_clf.decision_function(samp.reshape(1,-1))
            if conf > best_conf:
                best_disp = i
                best_conf = conf
        best_disps[r-rstart,c-cstart] = best_disp
        #print(best_disp)
            

img_arr = np.zeros((rend-rstart,cend-cstart))
img2_arr = np.zeros((rend-rstart,cend-cstart))
img4_arr = np.zeros((rend-rstart,cend-cstart))

for r in range(rend-rstart):
    for c in range(cend-cstart):
        img_arr[r,c] = best_disps[r,c]
        pixels = loader.img_lookup(loader.Left_Pixels, c + cstart, r+rstart)
        img2_arr[r,c] = 0.299 * pixels[0] + 0.587 * pixels[1] + 0.114*pixels[2]
        pixels = loader.img_lookup(loader.Right_Pixels, c + cstart - best_disps[r,c], r + rstart)
        img4_arr[r,c] = 0.299 * pixels[0] + 0.587 * pixels[1] + 0.114*pixels[2]



#calculate stats and create output images
total = 0
total_within1 = 0
total_within2 = 0
total_within5 = 0
meansquare = 0.0
for r in range(rend-rstart):
    for c in range(cend-cstart):
        (res, disp) = load_pfm.lookupL(r + rstart, c + cstart, loader.arr0, loader.arr1)
        if res == 2:
            err = abs(disp - img_arr[r,c])
            if err <= 1:
                total_within1 += 1
            if err <= 2:
                total_within2 += 1
            if err <= 5:
                total_within5 += 1
            meansquare += err **2
            total += 1
print("within 1:", total_within1 / total)
print("within 2:", total_within2 / total)
print("within 5:", total_within5 / total)
print("mean square error:", math.sqrt(meansquare / total))


arrImag = (img_arr * 256 / 300).astype('uint8')
newImg = Image.fromarray(arrImag)
newImg.save("outtest.png")    
arrImag = (img2_arr).astype('uint8')
newImg = Image.fromarray(arrImag)
newImg.save("imggray.png")    
arrImag = (img3_arr * 256 / 300).astype('uint8')
newImg = Image.fromarray(arrImag)
newImg.save("dispgray.png")   
arrImag = (img4_arr).astype('uint8')
newImg = Image.fromarray(arrImag)
newImg.save("Right_pred.png")  

