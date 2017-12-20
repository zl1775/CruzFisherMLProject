import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys



pi = np.pi
sigmas = [1,3,6]
thetas = [0, pi/6, pi/4, pi/3, pi/2, 2*pi/3, 3*pi/4, 5*pi/6]
#sigmas = [3]
#thetas = [0]
lambdas = []
size = 21
offset = 10

left = sys.argv[1]
right = sys.argv[2]

DIR = sys.argv[3].rstrip('/') + '/'


#Compute constant  C2 for 2d morlet wavelet. This constant is set to make the sum (integral) of the wavelet equal to 0.
def calcC2(sigma, theta):
    num = 0.0
    denom = 0.0
    for i in range(0,size):
        for j in range(0,size):
            u = np.array([j-offset,i-offset]) # displacement from center
            etheta = np.array([np.cos(theta), np.sin(theta)]) #direction vector
            num += np.exp(1j * (pi / (2*sigma)) * np.dot(u,etheta)) * np.exp(-1 * np.dot(u,u) / (2 * sigma*sigma))
            denom += np.exp(-1 * np.dot(u,u) / (2 * sigma * sigma))
    return num/denom

#Calculate the constant C1 for 2d morlet wavelet. This is set to normalize the sum of the square of the wavelet to 1,
def calcC1(sigma, theta, c2):
    z = 0.0
    for i in range(0,size):
        for j in range(0,size):
            u = np.array([j-offset,i-offset]) # displacement from center
            etheta = np.array([np.cos(theta), np.sin(theta)]) #direction vector
            #We only care about the real part here (hence the cos) since the imaginary part will cancel.
            #The below expression is the real part of the square of the wavelet
            z += ((1 - 2 * c2 * np.cos((pi/(2*sigma)) * np.dot(u,etheta))) + c2*c2) * np.exp(-1 * np.dot(u,u) / (sigma * sigma))
    return 1 / np.sqrt(z)


#Gaussian blur
def calcC(sig):
    z = 0.0
    for i in range(0,size):
        for j in range(0,size):
            u = np.array([j-offset,i-offset])
            z += np.exp(-2*(np.dot(u,u))/(2*sig*sig))
    return 1 / np.sqrt(z)

def imgLookup(img, pixels, x, y):
    if (x < 0) or (x >= img.width) or (y < 0) or (y >= img.height):
        return 0
    elif isinstance(pixels[x,y], tuple):
        return pixels[x,y][0]
    else:
        return pixels[x,y]


rl = 0
counter = 1
Cs = np.zeros((len(sigmas), len(thetas), 2), dtype=complex)
for i, sigma in enumerate(sigmas):
    for j, theta in enumerate(thetas):

        C2 = calcC2(sigma, theta)
        C1 = calcC1(sigma, theta, C2)
        Cs[i, j, 0] = C1
        Cs[i, j, 1] = C2

for si, sigma in enumerate(sigmas):
    for tj, theta in enumerate(thetas):

        C2 = Cs[si, tj, 1]
        C1 = Cs[si, tj, 0]
        psi = np.zeros((size,size), dtype=complex)
        print(counter)
        etheta = np.array([np.cos(theta), np.sin(theta)])
        for i in range(0,size):
            for j in range(0,size):
                u = np.array([j-offset,i-offset])
                psi[i,j] = (C1/sigma)*(np.exp(1j * (pi/(2*sigma)) * np.dot(u,etheta)) - C2) * np.exp(-1 * np.dot(u,u) / (2*sigma*sigma))
        rl = np.real(psi)
        plt.imshow(rl,'gray')
        plt.title("Real: sigma = " + str(sigma) + ", theta = " + str(theta))
        plt.savefig(DIR + "waveletR" + str(si) + "_" + str(tj) + ".png")

        im = np.imag(psi)
        plt.imshow(im,'gray')
        plt.title("Imaginary: sigma = " + str(sigma) + ", theta = " + str(theta))
        plt.savefig(DIR +"waveletI" + str(si) + "_" + str(tj) + ".png")
        lambdas.append(psi)
        counter += 1

with open("lambdas.pickle", 'wb') as f:
    pickle.dump(lambdas, f, pickle.HIGHEST_PROTOCOL)

img = Image.open(left)
(col, row) = img.size
pixels = img.load()
grayPixL = np.zeros((row,col))
redPixL = np.zeros((row,col))
greenPixL = np.zeros((row,col))
bluePixL = np.zeros((row,col))
for c in range(col):
    for r in range(row):
        grayPixL[r,c] = 0.299 * pixels[c,r][0] + 0.587 * pixels[c,r][1] + 0.114*pixels[c,r][2]
        redPixL[r,c] = pixels[c,r][0]
        greenPixL[r,c] = pixels[c,r][1] 
        bluePixL[r,c] = pixels[c,r][2]

img = Image.open(right)
(col, row) = img.size
pixels = img.load()
grayPixR = np.zeros((row,col))
redPixR = np.zeros((row,col))
greenPixR = np.zeros((row,col))
bluePixR = np.zeros((row,col))
for c in range(col):
    for r in range(row):
        grayPixR[r,c] = 0.299 * pixels[c,r][0] + 0.587 * pixels[c,r][1] + 0.114*pixels[c,r][2]
        redPixR[r,c] = pixels[c,r][0]
        greenPixR[r,c] = pixels[c,r][1] 
        bluePixR[r,c] = pixels[c,r][2]

counter = 0
#mms = MinMaxScaler(feature_range=(0.0, 256.0), copy=False)
for si, sigma in enumerate(sigmas):
    for tj, theta in enumerate(thetas):
        print(counter)
        newPixels = np.zeros((row,col), dtype=complex)
        newPixelsR = np.zeros((row,col), dtype=complex)
        newPixelsG = np.zeros((row,col), dtype=complex)
        newPixelsB = np.zeros((row,col), dtype=complex)


        newPixels = signal.convolve2d(grayPixL, lambdas[counter], mode='same', boundary='fill')
        newPixelsR = signal.convolve2d(redPixL, lambdas[counter], mode='same', boundary='fill')
        newPixelsG = signal.convolve2d(greenPixL, lambdas[counter], mode='same', boundary='fill')
        newPixelsB = signal.convolve2d(bluePixL, lambdas[counter], mode='same', boundary='fill')

        #Left - gray
        real_part = (np.real(newPixels))
        img_part = (np.imag(newPixels))
        r_max = np.max(real_part)
        i_max = np.max(img_part)
        r_min = np.min(real_part)
        i_min = np.min(img_part)

        real_part -= r_min
        img_part -= i_min

        real_part *= 256.0 / (r_max - r_min)
        img_part *= 256.0 / (i_max - i_min)



        arrReal = (real_part).astype('uint8')
        newImg = Image.fromarray(arrReal)
        newImg.save(DIR + "convolveGray_Left_Real_" + str(si) + "_" + str(tj) + ".png")

        arrImag = (img_part).astype('uint8')
        newImg = Image.fromarray(arrImag)
        newImg.save(DIR + "convolveGray_Left_Imag_" + str(si) + "_" + str(tj) + ".png")

        #Left - red
        real_part = (np.real(newPixelsR))
        img_part = (np.imag(newPixelsR))
        r_max = np.max(real_part)
        i_max = np.max(img_part)
        r_min = np.min(real_part)
        i_min = np.min(img_part)

        real_part -= r_min
        img_part -= i_min

        real_part *= 256.0 / (r_max - r_min)
        img_part *= 256.0 / (i_max - i_min)


        arrReal = (real_part).astype('uint8')
        newImg = Image.fromarray(arrReal)
        newImg.save(DIR +"convolveRed_Left_Real_" + str(si) + "_" + str(tj) + ".png")

        arrImag = (img_part).astype('uint8')
        newImg = Image.fromarray(arrImag)
        newImg.save(DIR +"convolveRed_Left_Imag_" + str(si) + "_" + str(tj) + ".png")

        #Left - green
        real_part = (np.real(newPixelsG))
        img_part = (np.imag(newPixelsG))
        r_max = np.max(real_part)
        i_max = np.max(img_part)
        r_min = np.min(real_part)
        i_min = np.min(img_part)

        real_part -= r_min
        img_part -= i_min

        real_part *= 256.0 / (r_max - r_min)
        img_part *= 256.0 / (i_max - i_min)


        arrReal = (real_part).astype('uint8')
        newImg = Image.fromarray(arrReal)
        newImg.save(DIR + "convolveGreen_Left_Real_" + str(si) + "_" + str(tj) + ".png")

        arrImag = (img_part).astype('uint8')
        newImg = Image.fromarray(arrImag)
        newImg.save(DIR + "convolveGreen_Left_Imag_" + str(si) + "_" + str(tj) + ".png")

        #Left - blue
        real_part = (np.real(newPixelsB))
        img_part = (np.imag(newPixelsB))
        r_max = np.max(real_part)
        i_max = np.max(img_part)
        r_min = np.min(real_part)
        i_min = np.min(img_part)

        real_part -= r_min
        img_part -= i_min

        real_part *= 256.0 / (r_max - r_min)
        img_part *= 256.0 / (i_max - i_min)


        arrReal = (real_part).astype('uint8')
        newImg = Image.fromarray(arrReal)
        newImg.save(DIR + "convolveBlue_Left_Real_" + str(si) + "_" + str(tj) + ".png")

        arrImag = (img_part).astype('uint8')
        newImg = Image.fromarray(arrImag)
        newImg.save( DIR + "convolveBlue_Left_Imag_" + str(si) + "_" + str(tj) + ".png")






        newPixels = np.zeros((row,col), dtype=complex)
        newPixelsR = np.zeros((row,col), dtype=complex)
        newPixelsG = np.zeros((row,col), dtype=complex)
        newPixelsB = np.zeros((row,col), dtype=complex)


        newPixels = signal.convolve2d(grayPixR, lambdas[counter], mode='same', boundary='fill')
        newPixelsR = signal.convolve2d(redPixR, lambdas[counter], mode='same', boundary='fill')
        newPixelsG = signal.convolve2d(greenPixR, lambdas[counter], mode='same', boundary='fill')
        newPixelsB = signal.convolve2d(bluePixR, lambdas[counter], mode='same', boundary='fill')


        #Right - gray
        real_part = (np.real(newPixels))
        img_part = (np.imag(newPixels))
        r_max = np.max(real_part)
        i_max = np.max(img_part)
        r_min = np.min(real_part)
        i_min = np.min(img_part)

        real_part -= r_min
        img_part -= i_min

        real_part *= 256.0 / (r_max - r_min)
        img_part *= 256.0 / (i_max - i_min)


    
        arrReal = (real_part).astype('uint8')
        newImg = Image.fromarray(arrReal)
        newImg.save(DIR + "convolveGray_Right_Real_" + str(si) + "_" + str(tj) + ".png")

        arrImag = (img_part).astype('uint8')
        newImg = Image.fromarray(arrImag)
        newImg.save(DIR + "convolveGray_Right_Imag_" + str(si) + "_" + str(tj) + ".png")


        #Right - red
        real_part = (np.real(newPixelsR))
        img_part = (np.imag(newPixelsR))
        r_max = np.max(real_part)
        i_max = np.max(img_part)
        r_min = np.min(real_part)
        i_min = np.min(img_part)

        real_part -= r_min
        img_part -= i_min

        real_part *= 256.0 / (r_max - r_min)
        img_part *= 256.0 / (i_max - i_min)


    
        arrReal = (real_part).astype('uint8')
        newImg = Image.fromarray(arrReal)
        newImg.save(DIR + "convolveRed_Right_Real_" + str(si) + "_" + str(tj) + ".png")

        arrImag = (img_part).astype('uint8')
        newImg = Image.fromarray(arrImag)
        newImg.save(DIR + "convolveRed_Right_Imag_" + str(si) + "_" + str(tj) + ".png")


        #Right - green
        real_part = (np.real(newPixelsG))
        img_part = (np.imag(newPixelsG))
        r_max = np.max(real_part)
        i_max = np.max(img_part)
        r_min = np.min(real_part)
        i_min = np.min(img_part)

        real_part -= r_min
        img_part -= i_min

        real_part *= 256.0 / (r_max - r_min)
        img_part *= 256.0 / (i_max - i_min)


    
        arrReal = (real_part).astype('uint8')
        newImg = Image.fromarray(arrReal)
        newImg.save(DIR + "convolveGreen_Right_Real_" + str(si) + "_" + str(tj) + ".png")

        arrImag = (img_part).astype('uint8')
        newImg = Image.fromarray(arrImag)
        newImg.save(DIR + "convolveGreen_Right_Imag_" + str(si) + "_" + str(tj) + ".png")


        #Right - blue
        real_part = (np.real(newPixelsB))
        img_part = (np.imag(newPixelsB))
        r_max = np.max(real_part)
        i_max = np.max(img_part)
        r_min = np.min(real_part)
        i_min = np.min(img_part)

        real_part -= r_min
        img_part -= i_min

        real_part *= 256.0 / (r_max - r_min)
        img_part *= 256.0 / (i_max - i_min)

    
        arrReal = (real_part).astype('uint8')
        newImg = Image.fromarray(arrReal)
        newImg.save(DIR + "convolveBlue_Right_Real_" + str(si) + "_" + str(tj) + ".png")

        arrImag = (img_part).astype('uint8')
        newImg = Image.fromarray(arrImag)
        newImg.save(DIR + "convolveBlue_Right_Imag_" + str(si) + "_" + str(tj) + ".png")


        counter += 1






