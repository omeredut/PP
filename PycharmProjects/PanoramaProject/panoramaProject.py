import numpy as np
import cv2 as cv
import random
import os


#1
#-----------Image pair registration-----------#
#This function get image path anr return img mat 2D in grayscale format
def loadImageInGrayscale(imageName):
    return cv.imread(imageName, cv.IMREAD_GRAYSCALE)



#2
#-----------Feature point detection, descriptor extraction and matching-----------#
#This fuunction get 2 images in grayscale format and return 2 sets of coordinate matches in the 2 images - pos1, pos2
def getPositions(img1, img2):

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    pos1 = []
    pos2 = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            x1, y1 = kp1[img1_idx].pt
            x2, y2 = kp2[img2_idx].pt
            p1 = [int(x1), int(y1)]
            p2 = [int(x2), int(y2)]
            # Append to each list
            pos1.append(p1)
            pos2.append(p2)
    return np.array(pos1), np.array(pos2)




#3
#-----------Registering the transformation-----------#
#This function get matching positions in image 1 and homography and return the matches positions in image 2
def applyHomography(pos1, H12):
    n = len(pos1)
    ones = np.ones((n, 1),dtype=int)
    p = np.concatenate((pos1, ones), axis=1)
    pos2 = []
    for v in p:
        t = np.dot(H12, v)
        pos2.append([np.divide(t[0],t[2]), np.divide(t[1],t[2])])
    return np.array(pos2)





#4
#This function get matching positions in image 1 and image 2 and compute the homography by DLT algorithm
def leastSquaresHomography(p1,p2):
    n = len(p1)
    A = np.zeros((2*n,9))
    for i in range(n):
        x = p1[i][0]
        y = p1[i][1]
        x_t = p2[i][0]
        y_t = p2[i][1]
        A[2*i-1, :] = [-x, -y, -1, 0, 0, 0, x * x_t, x_t * y, x_t]
        A[2 * i, :] = [0, 0, 0, -x, -y, -1, x * y_t, y_t * y, y_t]
    [u,s,v] = np.linalg.svd(A)
    l = v[-1,:]/v[-1,-1]
    H = l.reshape((3,3))
    return H

#5
#This function get matching positions in image 1 and image 2 and, numIters - number of iterations, and inlierTol - the toleration that considered inliers.
# This function find the best homograhpy and the most inliers and return them.
def ransacHomography(pos1, pos2, numIters, inlierTol):

    inliers = []

    n = len(pos1)
    #random 4 different integers
    for _ in range(numIters):
        J = random.sample(range(n), 4)
        P1 = np.array([pos1[j] for j in J])
        P2 = np.array([pos2[j] for j in J])
        h = leastSquaresHomography(P1, P2)
        #h, _ = cv.findHomography(P1, P2)
        pos1_tag = applyHomography(np.array(pos1), h)
        matches = []
        #for j in J:
        for j in range(len(pos1_tag)):
            #if j not in J and np.linalg.norm((pos1_tag[j] - pos2[j])) < inlierTol and np.linalg.norm((pos2[j] - pos1_tag[j])) < inlierTol:
            #if np.sum((pos1_tag[j] - pos2[j]) ** 2) < inlierTol:
            if np.sum(np.power(pos1_tag[j] - pos2[j], 2)) < inlierTol and np.sum(np.power(pos2[j] - pos1_tag[j], 2)) < inlierTol:
                matches.append(j)
        if len(matches) > len(inliers):
            inliers = matches
    P1j = np.array([pos1[i] for i in inliers])
    P2j = np.array([pos2[i] for i in inliers])
    #H12, _ = cv.findHomography(P1j, P2j)
    H12 = leastSquaresHomography(P1j, P2j)

    return H12, inliers












#5
#This function gets 2 images [im1, im2], 2 matches positions of those images [pos1, pos2] and list of inlind matches
#and mark the mathes points in red, outliers lines in blue, and inlien lines in yellow and show the map matches.
def displayMatches(im1, im2, pos1, pos2, inlind):
    h1,w1 = im1.shape
    for i in range(len(pos2)):
        pos2[i][0] += w1

    horizontal_image = np.concatenate((im1,im2), axis=1)
    horizontal_image = cv.cvtColor(horizontal_image, cv.COLOR_GRAY2BGR)
    #colors
    red = (0,0,255)
    blue = (255,0,0)
    yellow = (0,255,255)

    for i in range(len(pos1)):
        if i in inlind:
            # the inlier lines
            cv.line(horizontal_image, (int(pos1[i][0]), int(pos1[i][1])), (int(pos2[i][0]), int(pos2[i][1])), yellow)
        else:
            #the outliers lins
            cv.line(horizontal_image, (int(pos1[i][0]), int(pos1[i][1])), (int(pos2[i][0]), int(pos2[i][1])), blue)

    for i in range(len(pos1)):
        cv.circle(horizontal_image, (int(pos1[i][0]), int(pos1[i][1])), 1, red)

    for i in range(len(pos2)):
        cv.circle(horizontal_image, (int(pos2[i][0]), int(pos2[i][1])), 1, red)

    cv.imshow("Display Matches", horizontal_image)
    cv.waitKey()



#6
#This function gets Hpair - array or dict of M?1 3x3 homography matrices that transforms between coordinate systems i and i+1,
#and m ? Index of coordinate system we would like to accumulate the given homographies towards.
#The function returns Htot ? array or dict of M 3x3 homography matrices where Htot[i] transforms coordinate system i to the coordinate system having the index m.
def accumulateHomographies(Hpair, m):
    len_Hpair = len(Hpair)
    if m < 0 or m >= len_Hpair:
        return "Invalid m"
    Htot = []
    for i in range(len_Hpair+1):
        H_temp = np.eye(3)
        if i < m:
            for j in range(i, m):
                H_temp = np.dot(Hpair[j], H_temp)
            Htot.append(H_temp)
        elif i == m:
            Htot.append(H_temp)
        else:
            for j in range(m, i):
                H_temp = np.dot(H_temp, np.linalg.inv(Hpair[j]))
            Htot.append(H_temp)
    Htot = np.array(Htot)
    Htot = [[[x/Ht[2][2] for x in h] for h in Ht] for Ht in Htot]
    return Htot








def renderPanorama(im, H):
    len_im = len(im)
    i_corners = np.zeros((len_im, 4, 2))
    i_centers = np.zeros((len_im,2))

    for i in range(len_im):
        r, c = im[i].shape
        i_corners[i] = applyHomography(np.array([[0,0], [c-1,0], [c-1,r-1], [0, r-1]]), H[i])
        i_centers[i] = applyHomography(np.array([[c//2,r//2]]), H[i])
    x_corners = np.array([[corner[0] for corner in corners] for corners in i_corners])
    x_min = int(min(x_corners.flatten()))
    x_max = int(max(x_corners.flatten()))
    y_corners = np.array([[corner[1] for corner in corners] for corners in i_corners])
    y_min = int(min(y_corners.flatten()))
    y_max = int(max(y_corners.flatten()))
    x_centers = np.array([center[0] for center in i_centers])


    ###strips distribution###
    p_strip = np.zeros((len_im, 2), dtype=int)
    for i in range(len_im):
        if i == 0:
            p_strip[i] = np.array([x_min, (x_centers[i] + x_centers[i+1])//2])
        elif i == len_im-1:
            p_strip[i] = np.array([(x_centers[i-1]+x_centers[i])//2, x_max + 1])
        else:
            p_strip[i] = np.array([(x_centers[i-1]+x_centers[i])//2, (x_centers[i] + x_centers[i+1])//2])


    ########BackWrap########
    stp = []
    for i in range(len(im)):
        Hinv = np.linalg.inv(H[i])
        Hinv = [[x/Hinv[2][2] for x in h] for h in Hinv]
        r, c = im[i].shape
        strp = np.zeros((y_max - y_min + 1, p_strip[i][1] - p_strip[i][0] + 1))
        for y in range(y_max - y_min + 1):
            for x in range(p_strip[i][1] - p_strip[i][0] + 1):
                p_xy = applyHomography([[x + p_strip[i][0], y + y_min]], Hinv)
                p_x = int(p_xy[0][0])
                p_y = int(p_xy[0][1])
                if p_x >= 0 and p_x < c and p_y >= 0 and p_y < r:
                    strp[y][x] = im[i][p_y][p_x]# / 255
        stp.append(strp)
    panorama = np.concatenate((stp[:]), axis=1)
    return panorama






def generatePanorama(sequenceNameImages, showMatches=False):

    #Load grayscale sequence images
    images = []
    Hcs = []
    try:
        for nameImage in sequenceNameImages:
            images.append(loadImageInGrayscale(nameImage))
    except:
        print("Invalid input")
        return

    #Find matches in all the images
    num_of_images = len(images)
    positions = []
    for i in range(num_of_images-1):
        pos1, pos2 = getPositions(images[i], images[i+1])

        inlind = []
        #H12, _ = cv.findHomography(pos1, pos2, method=cv.RANSAC, ransacReprojThreshold=5, maxIters=250)
        H12, inlind = ransacHomography(pos1, pos2, 5, 250)
        Hcs.append(H12)


        if showMatches:
            displayMatches(images[i], images[i+1], pos1, pos2, inlind)

    Hcs = np.array(Hcs)
    Hpcs = accumulateHomographies(Hcs, len(Hcs)//2)
    bgrImages = []
    for nameImage in sequenceNameImages:
        bgrImages.append(cv.imread(nameImage))

    bgrImages = np.array(bgrImages)
    b_panorama = renderPanorama(bgrImages[:,:,:,0], Hpcs)
    g_panorama = renderPanorama(bgrImages[:,:,:,1], Hpcs)
    r_panorama = renderPanorama(bgrImages[:,:,:,2], Hpcs)

    panorama = cv.merge((b_panorama, g_panorama, r_panorama))
    return panorama










#oxford
oxfordImages = []
for i in range(1,3):
    #path = os.path.join(os.getcwd(),"data", "inp", "examples", "oxford"+str(i)+".jpg")
    #oxfordImages.append(path)
    oxfordImages.append("data/inp/examples/oxford"+str(i)+".jpg")
oxfordPanorama = generatePanorama(oxfordImages, showMatches=False)
cv.imwrite("data/out/examples/oxford.jpg", oxfordPanorama)

#backyard
backyardImages = []
for i in range(1,4):
    backyardImages.append("data/inp/examples/backyard"+str(i)+".jpg")
backyardPanorama = generatePanorama(backyardImages, showMatches=False)
cv.imwrite("data/out/examples/backyard.jpg", backyardPanorama)


#office
officeImages = []
for i in range(1,5):
    officeImages.append("data/inp/examples/office"+str(i)+".jpg")
officePanorama = generatePanorama(backyardImages, showMatches=False)
cv.imwrite("data/out/examples/office.jpg", officePanorama)





######################--------mine---------#########################
#garden
gardenImages = []
for i in range(1,3):
    gardenImages.append("data/inp/mine/garden"+str(i)+".jpg")
gardenPanorama = generatePanorama(gardenImages, showMatches=False)
cv.imwrite("data/out/mine/garden.jpg", gardenPanorama)


#amplifier
amplifierImages = []
for i in range(1,5):
    amplifierImages.append("data/inp/mine/amplifier"+str(i)+".jpg")
amplifierPanorama = generatePanorama(amplifierImages, showMatches=False)
cv.imwrite("data/out/mine/amplifier.jpg", amplifierPanorama)


