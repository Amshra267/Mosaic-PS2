import numpy as np
import cv2
import imutils
import os
from imutils.perspective import four_point_transform

# Give this function an image and direction, it will check in that direction and if encounters a white pixel it will make all other remaining pixels in that direction white.
# ht - horizontal top
# hb - horizontal bottom
# vl - vertical left
# vr - vertical right
def fill_dirn(thresh_img, dirn):
    thresh_x = thresh_img.copy()
    if dirn == "ht":
        for dirn_i in range(thresh_img.shape[1]):
            for dirn_j in range(thresh_img.shape[0]):
                if thresh_img[dirn_j][dirn_i]==255:
                    thresh_x[dirn_j:, dirn_i]=255
                    break
    
    if dirn == "hb":
        for dirn_i in range(thresh_img.shape[1]):
            for dirn_j in range(thresh_img.shape[0]):
                if thresh_img[thresh_img.shape[0]-1-dirn_j][dirn_i]==255:
                    thresh_x[0:thresh_img.shape[0]-dirn_j, dirn_i]=255
                    break
    
    if dirn == "vl":
        for dirn_i in range(thresh_img.shape[0]):
            for dirn_j in range(thresh_img.shape[1]):
                if thresh_img[dirn_i][dirn_j]==255:
                    thresh_x[dirn_i, dirn_j:]=255
                    break
    
    if dirn == "vr":
        for dirn_i in range(thresh_img.shape[0]):
            for dirn_j in range(thresh_img.shape[1]):
                if thresh_img[dirn_i][thresh_img.shape[1]-1-dirn_j]==255:
                    thresh_x[dirn_i, 0:thresh_img.shape[1]-dirn_j]=255
                    break
    
    return thresh_x.astype("uint8")

# find area of max area contour
def find_cnta(thresh_img):
    cnts_t, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_t = sorted(cnts_t, key= cv2.contourArea, reverse = True)
    return cv2.contourArea(cnts_t[0])

# provides the approx which includes the plate.
# If it can't detect the plate and can't fulfil the area constraint, it will return the approx of whole image
def extract_plate(gray):

    _ ,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    cv2.imshow("Binary1", binary)
    binary = check_h_border(binary)
    binary = check_v_border(binary)
    binary = cv2.copyMakeBorder(binary, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)


    binary = cv2.bitwise_not(binary)
    cv2.imshow("BINARY", binary)
    H = cv2.Sobel(binary, cv2.CV_8U, 0, 2)
    V = cv2.Sobel(binary, cv2.CV_8U, 2, 0)

    rows,cols = gray.shape[:2]

    contours,_ = cv2.findContours(V, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        # rows/3 is the threshold for length of line
        if h > rows/3:
            cv2.drawContours(V, [cnt], -1, 255, -1)
            cv2.drawContours(binary, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(V, [cnt], -1, 0, -1)

    contours,_ = cv2.findContours(H, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        # cols/3 is the threshold for length of line
        if w > cols/3:
            cv2.drawContours(H, [cnt], -1, 255, -1)
            cv2.drawContours(binary, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(H, [cnt], -1, 0, -1)
    cv2.imshow("SOBEL", V)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    H = cv2.morphologyEx(H, cv2.MORPH_DILATE, kernel,iterations = 3)
    V = cv2.morphologyEx(V, cv2.MORPH_DILATE, kernel, iterations = 3)

    thresh_vr = fill_dirn(V, "vr")
    thresh_vl = fill_dirn(V, "vl")
    thresh_ht = fill_dirn(H, "ht")
    thresh_hb = fill_dirn(H, "hb")
    thresh_v = cv2.bitwise_and(thresh_vl, thresh_vr)
    thresh_h = cv2.bitwise_and(thresh_ht, thresh_hb)
    thresh_xx = cv2.bitwise_and(thresh_h, thresh_v)
    #cv2.imshow("THRESH_V", thresh_v)
    nnx = np.zeros(thresh_xx.shape, np.uint8)
    cnts = cv2.findContours(thresh_v, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    #print(len(c))
    if len(c)==0:
        approx_f = np.array([[[0, 0]], [[cols, 0]], [[cols, rows]], [[0, rows]]])
        return approx_f
    else:
        c = c[0]
    hull = cv2.convexHull(c, False)
    cv2.drawContours(nnx, [hull], 0, 255, -1, 8)
    cv2.drawContours(nnx, [hull], 0, 255, 5, 8)# after getting the mask extending its boundary to get a bigger area
    cv2.imshow("NNX", nnx)
    nnx_dst = nnx.copy()
    print("HULL LENGTH: ", len(hull))
    # Tried cornerHarris but its not working properly
    # dst = cv2.cornerHarris(nnx, 5, 5, 0.1)
    # dst = cv2.dilate(dst, None)
    # nnx_dst[dst>0.01*dst.max()]=255
    # cv2.imshow("HARRIS", nnx_dst)
    # print("CORNER HARRIS: ", len(dst))
    # In the final mask of the plate calculating the approx, minAreaRect will not be much suitable
    cnts, _ = cv2.findContours(nnx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(c)==0:
        approx_f = np.array([[[0, 0]], [[cols, 0]], [[cols, rows]], [[0, rows]]])
        return approx_f
    else:
        c = c[0]
    approx_f = cv2.approxPolyDP(c, 0.06*cv2.arcLength(c, True), True)# Tested on different images 0.06 is suitable for most of them
    
    if len(approx_f)==4:
        # If rectangle detected
        print("RECT DETECTED SUCCESSFULLY")
    else:
        # If mask detected but rect not detected returning bounding rect of mask (can be changed to minAreaRect)
        re = cv2.boundingRect(c)
        cv2.rectangle(nnx_dst, (re[0], re[1]), (re[0]+re[2], re[1]+re[3]), 255, -1)
        cnts, _ = cv2.findContours(nnx_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        approx_f = cv2.approxPolyDP(c, 0.06*cv2.arcLength(c, True), True)
    if cv2.contourArea(c)<(0.2*rows*cols):
        # If mask not detected returns approx of whole image
        approx_f = np.array([[[0, 0]], [[cols, 0]], [[cols, rows]], [[0, rows]]])
    return approx_f




def approx_update(aprx):
    k = 5
    aprx[0][0][1] = aprx[0][0][1]-k
    aprx[0][0][0] = aprx[0][0][0]-k
    aprx[1][0][1] = aprx[1][0][1]+(k+5)
    aprx[1][0][0] = aprx[1][0][0]-(k+5)
    aprx[2][0][1] = aprx[2][0][1]+k
    aprx[2][0][0] = aprx[2][0][0]+k
    aprx[3][0][1] = aprx[3][0][1]-k
    aprx[3][0][0] = aprx[3][0][0]+k
    return aprx

# 1st gamma correction method (not working properly)
def linear_stretching(input, lower_stretch_from, upper_stretch_from):
    """
    Linear stretching of input pixels
    :param input: integer, the input value of pixel that needs to be stretched
    :param lower_stretch_from: lower value of stretch from range - input
    :param upper_stretch_from: upper value of stretch from range - input
    :return: integer, integer, the final stretched value
    """

    lower_stretch_to = 0  # lower value of the range to stretch to - output
    upper_stretch_to = 255  # upper value of the range to stretch to - output

    output = (input - lower_stretch_from) * ((upper_stretch_to - lower_stretch_to) / (upper_stretch_from - lower_stretch_from)) + lower_stretch_to

    return output

def gamma_correction(moon):
    """
    Restore the contrast in the faded image using linear stretching.
    """

    # assign variable to max and min value of image pixels
    max_value = np.max(moon)
    min_value = np.min(moon)

    # cycle to apply linear stretching formula on each pixel
    for y in range(len(moon)):
        for x in range(len(moon[y])):
            moon[y][x] = linear_stretching(moon[y][x], min_value, max_value)
    return moon

# 2nd gamma correction method, increase illumination
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def check_h_border(thresh_img):
    n_row, n_column = thresh_img.shape
    thresh_img = cv2.bitwise_not(thresh_img)
    prs = 0
    i = 0
    while(i<(n_row//5)):
        row = thresh_img[i]
        row = row/255
        rs = np.sum(row)
        if rs<(prs//2):
            thresh_img[:i] = 0
            
        else:
            if rs>=prs:
                prs = rs
        i += 5

    thresh_img = cv2.flip(thresh_img, 0)
    prs = 0
    i = 0
    while(i<(n_row//5)):
        row = thresh_img[i]
        row = row/255
        rs = np.sum(row)
        if rs<(prs//2):
            thresh_img[:i] = 0
            
        else:
            if rs>=prs:
                prs = rs
        i += 5
    thresh_img = cv2.flip(thresh_img, 0)
    thresh_img = cv2.bitwise_not(thresh_img)

    return thresh_img

def check_v_border(thresh_img):
    n_row, n_column = thresh_img.shape
    thresh_img = cv2.bitwise_not(thresh_img)
    pcs = 0
    i = 0
    while(i<(n_column//15)):
        column = thresh_img[:, i]
        column = column/255
        cs = np.sum(column)
        if cs<(pcs//2):
            thresh_img[:, :i] = 0
            break
        else:
            if cs>=pcs:
                pcs = cs
            i += 5

    thresh_img = cv2.flip(thresh_img, 1)
    pcs = 0
    i = 0
    while(i<(n_column//15)):
        column = thresh_img[:, i]
        column = column/255
        cs = np.sum(column)
        if cs<(pcs//2):
            thresh_img[:, :i] = 0
            break
        else:
            if cs>=pcs:
                pcs = cs
            i += 5

    thresh_img = cv2.flip(thresh_img, 1)
    thresh_img = cv2.bitwise_not(thresh_img)

    return thresh_img

def extraction(image):

    # in case of a rear view of whole car take roi of the plate portion, in case of only plate take roi of whole image
    image = cv2.resize(image, (320, 240))
    image = cv2.normalize(image, image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
   # r = cv2.selectROI("ROI", image, showCrosshair=False, fromCenter=False)
   # image = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #image = cv2.resize(image, (320, 240))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("GRAY", gray)
    # Plate Exatraction
    aprx_f = extract_plate(gray.copy())
    #approx_f = approx_update(approx_f)
    nnx_f = np.zeros(gray.shape, np.uint8)
    nnx_f = cv2.drawContours(nnx_f, [aprx_f], 0, 255, -1)
    cv2.imshow("NNX_F", nnx_f)
    wraped = cv2.bitwise_and(gray, gray, mask = nnx_f)
    
    # Four point Transform (approx_f should be reshaped in size (4, 2))
    wraped = four_point_transform(wraped, aprx_f.reshape(4, 2))
    cv2.imshow("Wraped", wraped)
    #wraped = cv2.resize(wraped, (320, 240))
    # Normalize image
    #norm = cv2.normalize(wraped, wraped, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #cv2.imshow("NORM", norm)

    # Gamma adjustment and Thresholding ((80, 100) is better range)
    #wraped = adjust_gamma(wraped, gamma=1.5)
    blurred = cv2.GaussianBlur(wraped, (5, 5), 0)
    ret, thresh = cv2.threshold(wraped, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    

    # Detecting contours of characters and filling their inside spaces and drawing contours in another black region which filters out noise
    thresh_n = thresh.copy()
    #thresh_n = cv2.erode(thresh_n, np.zeros((3, 3), np.uint8), iterations=1)
    thresh_n = cv2.copyMakeBorder(thresh_n, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    cnts, _ = cv2.findContours(thresh_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    thresh_nn = np.zeros(thresh_n.shape, np.uint8)
    rects = []
    for i, c in enumerate(cnts):
        rect_t = cv2.boundingRect(c)
        if (rect_t[2]*rect_t[3])<(0.1*480*360) and (rect_t[2]*rect_t[3])>500:
            thresh_nn = cv2.drawContours(thresh_nn, [c], 0, 255, -1)
            print(rect_t[2]*rect_t[3])
    
    # From the previous black region detecting bonding rects and cropping that portion from the previous threshold image (thresh_n)
    # This time tere will be only one contour for a character at max because its inside region is filled
    #thresh_nn = cv2.erode(thresh_nn, np.ones((5, 5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(thresh_nn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rects = []
    for c in cnts:
        rect_t = cv2.boundingRect(c)
        if (rect_t[2]*rect_t[3])<(0.08*480*360) and (rect_t[2]*rect_t[3])>500:
            rects.append(rect_t)

    # Sowing the extracted rects
    for ind, rect_t in enumerate(rects):
        # Get the 4 points of the bounding rectangle
        x, y, w, h = rect_t
        # Draw a straight rectangle with the points
        s = thresh_n[y:(y+h), x:(x+w)]
        cv2.imshow(str(ind), s)
        yield s

if __name__ == '__main__':
    path = input("Path: \n")
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    
    # in case of a rear view of whole car take roi of the plate portion, in case of only plate take roi of whole image
    


    # #cv2.imshow("EDGED", edged)
    # cv2.imshow("THRESH", thresh)
    # cv2.imshow("THRESH_N", thresh_n)
    # cv2.imshow("THRESH_NN", thresh_nn)
    # #cv2.imshow("Wraped", wraped)
    # cv2.imshow("BLURRED", blurred)
    # cv2.imshow("original", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()