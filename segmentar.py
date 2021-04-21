"""
   This script contains all the codes for segmenting the Indian number plates

   It contains a hierarchial procedure for extracting very noisy and bad images in Indian number plates

"""

import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import argrelextrema
W=320 # width is fixed



def fill_dirn(thresh_img, dirn):

    """
    It is used for filling
    Give this function an image and direction, it will check in that direction and if encounters a white pixel it will make all other remaining pixels in that direction white.

    Arguments:
        dirn: direction of filling - includes
                                        ht - horizontal top
                                        hb - horizontal bottom
                                        vl - vertical left
                                        vr - vertical right
    Returns:
        Filled image along a particular direction
    
    """
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


def smooth_data_convolve_my_average(arr, span):                                ## Function for smoothing the curve 
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

    # The "my_average" part: shrinks the averaging window on the side that 
    # reaches beyond the data, keeps the other side the same size as given 
    # by "span"
    re[0] = np.average(arr[:span])
    for i in range(1, span + 1):
        re[i] = np.average(arr[:i + span])
        re[-i] = np.average(arr[-i - span:])
    return re

def v_projection_or_bruteforce_trimming(img, half_detected = True, min=None):       # Our novel proposed and implemeted approach for bigger noiser data
                                                                                    # But this approach have some false positive which can be addressed 
                                                                                    # in future if we get more time,
    """
    This function is used for divind a giving image into subparts if multiple letters are present else return the same image
    """
    h, w = img.shape
    combined_list = [0] # list contains max and intial and final
    split_imgs = []
    #vertical_projection
    col_pix_count = np.sum(img, axis = 0)
  # Using moving average method with scipy lowess to find the maxima and reduce noise to cut the parts,-----------
    maxs = argrelextrema(smooth_data_convolve_my_average(col_pix_count, 3), np.less)[0]
    combined_list.extend(maxs)
    combined_list.append(w-1)
   # print(combined_list)
    if len(maxs)==0:
        return [img]

    elif half_detected: # if segmentation is half done already then
        filtered_list = [0]
        i=0
        j = 0
     #   print("Minimum : ",min)
        if combined_list[-1]<=min:
            return [img]
        if combined_list[-1]>40:
            min = 17
        while((i < len(combined_list)-1) and (i+j < len(combined_list)-1)):
            diff = -(combined_list[i]-combined_list[i+1+j])
            if diff<=min:
                j+=1
            else:
                if -(combined_list[i+j+1]-combined_list[-1])<min:
                    filtered_list.append(combined_list[-1])
                    break
                filtered_list.append(combined_list[i+j+1])
                i+=j+1
                j=0
                
    else:
        filtered_list = [0]
        i=0
        j = 0
        min=10
        while((i < len(combined_list)-1) and (i+j < len(combined_list)-1)):
            diff = -(combined_list[i]-combined_list[i+1+j])
            if diff<=min:
                j+=1
            else:
                if -(combined_list[i+j+1]-combined_list[-1])<min:
                    filtered_list.append(combined_list[-1])
                    break
                filtered_list.append(combined_list[i+j+1])
                i+=j+1
                j=0

   # print(filtered_list)
    #plt.plot(col_pix_count, color = "b")
    #plt.plot(smooth_data_convolve_my_average(col_pix_count, 3), color = "r")
    #plt.show()
    for i in range(len(filtered_list)-1):
        split_imgs.append(img[:,filtered_list[i]: filtered_list[i+1]])
    return split_imgs

def extract_plate(gray):
    """
    this function provides the approx polyDP coordinates which includes the plate.
    If it can't detect the plate and can't fulfil the area constraint, it will return the approx of whole image

    Arguments:
           gray: Grayscale image
    
    Retuens:
          approx points along with binary thresholded image
    
    """

    h1, w1 = gray.shape
    dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.bilateralFilter(dilated_img, 11, 17, 17)  #using median blur to remove the undesired shadow along with abs difference and normalization
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    ret,binary1 = cv2.threshold(norm_img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
#     binary = check_h_border(binary)
#     binary1 = check_v_border(binary)
    binary = cv2.bitwise_not(binary1)
#     cv2.imshow("BINARY", binary)
    
    """ NOW we will use the vertical filling to determine the effective area of number plate """

    ### ----------------------------------------Firstly calculating and using vertival sobel---------------------------------#####
   
    V = cv2.Sobel(binary, cv2.CV_8U, 2, 0)
    V= cv2.dilate(V, np.ones((3,2), np.uint8), iterations =1)
    contours,_ = cv2.findContours(V, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        # rows/3 is the threshold for length of line
        if h > h1/4:
            cv2.drawContours(V, [cnt], -1, 255, -1)
            cv2.drawContours(binary, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(V, [cnt], -1, 0, -1)
    
    
#     cv2.imshow("Sobel", V)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
    V = cv2.morphologyEx(V, cv2.MORPH_DILATE, kernel, iterations = 3)
    thresh_vr = fill_dirn(V, "vr")
    thresh_vl = fill_dirn(V, "vl")
    thresh_v = cv2.bitwise_and(thresh_vl, thresh_vr)
    #cv2.imshow("THRESH_V", thresh_v)
    nnx = np.zeros(thresh_v.shape, np.uint8)
    cnts = cv2.findContours(thresh_v, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(c)==0:
        approx_f = np.array([[[0, 0]], [[w1, 0]], [[w1, h1]], [[0, h1]]])
        return approx_f, binary1
    else:
        c = c[0]
    coordi2 = cv2.boundingRect(c)
    width_v = coordi2[2]
    hull = cv2.convexHull(c, False)
    cv2.drawContours(nnx, [hull], 0, 255, -1, 8)
    cv2.drawContours(nnx, [hull], 0, 255, 5, 8)# after getting the mask extending its boundary to get a bigger area
#     cv2.imshow("NNX", nnx)
    
    ###----------------For H-SOBEL----------------------------------------
    H = cv2.Sobel(binary, cv2.CV_8U, 0, 2)
    H= cv2.dilate(H, np.ones((2,3), np.uint8), iterations =1)
    thresh_ht = fill_dirn(H, "hb")
    thresh_hb = fill_dirn(H, "ht")
    thresh_h = cv2.bitwise_and(thresh_ht, thresh_hb)
    cnts = cv2.findContours(thresh_h, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse  =True)[0]
    coordi = cv2.boundingRect(cnts)
    width_h = coordi[2]
    cv2.rectangle(thresh_h, (coordi[0], coordi[1]), (coordi[0]+coordi[2], coordi[1]+coordi[3]), 255, -1)
#     cv2.imshow("Sobel-H", H)
#     cv2.imshow("thresh_h", thresh_h)

   ##-------------------------------------------------------------------------


   #####------------------Taking help of both V and H sobel to get the resultant nxx mask----------------
    
#     print(width_v/width_h)
    if width_v/width_h < 0.6: ## if width relativeness less than 0.6 so detect bigger width
        cv2.rectangle(nnx,  (coordi[0], coordi2[1]), (coordi2[0]+coordi2[2], coordi2[1]+coordi2[3]), 255, -1)

#     cv2.imshow("nnx", nnx)
    nnx_dst = nnx.copy()
    cnts, _ = cv2.findContours(nnx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(c)==0:
        approx_f = np.array([[[0, 0]], [[w1, 0]], [[w1, h1]], [[0, h1]]])
        return approx_f, binary1
    else:
        c = c[0]
    approx_f = cv2.approxPolyDP(c, 0.05*cv2.arcLength(c, True), True)# Tested on different images 0.06 is suitable for most of them
   # print("Approx_f :",len(approx_f))
    if len(approx_f)==4:
        # If rectangle detected
        print("RECT DETECTED SUCCESSFULLY")
    else:
        # If mask detected but rect not detected returning bounding rect of mask (can be changed to minAreaRect)
        re  = cv2.boundingRect(c)
        cv2.rectangle(nnx_dst, (re[0], re[1]), (re[0]+re[2], re[1]+re[3]), 255, -1)
        cnts, _ = cv2.findContours(nnx_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        approx_f = cv2.approxPolyDP(c, 0.05*cv2.arcLength(c, True), True)
        if len(approx_f!=4):
            approx_f = np.array([[[0, 0]], [[w1, 0]], [[w1, h1]], [[0, h1]]])
        

    if cv2.contourArea(c)<(0.2*w1*h1):
        # If mask not detected returns approx of whole image
        approx_f = np.array([[[0, 0]], [[w1, 0]], [[w1, h1]], [[0, h1]]])
    return approx_f, binary1


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


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    pts=  pts.reshape(4,2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth-1, maxHeight-1), flags = cv2.INTER_AREA)
    # return the warped image
    return warped
  

def sort_x(cnt):
    """
    Retuening the width value of a contour for sorting
    """
    return cnt[0]

def extraction(path):
    """
    Final generator type function for detecting the plate and combining all helper functions to segment the letters
    """

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]
    H = int(W*h/w)
    image = cv2.resize(image, (W, H), cv2.INTER_AREA)
    image = cv2.normalize(image, image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Plate Exatraction
    aprx_f, binary = extract_plate(gray.copy())
    
    nnx_f = np.zeros(binary.shape, np.uint8)
    nnx_f = cv2.drawContours(nnx_f, [aprx_f], 0, 255, -1)
#     cv2.imshow("NNX_F", nnx_f)
    wraped = cv2.bitwise_and(binary, binary, mask = nnx_f)

    # Four point Transform
    wraped = four_point_transform(wraped, aprx_f)

#     bg_img = cv2.bilateralFilter(wraped, 13, 15, 15)  #using median blur to remove the undesired shadow along with abs difference and normalization
#     _, thresh = cv2.threshold(bg_img, 110, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    thresh_n = wraped.copy()

#     cv2.imshow("wraped", wraped)
    thresh_n = cv2.copyMakeBorder(thresh_n, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    cnts, _ = cv2.findContours(thresh_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    thresh_nn = np.zeros(thresh_n.shape, np.uint8)
    rects = []
    for i, c in enumerate(cnts):
        rect_t = cv2.boundingRect(c)
        if ((rect_t[2]*rect_t[3])<(0.7*w*h) and (rect_t[2]*rect_t[3])>160) and rect_t[2]>10:
            thresh_nn = cv2.drawContours(thresh_nn, [c], 0, 255, -1)
         #   print(rect_t[2]*rect_t[3])
    
    # From the previous black region detecting bonding rects and cropping that portion from the previous threshold image (thresh_n)
    # This time tere will be only one contour for a character at max because its inside region is filled
    #thresh_nn = cv2.erode(thresh_nn, np.ones((5, 5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(thresh_nn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rects = []
    print("Contours : ",len(cnts))
    for c in cnts:
        rect_t = cv2.boundingRect(c)
        if ((rect_t[2]*rect_t[3])<(0.7*w*h) and (rect_t[2]*rect_t[3])>160) and (rect_t[2]>10):
#             print(rect_t[2]*rect_t[3])
            rects.append(rect_t)

    #-------------------Now return the characters if segmented else applying further cutting -----------#
    chars = []
    if len(rects)!=0:  # means at least some characters detected

        rects = sorted(rects, key = sort_x)

       #Appraoch-2 here starts the second hierarchial approach after normal sobel technique
        widths = np.array([rect[2] for rect in rects])
        median = np.median(widths[np.argsort(widths)])
        # Sowing the extracted rects
        for ind, rect_t in enumerate(rects):
            # Get the 4 points of the bounding rectangle
            x, y, w, h = rect_t
            # Draw a straight rectangle with the points
#             cv2.imshow("THRESH_NN", thresh_nn)
            s = thresh_n[y:(y+h), x:(x+w)]
            s = cv2.bitwise_not(s)
            trimmed = v_projection_or_bruteforce_trimming(s, half_detected = True, min = median)
            
            for trim in trimmed:
                h1, w1 = trim.shape
                if h1>w1:
                    diff = h1-w1
                    trim = cv2.copyMakeBorder(trim, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=0)
                else:
                    diff = w1-h1
                    trim = cv2.copyMakeBorder(trim, 2, 2, 4, 4, cv2.BORDER_CONSTANT, value=0)

                trim = cv2.resize(trim, (64,64), cv2.INTER_AREA)
                #trim = cv2.erode(trim, np.ones((2,2), np.uint8), iterations = 1)
                chars.append(trim)
#                 cv2.imshow("final", cv2.bitwise_not(cv2.bitwise_and(binary, binary, mask = nnx_f)))
            
    else:
        ## -----Here we will use the concept of minimum extraction to reove the plates--------------##
        h,w = thresh_n.shape
        
        final = cv2.bitwise_not(cv2.bitwise_and(binary, binary, mask = nnx_f))
        final = four_point_transform(final, aprx_f)
#         cv2.imshow("final", final)
        trimmed = v_projection_or_bruteforce_trimming(final, half_detected = False)
        for trim in trimmed:
            h1, w1 = trim.shape
            if h1>w1:
                diff = h1-w1
                trim = cv2.copyMakeBorder(trim, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=0)
            else:
                diff = w1-h1
                trim = cv2.copyMakeBorder(trim, 2, 2, 4, 4, cv2.BORDER_CONSTANT, value=0)
            
            trim = cv2.resize(trim, (64,64), cv2.INTER_AREA)
          #  trim = cv2.erode(trim, np.ones((2,2), np.uint8), iterations = 1)
            chars.append(trim)
#     cv2.imshow("original", image)
    return chars



if __name__ == '__main__':

    extraction(input("Path: \n"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    SP(1 L)
    Y
    Y
    SP
    SP
    Y
    SP

    """