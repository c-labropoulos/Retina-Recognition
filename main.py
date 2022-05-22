
# import required module
import os
import numpy as np
import cv2
from PIL import Image

# assign directory
directory = '/home/chris/Desktop/RIDB-20220510T095046Z-001/RIDB'

for image in os.listdir(directory):
    image_filename = directory + "/" + image
    # print(image_filename)
    # Converting the image to RGB mode
  #  img = Image.open(image_filename)
  #  img = np.asarray(img)
    #img_gray = img.convert('L')
    img_gray = np.array(Image.open(image_filename).convert('L'))
    # Calling the floodfill() function and
    # passing it image, seed, value and
    # thresh as arguments
    img_floodfill = img_gray.copy()

    cv2.floodFill(img_floodfill, None, (0, 0), 255)

    h, w = img_floodfill.shape[:2]

    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    #dilate the floodfill result to cover pixels near the ROI (retina image)
    dil_kernel = np.ones((13, 13), np.uint8)
    #bitwiste_not and binarization
    floodfill_mask = cv2.bitwise_not(src=img_floodfill)
    floodfill_mask = cv2.threshold(floodfill_mask, 1, 255, cv2.THRESH_BINARY)[1]
    floodfill_mask = cv2.erode(floodfill_mask, dil_kernel, iterations=1)
    # enhance contrast of the grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_contrast = clahe.apply(img_gray)
    img_contrast = 255 - img_contrast
    img_contrast = clahe.apply(img_contrast)
    #blur image with the gaussian kernel
    img_gauss = cv2.GaussianBlur(img_contrast, (7, 7), 0)
#adaptive thresholding with gaussian-weighted sum of the neighbourhood values
    im_threshold = cv2.adaptiveThreshold(img_gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, C=0)
#combination threshold output with previously prepared mask
    newimg = cv2.bitwise_and(im_threshold, floodfill_mask)
    #blur image with the median filter  and invert values
    newimg = cv2.medianBlur(newimg, 5)
    newimg = cv2.bitwise_not(newimg)
    #fill the holes with morphological close operation and invert the image
    newimg = cv2.morphologyEx(newimg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    newimg = cv2.bitwise_not(newimg)
    #get objects stats and remove background object
    nb_components, output, stats, centroids =  cv2.connectedComponentsWithStats(newimg, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    #remove smaller objects (area < 500)
    img2 = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if stats[i + 1, cv2.CC_STAT_AREA] >= 500:
            img2[output == i + 1] = 255
    #get the skeleton of the preserved objects
    img_obj_filtered=img2.copy()
    skel_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(img_obj_filtered.shape, np.uint8)
    while True:
      opened = cv2.morphologyEx(img_obj_filtered, cv2.MORPH_OPEN,skel_element)
      temp = cv2.subtract(img_obj_filtered, opened)
      eroded = cv2.erode(img_obj_filtered, skel_element)
      skel = cv2.bitwise_or(skel, temp)
      img_obj_filtered = eroded.copy()
      if cv2.countNonZero(img_obj_filtered) == 0:
        break
    cv2.imshow("Retina", skel)
    cv2.waitKey()