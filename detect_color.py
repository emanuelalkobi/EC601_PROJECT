import numpy as np
import os
import sys
import cv2

def count_nonblack_np(img):
    return img.any(axis=-1).sum()

def detect_team(image,color1,colo2, show = False):
    # define the list of boundaries
    boundaries = [
    ([17, 15, 100], [50, 56, 200]), #red
    ([25, 146, 190], [96, 174, 250]), #yellow
    ([17, 90, 20], [100, 250, 100]), #green
    ([90, 3, 1], [230, 70, 100]) #blue
    ]

    colors=['red','yellow','green','blue']
    i = 0
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        tot_pix = count_nonblack_np(image)
        color_pix = count_nonblack_np(output)
        ratio = color_pix/tot_pix
        if ratio > 0.01 :
            return colors[i]
        i += 1

        if show == True:
            #cv2.imshow("images", np.hstack([image, output]))
            #cv2.waitKey(100000)
            #cv2.destroyAllWindows()
            cv2.imwrite(str(i)+".jpg",np.hstack([image, output]))
    return 'not_sure'




#def main():
#    img=sys.argv[1]
#    print("image name is ",img)
#    img=cv2.imread(sys.argv[1])
   # detect_team(img, show = True)
#    detect_team(img, True)
#main()
