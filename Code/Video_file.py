import cv2
import time
import datetime
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

def getmin(p1,p2,p3,p4): # get min value of x and y from all 4 points
    min_x =  min(p1[0],p2[0],p3[0],p4[0])
    min_y =  min(p1[1],p2[1],p3[1],p4[1])
    return(min_x,min_y) 

def getmax(p1,p2,p3,p4): #get max value of x and y from all 4 points
    max_x =  max(p1[0],p2[0],p3[0],p4[0])
    max_y =  max(p1[1],p2[1],p3[1],p4[1])
    return(max_x,max_y)

def maskimage(image, p1,p2,p3,p4): 
    mask = np.zeros(image.shape, dtype=np.uint8) # create a empty matrix of size equal to image
    roi_corners = np.array([[p1,p2,p3,p4]], dtype=np.int32)
    channel_count = image.shape[2] # channel/color count
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\GoPro\\New folder\\'
    cv2.imwrite(dirname+parking_space_loc+'Masked_Image.bmp', masked_image)
    return(masked_image)

def cannyedgedetection(spot):
    sigma=0.33
    median = np.median(spot)
    lower_bound = int(max(0, (1.0 - sigma) * median)) 
    upper_bound = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(spot,lower_bound,upper_bound)
    white_pixels = countwhitepixels(edges)
    dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\GoPro\\New folder\\Edges\\'
    cv2.imwrite(dirname+parking_space_loc+'Edge.bmp', edges)
    return white_pixels

def countwhitepixels(image):
    count = 0
    for row in image:
        for col in row:
            if col == 255:
                count = count + 1
    return(count)
    
def drawBoundBox(image, p1, p2, p3, p4, color):
    points = np.array([p1,p3,p4,p2], np.int32)
    points = points.reshape((-1,1,2))
    cv2.polylines(image,[points],True,color)
    return image

with open('C:\\Users\\Nitz Mistry\\Pictures\\GoPro\\New folder\\coords_new.json') as empty_spot_data_json:
    empty_spot_data = json.load(empty_spot_data_json)
    
with open('C:\\Users\\Nitz Mistry\\Pictures\\GoPro\\New folder\\spot_data.json') as spot_data:
    spot_data = json.load(spot_data)
    
#f = open("C:\\Users\\Nitz Mistry\\Pictures\\GoPro\\parking_output.txt", "w")


if __name__ == "__main__":
    
    frame_count = 0
    #seconds = 0
    #seconds_end = 120
    logic_threshold = 400
    white_diff_threshold = 10
    
    red_color = (0,0,255)
    green_color = (0,255,0)
    spot_colors = []
    
    #vid = cv2.VideoCapture('http://njitITSRC:njit1234@166.249.52.131:9002/stream1')
    #vid = cv2.VideoCapture('rtsp://njitITSRC:njit1234@192.168.137.135:8081/stream1')
    #vid = cv2.VideoCapture('http://192.168.1.23:8081')
    vid = cv2.VideoCapture('C:\\Users\\Nitz Mistry\\Pictures\\GoPro\\New folder\\gopro.mp4',0)
    #frame = cv2.imread('C:\\Users\\Nitz Mistry\\Pictures\\vlcsnap-2019-08-08-01h26m58s143.png')
    
    #vid = cv2.VideoCapture('http://192.168.13.101:8081')
    
    if vid.isOpened() == False:
        print("ERROR: File Not Found or Wrong Video Codex Used")
    
    print(vid.isOpened())
    
    while vid.isOpened():
        ret, image = vid.read()
        cur_time = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%Y_%H_%M_%S')
        
        #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_13_8\\images\\'
        #cv2.imwrite(dirname+'image_'+cur_time+'.jpg', image)
        
        #print(ret)
        #cv2.imshow("Frame", frame)
        #ret = False
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        if ret == True:
            #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_25_7\\'
            #cv2.imwrite(dirname+'image.jpg', image)
            #cv2.imshow("Frame", image)
            #if cv2.waitKey(0) & 0xFF == ord('q'):
            #    break
            #image=frame
            if frame_count % 60 == 0:
                #print(seconds)
                #f.truncate(0)
                spot_coords = empty_spot_data['shapes']
            
                for parking_spot in spot_coords:
                    #if(int(parking_spot['label']) == 1):
                        #print("############################")
                    parking_space_loc = 'Space_#_'+str(parking_spot['label'])
                    points=parking_spot['points']
                    
                    p1 = (int(points[0][0]),int(points[0][1]))
                    p2 = (int(points[1][0]),int(points[1][1]))    
                    p3 = (int(points[2][0]),int(points[2][1]))
                    p4 = (int(points[3][0]),int(points[3][1])) 
                    #print(p1,p2,p3,p4)
                    masked_image = maskimage(image,p1,p2,p3,p4)
                    min_point = getmin(p1,p2,p3,p4)
                    #print(min_point)
                    max_point = getmax(p1,p2,p3,p4)
                    #print(max_point)
                    masked_parking_space = masked_image[min_point[1]:max_point[1], min_point[0]:max_point[0]]
                    #print(masked_parking_space)
                    #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_25_7\\'
                    #cv2.imwrite(dirname+parking_space_loc+'masked_parking_space.bmp', masked_parking_space)
                    
                    denoise = cv2.fastNlMeansDenoisingColored(masked_parking_space,None,20,21,7,31) #gets rid of image noise
                    #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_9_7\\'
                    #cv2.imwrite(dirname+parking_space_loc+'denoise.bmp', denoise)
                    
                    gray_image = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
                    #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_9_7\\'
                    #cv2.imwrite(dirname+parking_space_loc+'gray_image.bmp', gray_image)
                    
                    blur = cv2.GaussianBlur(gray_image,(5,5),0)
                    #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_9_7\\'
                    #cv2.imwrite(dirname+parking_space_loc+'blur.bmp', blur)
                    
                    number_of_white = cannyedgedetection(blur)
                    #print(number_of_white)

                    white_diff_from_empty = number_of_white - spot_data['spots'][int(parking_spot['label'])-1][1]
                    print(white_diff_from_empty)
                    
                    if white_diff_from_empty > white_diff_threshold:
                        spot_colors.append([p1,p2,p3,p4, red_color])
                        print(parking_space_loc + ": FULL")
                        #f.writelines([parking_space_loc + ": FULL"])
                        #f.write('\n')
                           
                    else:
                        spot_colors.append([p1,p2,p3,p4, green_color])
                        print(parking_space_loc + ": EMPTY")
                        #f.writelines([parking_space_loc + ": EMPTY"])
                        #f.write('\n')
            
            for spot in spot_colors: #loop through the list of spotColors
                frame = drawBoundBox(image, spot[0], spot[1], spot[3], spot[2], spot[4]) #applies the color to each spot
            
            dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_13_8\\Images\\'
            cv2.imwrite(dirname+'image_'+cur_time+'.jpg', image)
            
            cv2.imshow("Frame", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            #if seconds == seconds_end:
            #    break

            frame_count = frame_count + 1
            #print(frame_count)
            #if frame_count == 30:
            #    seconds = seconds + 1
            #    frame_count = 0
            #if seconds == 60:
            #    break

        else:
            break
#f.close()
vid.release()
cv2.destroyAllWindows()