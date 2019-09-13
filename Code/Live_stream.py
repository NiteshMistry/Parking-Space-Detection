import cv2
import time
import datetime
import numpy as np
import os
import json
from PIL import Image
import schedule
import requests

def getmin(p1,p2,p3,p4):
    """To get min value of x and y from all 4 points."""
    min_x =  min(p1[0],p2[0],p3[0],p4[0])
    min_y =  min(p1[1],p2[1],p3[1],p4[1])
    return(min_x,min_y) 

def getmax(p1,p2,p3,p4):
    """To get max value of x and y from all 4 points."""
    max_x =  max(p1[0],p2[0],p3[0],p4[0])
    max_y =  max(p1[1],p2[1],p3[1],p4[1])
    return(max_x,max_y)

def maskimage(image, p1,p2,p3,p4):
    """Takes image and parking lot coordinates. Keeps the original color for that parking spot. Masks the other area."""
    mask = np.zeros(image.shape, dtype=np.uint8) # create a empty matrix of size equal to image
    roi_corners = np.array([[p1,p2,p3,p4]], dtype=np.int32) # convert parking coordinates to array
    channel_count = image.shape[2] # get channel/color count
    ignore_mask_color = (255,)*channel_count # this is used in next line to ignore mask color on parking spot coordinates
    cv2.fillPoly(mask, roi_corners, ignore_mask_color) # open cv function to draw polygon on masked image and ignore mask color on that coordinates
    masked_image = cv2.bitwise_and(image, mask) # open cv function to merge image and mask/empty matrix
    #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_9_7\\'
    #cv2.imwrite(dirname+parking_space_loc+'Masked_Image.bmp', masked_image)
    return(masked_image)

def cannyedgedetection(spot):
    """This function draws the edges on the image and returns the total white pixel from the edges."""
    sigma=0.33
    median = np.median(spot)
    lower_bound = int(max(0, (1.0 - sigma) * median)) 
    upper_bound = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(spot,lower_bound,upper_bound) # opencv function to create edges on the image
    white_pixels = countwhitepixels(edges)
    #dirname = 'C:\\Users\\itsrc01\\Desktop\\Parking Spot Detection\\Test_15_7\\Edges\\'
    #cv2.imwrite(dirname+parking_space_loc+'Edge.bmp', edges)
    return white_pixels

def countwhitepixels(image):
    """This function counts number of white pixels in an edge detected image."""
    count = 0
    for row in image:
        for col in row:
            if col == 255:
                count = count + 1
    return(count)
    
def drawBoundBox(image, p1, p2, p3, p4, color):
    """This function draws the polygon on the image with the color specified."""
    points = np.array([p1,p3,p4,p2], np.int32)
    points = points.reshape((-1,1,2))
    cv2.polylines(image,[points],True,color)
    return image


def processing():
    """
    This function does following tasks:
    1) Reads coords.json, which has pixel coordinates for the parking spot
    2) Reads spot_data.json, which has white pixel count for each empty parking spot
    3) Connects with the camera from the IP provided
    4) Applies the logic for each parking spot and saves the result in the python dictionary
    5) Converts the python dictionary into json and sends to server
    6) Draws the colored polygon according to the result and save it as .jpg on local machine with a timestamp
    7) Sends the .jpg image to the server
    8) Releases the camera object
    """
    
    dirname = 'C:\\Users\\itsrc01\\Desktop\\Parking Spot Detection\\Test_1_8\\'
    
    
    with open(dirname+'coords.json') as empty_spot_data_json: # has pixel coordinates for the parking spot
        empty_spot_data = json.load(empty_spot_data_json)
    
    with open(dirname+'spot_data.json') as spot_data: # has white pixel count for each empty parking spot
        spot_data = json.load(spot_data)
    
    #f = open("C:\\Users\\itsrc01\\Desktop\\Parking Spot Detection\\Test_15_7\\parking_output.txt", "w")
    
    #logic_threshold = 400
    white_diff_threshold = 10 # this is a difference threshold of white pixels between current parking spot and empty parking spot
    
    red_color = (0,0,255) # red color value in BGR
    green_color = (0,255,0) # green color value in BGR
    spot_colors = [] 
    spot_coords = empty_spot_data['shapes'] # get parking spot coordinates
    data = {} # created python dictionary to store each parking spot results
    data['spots'] = []
    
    vid = cv2.VideoCapture('http://192.168.13.101:8081/') # local IP for connecting with the camera
    
    if vid.isOpened() == False: # if false print error
        print("ERROR: File Not Found or Wrong Video Codex Used")
    
    ret, image = vid.read() # vid.read() returns 2 value, 1st is TRUE (if frame is available) or FALSE (if frame is not available) and 2nd is the frame
    
    #cv2.imwrite(dirname+'image.jpg', image) # to save the frame in local directory
    
    cur_time = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%Y_%H_%M_%S') # get the current timestamp in mm_dd_yyyy_hrs_mins_secs
    
    if ret == True:
        for parking_spot in spot_coords: # iterate over all the parking spot
            if(int(parking_spot['label']) == 1):
                print("############################")
            parking_space_loc = 'Space_#_'+str(parking_spot['label']) # get parking spot number and format it like Space_#_1
            points=parking_spot['points'] # get coordinates for 4 polygon points
                    
            p1 = (int(points[0][0]),int(points[0][1])) # point 1
            p2 = (int(points[1][0]),int(points[1][1])) # point 2
            p3 = (int(points[2][0]),int(points[2][1])) # point 3
            p4 = (int(points[3][0]),int(points[3][1])) # point 4
                      
            masked_image = maskimage(image,p1,p2,p3,p4) # this returns the image keeping parking spot colored and rest area masked
            min_point = getmin(p1,p2,p3,p4) # get min point
            #print(min_point)
            max_point = getmax(p1,p2,p3,p4) # get max point
            #print(max_point)
            masked_parking_space = masked_image[min_point[1]:max_point[1], min_point[0]:max_point[0]] # this returns the image cropping to just the parking spot
            #print(masked_parking_space)
            #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_9_7\\'
            #cv2.imwrite(dirname+parking_space_loc+'masked_parking_space.bmp', masked_parking_space)
                    
            denoise = cv2.fastNlMeansDenoisingColored(masked_parking_space,None,20,21,7,31) #gets rid of image noise
            #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_9_7\\'
            #cv2.imwrite(dirname+parking_space_loc+'denoise.bmp', denoise)
                    
            gray_image = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY) # convert color image to grayscale
            #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_9_7\\'
            #cv2.imwrite(dirname+parking_space_loc+'gray_image.bmp', gray_image)
                    
            blur = cv2.GaussianBlur(gray_image,(5,5),0) # blur the image
            #dirname = 'C:\\Users\\Nitz Mistry\\Pictures\\Test_9_7\\'
            #cv2.imwrite(dirname+parking_space_loc+'blur.bmp', blur)
                    
            number_of_white = cannyedgedetection(blur) # we pass the blurred image and get the edges detected. Edges are white color pixels, we count the total white pixels
            #print(number_of_white)

            white_diff_from_empty = number_of_white - spot_data['spots'][int(parking_spot['label'])-1][1] # white pixel difference between current parking spot state and empty parking spot state 
            #print(white_diff_from_empty)
            
            if white_diff_from_empty > white_diff_threshold: # if the difference exceeds threshold of 10, means the parking is FULL
                spot_colors.append([p1,p2,p3,p4, red_color]) # appends the coordinate and color
                print(parking_space_loc + ": FULL")
                #f.writelines([parking_space_loc + ": FULL"])
                #f.write('\n')
                data['spots'].append({ # appends the result to the dictonary
                    'spotnumber': str(parking_spot['label']),
                    'status': 'FULL'
                })
                           
            else:
                spot_colors.append([p1,p2,p3,p4, green_color]) # appends the coordinate and color
                print(parking_space_loc + ": EMPTY")
                #f.writelines([parking_space_loc + ": EMPTY"])
                #f.write('\n')
                data['spots'].append({ # appends the result to the dictonary
                    'spotnumber': str(parking_spot['label']),
                    'status': 'EMPTY'
                })
            
            for spot in spot_colors: # iterates between parking spots
                image = drawBoundBox(image, spot[0], spot[1], spot[3], spot[2], spot[4]) # draws the bounding box with the appropriate color
            
            req = requests.get('https://d4f5230c.ngrok.io/spot-data?data='+json.dumps(data,sort_keys=True)) # converts the dictionary into json and sends to the server
            print(req)
            
            
            cv2.imwrite(dirname+'image_'+cur_time+'.jpg', image) # save the drawed image to the local folder
            
            files = {'file':open(dirname+'image_'+cur_time+'.jpg','rb')} # retrives the image
            req = requests.post('http://d4f5230c.ngrok.io/uploadImage',files=files) # sends the images to the server
            print(req)
            
            vid.release() # releases the camera object

if __name__ == "__main__":
    """This is the main function which will call processing function at every 40 seconds"""
    schedule.every(0.40).minutes.do(processing)
    while 1:
        schedule.run_pending()
        time.sleep(1)