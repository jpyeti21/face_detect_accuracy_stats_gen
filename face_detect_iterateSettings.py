# Script Name:  Object Detection Settings Accuracy Statistics Generator
# Written by: Joshua Prescott
# Date: April-June '15
"""This script is designed to generate a csv file with statistical data for a series of iterations per photo (27 at current defaults). When a large number of photos have been sampled, a correlation may be made between the accuracy values and the facial detection settings (scale factor, minimum neighbors, and minimum size)"""
# NOTES: the final accuracy value needs to be normalied against the column Actual_Faces

import cv2
import sys
import numpy as np
import os
import csv
import wx
import time
import pandas

def imagetext(text, image, scale):
    """Overlays text onto an image, 1/10 up from bottom of image and centered horizontally.  Inputs are the text string, an image (already rendered in memory), and the scale (as used in cv2.putText)"""
    height, width, depth = image.shape
    resolution = height, width    
    print resolution
    print "image " + str(image)
    imageBL = 0, height - (height // 10)
    print "imageBL " + str(imageBL)
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, scale, 3)
    print "textsize"
    print textsize
    textbox, baseline = textsize
    h,v = textbox
    halfh = h//2
    halfimg = width//2
    startinghoriz = halfimg - halfh
    print "textbox"
    print textbox
    print "baseline"
    print baseline
    imageBL = startinghoriz, height - (height // 10)
    print "imageBL " + str(imageBL)    
    cv2.putText(image, text, imageBL, font, scale, (255,0,0), 3, cv2.CV_AA)
    return image

def calcrng(alist):
    """takes a list of string input, converts to int, sorts and then calculates range"""
    # convert list (str to int)
    intlist = []
    for item in alist:
        intlist.append(int(item))
    ##print intlist
    intlist.sort()
    ##print intlist
    listrange = intlist[-1] - intlist[0]
    ##print listrange
    return listrange    

def dialog(text, default_value):
    """Creates a dialog window for user input, displaying prompt and returning the user input to the script"""
    # initialize wx GUI app
    app = wx.PySimpleApp()    
    # Call input Dialog
    dialog0text = text
    dialog0 = wx.TextEntryDialog(None, dialog0text,"Text Entry", default_value, style=wx.OK|wx.CANCEL)
    if dialog0.ShowModal() == wx.ID_OK:
        print "You entered actual number of faces: %s" % dialog0.GetValue()
    user_input = str(dialog0.GetValue())
    dialog0.Destroy()
    return user_input

# format date time for use in csv name (allows for new csv each time script is run)
datetime_raw = str((time.strftime("%d/%m/%Y"))) + "_" + str((time.strftime("%H:%M:%S")))
print datetime_raw
datetime_raw2 = datetime_raw.replace("/", "-",2)
datetime = datetime_raw2.replace(":", "-", 2)
print datetime

# lets user manually enter paths upon script start, or choose to use hardcoded paths.
manualpaths = dialog("Enter paths manually (y/n)?", "n")
if manualpaths == "y":
    ## PATH TO HAAR CASCADES
    ## Call input Dialog
    dirHaarCascades = dialog("Path to Haar Cascades? ", r"C:\Users\DaddyCB\Documents\GIS\aaaPYTHON_II\FacialDetection\haarcascades")
    
    ## PATH TO PHOTOS FOLDER
    ## Call input Dialog
    photodir = dialog("Path to Photo Folder? ", r"C:\Users\DaddyCB\Documents\GIS\aaaPYTHON_II\FacialDetection\TestPhotos")
    
    ## PATH TO SAVE CSV
    ## Call input Dialog
    csvdir = dialog("Path to save CSV? ", r"C:\Users\DaddyCB\Documents\GIS\aaaPYTHON_II\FacialDetection\csvProduction")
    
    ## NAME OF HAAR CASCADE TO USE
    ## Call input Dialog
    haarname = dialog("Name of Haar Cascade to use? ", r"haarcascade_frontalface_default.xml")

if manualpaths == "n":
    # PATH to haar cascade folder
    dirHaarCascades = r"C:\Users\DaddyCB\Documents\GIS\aaaPYTHON_II\FacialDetection\haarcascades"
    ##dirHaarCascades = HaarCascadePathInput
    
    # PATH to test photos folder
    photodir = r"C:\Users\DaddyCB\Documents\GIS\aaaPYTHON_II\FacialDetection\TestPhotos"
    ##photodir = PhotoFolderPathInput
    
    # PATH to directory in which to save .csv
    csvdir = r"C:\Users\DaddyCB\Documents\GIS\aaaPYTHON_II\FacialDetection\csvProduction"
   
    ##os.chdir(CSVSavePathInput)
    
    # NAME of the haar cascade file to use
    haarname = "haarcascade_frontalface_default.xml"
   
os.chdir(csvdir)
haarnametxt = haarname.split(".", 2)

# make list of photo names
photolist = os.listdir(photodir)
# get rid of thumbnail.db file
newphotolist = []
for names in photolist:
    if names.endswith(".db") == False:
        newphotolist.append(names)
photolist = newphotolist
photolistlen = len(photolist)

#set variable lists for single iteration (TESTING ONLY)
##varscaleFactor = [1.1]
##varminNeighbors = [5]
##varminSize = [(30, 30)]

#setting variables lists
varscaleFactor = [1.05, 1.1, 1.3]
scalelistlen = len(varscaleFactor)
varminNeighbors = [3, 5, 7]
minlistlen = len(varminNeighbors)
varminSize = [(20,20), (24,24), (30, 30)]
sizelistlen = len(varminSize)

# Calculate total number of iterations; for photos in list, and numbers of values to test in variable lists
itertotal = scalelistlen * minlistlen * sizelistlen * photolistlen
print "Total number of iterations will be: " + str(itertotal)

# start csv operations
# NAME of csv
csvname = "csvTest2_" + datetime + "_" + str(haarnametxt[0]) + ".csv"
f = open(csvname, 'wb')
writer = csv.writer(f)

# list of field names
fieldnames = ["image_name", "heighth_width_Cdepth", "simplified_resolution", "scale_set", "neighbor_set", "size_set", "detected_Faces", "Actual_Faces", "FALSE_POS", "FALSE_NEG", "Setting_ID", "accuracy_value"]

# write field names into csv
writer.writerow(fieldnames)

# iterate over images, and 3 different settings, with multiple variables for each setting
count = 0
rangeListFP = []
rangeListFN = []
for photo in photolist:
    
    # set paths needed for cv2 to do its magic
    imagePath = photodir + "\\" + photo
    # Read the image
    original_image = cv2.imread(imagePath)
    # Find image size and color depth
    height, width, depth = original_image.shape
    resolution = height, width, depth
    size = height, width
    # average of height, width to be used as a simplified resolution identifier
    average = np.mean(size)
    averageint = average.astype(np.int64)
    averagestr = str(averageint)
    
    # Resize image
    ## we need to keep in mind aspect ratio so the image does
    ## not look skewed or distorted -- therefore, we calculate
    ## the ratio of the new image to the old image
    # resizing image
    # NEED TO ADDRESS THIS ISSUE!  MUST PERFORM THIS ANALYSIS USING ORIGINAL IMAGE RESOLUTIONS ####
    r = 1000.0 / original_image.shape[1]
    dim = (1000, int(original_image.shape[0] * r))
    ## perform the actual resizing of the image and show it
    resized_image = cv2.resize(original_image, dim, interpolation = cv2.INTER_AREA)        
    
    # create an image window for each iteration and display the image (window named via recordlist)
    cv2.imshow(str(photo) ,resized_image)   
    
    # Get user input; Only on photo iteration; How many faces in photo?
    # Call input Dialog function
    actualfaces = dialog("Enter actual number of detectable faces:", "")
    
    # after user input, destroy image window and move onto setting iterations and facial detection
    cv2.destroyAllWindows()
    
    # iterate over first scalesettings, then neighborsettings and finally sizesettings, using lists containtin value ranges for each
    for scalesetting in varscaleFactor:
        for neighborsetting in varminNeighbors:
            for sizesetting in varminSize:
                count = count + 1
                # set paths needed for cv2 to do its magic
                imagePath = photodir + "\\" + photo
            
                cascPath = dirHaarCascades + "\\" + haarname
                
                # cv2 starts doing its thing
                # Create the haar cascade
                faceCascade = cv2.CascadeClassifier(cascPath)
                
                # Read the image
                original_image = cv2.imread(imagePath)
                
                ## Find image size and color depth
                height, width, depth = original_image.shape
                resolution = height, width, depth
                # print statements for debugging only
                #print resolution
                #print height, width, depth 
                
                ## we need to keep in mind aspect ratio so the image does
                ## not look skewed or distorted -- therefore, we calculate
                ## the ratio of the new image to the old image
                # resizing image
                # NEED TO ADDRESS THIS ISSUE!  MUST PERFORM THIS ANALYSIS USING ORIGINAL IMAGE RESOLUTIONS ####
                r = 1000.0 / original_image.shape[1]
                dim = (1000, int(original_image.shape[0] * r))
                 
                ## perform the actual resizing of the image and show it
                resized_image = cv2.resize(original_image, dim, interpolation = cv2.INTER_AREA)
                
                #convert to grayscale
                gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the image, scale factor default 1.1, minMeighbors 5, minSize 30,30
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor = scalesetting,
                    minNeighbors = neighborsetting,
                    minSize = sizesetting,
                    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                )
                detected_Faces = len(faces)
                # print statements for debugging only
                # print "Found {0} faces!".format(len(faces))
                # print faces
                
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    #drawing rectangles using resized_image image as ratio template
                    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                # create list of items to pass into csv    
                recordlist = [photo, str(resolution), averagestr, str(scalesetting), str(neighborsetting), str(sizesetting[1]), str(detected_Faces)]
                
                iterationstate = "Iteration " + str(count) + " of " + str(itertotal)
                
                # write text onto image
                resized_image = imagetext(iterationstate, resized_image, 3)
                
                # create an image window for each iteration and display the image plus face-rectangles (window named via recordlist)
                cv2.imshow(str(recordlist) ,resized_image)
                
                # TEST PRINT of values prior to writing them to csv
                print str(count) + " " + str(recordlist)                                        
                
                # Get user input, for accuracy of detection for each iteration - need 2 values, false positive and false negatives                          
                # Call input Dialog function (FALSE POSITIVES)
                dialogtext = "How many False Positives? "
                falsepos = dialog(dialogtext, "")
                
                ##dialog1text = "(" + str(count) + " of " + str(itertotal) + ")" + " False Positives: "
                ##dialog1 = wx.TextEntryDialog(None, dialog1text,"Text Entry", "Default Value", style=wx.OK|wx.CANCEL)
                ##if dialog1.ShowModal() == wx.ID_OK:
                    ##print "You entered False Positives: %s" % dialog1.GetValue()
                ##falsepos = str(dialog1.GetValue())
                ##dialog1.Destroy()
                
                # Call input Dialog (FALSE NEGATIVES)
                ##dialog2text = "(" + str(count) + " of " + str(itertotal) + ")" + " False Negatives: "
                ##dialog2 = wx.TextEntryDialog(None, dialog2text,"Text Entry", "Default Value", style=wx.OK|wx.CANCEL)
                ##if dialog2.ShowModal() == wx.ID_OK:
                ##    print "You entered False Negatives: %s" % dialog2.GetValue()
                ##falseneg = str(dialog2.GetValue())
                ##dialog2.Destroy()                        
                
                # Calculate FALSE NEGATIVES
                falseneg = int(detected_Faces) - (int(actualfaces) + int(falsepos))
                if falseneg < 0:
                    falseneg = falseneg * -1
                falseneg = str(falseneg)
                print falseneg
                print type(falseneg)
                # once photo window is drawn, this makes program wait until a key is pressed before moving onto the next iteration
                # commented because the dialog windows provide same result
                #cv2.waitKey(0)
                
                # add current iteration values to range lists
                rangeListFP.append(falsepos)
                rangeListFN.append(falseneg)                     
                
                # adds the user input to the list which will be written to csv
                recordlist.append(actualfaces)
                recordlist.append(falsepos)
                recordlist.append(str(falseneg))
                # adds the SettingID to the list
                recordlist.append(str(count))
                
                #reset counter to 0 after a full iteration set ####THIS NOT WORKING!!!#####
                if count == itertotal:
                    count = 0
                print recordlist
                
                # writes list of values to csv
                writer.writerow(recordlist)
                
                # destroy cascade of windows
                cv2.destroyAllWindows()
# prints final count of iterations                
print count

print "FP list"
print rangeListFP
print calcrng(rangeListFP)
print "FN list"
print rangeListFN
print calcrng(rangeListFN)
rngfalsepos = int(calcrng(rangeListFP))
rngfalseneg = int(calcrng(rangeListFN))
print rngfalsepos
print rngfalseneg
## accuracyvalue = falseneg + (rngfalseneg * (falsepos / rngfalsepos))

# closes csv
f.close()    

# re-open csv, calculate accuracy value and append it to csv rows

with open(csvname, 'rb') as old_csv:
    csv_reader = csv.reader(old_csv)
    with open(photo + "_" + csvname, 'wb') as new_csv:
        csv_writer = csv.writer(new_csv)
        csv_writer.writerow(fieldnames)
        for i, row in enumerate(csv_reader):
            if i != 0:
                row.append(int(row[7])-int(row[9])-int(row[8]))
                csv_writer.writerow(row)    

old_csv.close()
new_csv.close()

dataframe = pandas.DataFrame.from_csv(photo + "_" + csvname)
print dataframe

# Delete old csv
os.remove(csvname)
