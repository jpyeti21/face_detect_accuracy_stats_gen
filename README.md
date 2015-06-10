# face_detect_accuracy_stats_gen
This script is designed to generate a csv file with statistical data for a series of iterations per photo (27 at current defaults). When a large number of photos have been sampled, a correlation may be made between the accuracy values and the facial detection settings (scale factor, minimum neighbors, and minimum size)
This is the first step in a larger project, the ultimate goal of which is to write a program which will auto-rotate photos from a large collection based on facial detection, other human feature detection, or scene.
The core of this program is OpenCV, and is written in Python.
This program takes one or more images as input. There are three settings that OpenCV's faceCascade.detectMultiScale function uses to determine when it detects a face or not
These three settings are each given three values (low, medium, and high values). The program iterates over all three settings for the three values each (27 iterations for a single photo)
A dialogue pops up at the start of each photo's iterations asking how many faces are in the photo (user enters number of faces the computer should be able to detect using that particular haar cascade)
Another dialog pops up for each of the 27 iterations asking how many False Positives (how many faces did OpenCV identify that are NOT faces)
The program then automatically calculates how many False Negatives there were (detectable faces that it did not detect)
All this data is written to a CSV, including an accuracy value (based on the False Positives, False Negatives, and actual faces)
Once data for a large number of images has been collected, it can be statistically analyzed to determine which combination of settings is most accurate for a given image resolution
Results should be statistically broken down further, but I'm still working on that...
