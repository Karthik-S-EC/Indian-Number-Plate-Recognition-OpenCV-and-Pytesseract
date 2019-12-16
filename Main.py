import cv2
import os
import pytesseract

import DetectChars
import DetectPlates

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = True

def main(img="car.jpg"):

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()        

    if blnKNNTrainingSuccessful == False:                             
        print ("\nerror: KNN traning was not successful\n")            
        return                                                      
    # end if

    imgOriginalScene  = cv2.imread(img)               

    if imgOriginalScene is None:    
        print ("\nerror: image not read from file \n\n")      
        os.system("pause")                                 
        return                                       

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)          
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        

    cv2.imshow("imgOriginalScene", imgOriginalScene)           

    if len(listOfPossiblePlates) == 0:                         
        print ("\nno license plates were detected\n")            
    else:                                                     
               
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        licPlate = listOfPossiblePlates[0]
        cv2.imshow("imgPlate", licPlate.imgPlate)
        cv2.imwrite("img_plate.jpg",licPlate.imgPlate)
        cv2.imshow("imgThresh", licPlate.imgThresh)
        cv2.imwrite("img_thresh.jpg",licPlate.imgThresh)
        cv2.imwrite("Licence_plate.jpg",licPlate.imgPlate)

        if len(licPlate.strChars) == 0:                    
            print ("\nno characters were detected\n\n")      
            return                                         

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)           

        #print ("\nlicense plate read from image = " + licPlate.strChars + "\n")     
        #print ("----------------------------------------")
        cv2.imwrite('img.jpg',licPlate.imgPlate)
        text_img = pytesseract.image_to_string('img.jpg',lang='eng')
        print("\nLicense Plate read from image is: ",pytesseract.image_to_string('img.jpg',lang='eng'))
         
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return text_img

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)           
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

if __name__ == "__main__":
    main()
