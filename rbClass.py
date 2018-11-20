import cv2
import numpy as np

class Piece:
    def __init__(self, image, colorUpper, colorLower, flag):
        self.image = image
        self.procImage = image
	self.colorUpper = colorUpper
	self.colorLower = colorLower
	self.flag = flag
        self.sumImage = self.image[0:480, 0:302]
        self.contourArray = np.array([])
        self.pixelPX = 0
        self.pixelPY = 0
        self.spacePX = 0
        self.spacePY = 0

    def cropImage(self, save):   
        self.sumImage[:] = (255,255,255)
        self.procImage = self.image[0:480, 302:640]
        if save:
            cv2.imwrite("Cropped.jpg", self.procImage)
            cv2.imwrite("BlackSum.jpg", self.sumImage)


    def adjustGamma(self, gamma, save):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma= 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        self.procImage = cv2.LUT(self.procImage, table)
        if save:
            cv2.imwrite("GammaApp.jpg", self.procImage)	


    def turnSumBlack(self, save):
        lower_black = np.array([0, 0, 0], dtype="uint16")
        upper_black = np.array([0, 0, 0], dtype="uint16")
        self.sumImage = cv2.inRange(self.sumImage, np.array([0,0,0]), np.array([0,0,0]))
        if save:
            cv2.imwrite("SumImg.jpg", self.sumImage)


    def filterImage(self, save):
        self.procImage = cv2.pyrMeanShiftFiltering(self.procImage, 25, 90)
        if save:
            cv2.imwrite("Blurred.jpg", self.procImage)

    def filterHSV(self, save):
        self.procImage = cv2.cvtColor(self.procImage, cv2.COLOR_BGR2HSV)
        if save:
            cv2.imwrite("HSV.jpg", self.procImage)
    
    def filterColorInHSV(self, save):
        self.procImage = cv2.inRange(self.procImage, self.colorLower, self.colorUpper, self.image)
        if save:
            cv2.imwrite('ColorRange.jpg', self.procImage)

    def findColorContours(self):
        self.procImage = np.hstack([self.sumImage, self.procImage])
        _, self.contourArray, hierarchy = cv2.findContours(self.procImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print ("Number of contours found in %c color: %d" %(self.flag, len(self.contourArray)))

    def drawContoursInImage(self):
        cv2.drawContours(self.image, self.contourArray, -1, (0, 255,0), 4)
        cv2.imwrite("FoundContours.jpg", self.image)

    def contourPositionInPixels(self):
        index = 0
        while (index < len(self.contourArray)):
            cnt = self.contourArray[index]
            try:
                moments = cv2.moments(cnt)
                self.pixelPY = int(moments['m10'] / moments['m00'])
                self.pixelPX = int(moments['m01'] / moments['m00'])
                break
            except:
                index = index + 1

    def returnCalibratedPositionInSpace(self):
        self.contourPositionInPixels()
        if len(self.contourArray) > 0:
            self.spacePX = -1* (self.pixelPX/18) + 23
            self.spacePY = -1 * (self.pixelPY/18) + 15
            print ("Coordinate x of the contour center (Color %c): %d" %(self.flag, self.spacePX))
            print ("Coordinate y of the contour center (Color %c): %d" %(self.flag, self.spacePY))
            return [self.spacePX, self.spacePY]
        else:
            print("No Contours Found in the Image")
        
