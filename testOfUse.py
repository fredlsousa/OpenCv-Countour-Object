import cv2
import numpy as np
from rbClass import Piece


image = cv2.imread("Foto.jpg")

redLower = np.array([0,100,100], dtype="uint16")
redUpper = np.array([10,255,255], dtype="uint16")
redFlag = 'r'
redPiece = Piece(image, redUpper, redLower, redFlag)

blackLower = np.array([0, 0, 0], dtype = "uint16")
blackUpper = np.array([230, 255, 65], dtype = "uint16")
blackFlag = 'b'
blackPiece = Piece(image, blackUpper, blackLower, blackFlag)


redPiece.cropImage(1)
redPiece.adjustGamma(0.7, 1) #0.5 original -> less gamma means darker image, otherwise, clearer image
redPiece.turnSumBlack(1)
redPiece.filterImage(1)
redPiece.filterHSV(1)
redPiece.filterColorInHSV(1)
redPiece.findColorContours()
redPiece.drawContoursInImage()
x, y = redPiece.returnCalibratedPositionInSpace()
print("Main: %d %d" %(x, y))

blackPiece.cropImage(1)
blackPiece.adjustGamma(0.7, 1) #0.5 original -> less gamma means darker image, otherwise, clearer image
blackPiece.turnSumBlack(1)
blackPiece.filterImage(1)
blackPiece.filterHSV(1)
blackPiece.filterColorInHSV(1)
blackPiece.findColorContours()
blackPiece.drawContoursInImage()
xb, yb = blackPiece.returnCalibratedPositionInSpace()
print("Main: %d %d" %(xb, yb))
