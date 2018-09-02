import cv2
import numpy as np


def adjust_gamma(image, gamma=0.5): #0.5 original
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def capture_frame():
    cap = cv2.VideoCapture(0)

    # Capture frame
    for i in range(0, 5):
        ret, frame = cap.read()

    cv2.imwrite('Foto.jpg', frame)

    # When everything done, release the capture
    cap.release()


def contours_method():
    # definindo as imagens de entrada, cortando a area de interesse e cortando a area restante
    image = cv2.imread('Foto.jpg')
    image = adjust_gamma(image)
    sub_img = image[0:480, 74:640]  # Cortar apenas a area: 250:640, 130:480 ('Foto.png')
    sum_img = image[0:480, 0:74]
    cv2.imwrite('Cropped.jpg', sub_img)

    # definindo a mascara de cor preta para ser concatenada a imagem final
    lower_black = np.array([0, 0, 0], dtype="uint16")
    upper_black = np.array([0, 0, 0], dtype="uint16")
    black_mask = cv2.inRange(sum_img, lower_black, upper_black)
    # cv2.imwrite("SumImg.jpg", black_mask)

    # setando o filtro blurred para melhor precisao e menos ruido e passando a imagem para HSV
    blurred = cv2.pyrMeanShiftFiltering(sub_img, 25, 90)  # 25,90
    cv2.imwrite('Blurred.jpg', blurred)
    hsv_im = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    cv2.imwrite('HSV.jpg', hsv_im)

    # filtrando o vermelho na imagem e salvando
    red_lower = np.array([0,100,100], dtype="uint16")
    red_upper = np.array([10,255,255], dtype="uint16")
    red_range = cv2.inRange(hsv_im, red_lower, red_upper, image)
    cv2.imwrite('RedRange.jpg', red_range)

    #Filtrando a cor preta
    black_lower = np.array([0,0,0], dtype = "uint16")
    black_upper = np.array([100,255,30], dtype = "uint16")
    black_range = cv2.inRange(hsv_im, black_lower, black_upper, image)
    cv2.imwrite("BlackRange.jpg", black_range)
    final_black = cv2.hconcat([black_mask, black_range])
    _, contours_black, hierarchy_black = cv2.findContours(final_black, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print("Numero de contours pretos %d" % len(contours_black))
    cv2.drawContours(image, contours_black, -1, (255, 255, 255), 4)

    #Filtrando a cor azul
    blue_lower = np.array([100, 150, 0], dtype = "uint16")
    blue_upper = np.array([140, 255, 255], dtype = "uint16")
    blue_range = cv2.inRange(hsv_im, blue_lower, blue_upper, image)
    cv2.imwrite("BlueRange.jpg", blue_range)
    final_blue = cv2.hconcat([black_mask, blue_range])
    _, contours_blue, hierarchy_blue = cv2.findContours(final_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print("Numero de contours azuis %d" %len(contours_blue))
    cv2.drawContours(image, contours_blue, -1, (0, 0, 255), 4)

    # concatenando as imagens para que o desenho saia na posicao correta
    finalHSV = cv2.hconcat([black_mask, red_range])
    # cv2.imwrite('FinalP.jpg', finalHSV)

    # definindo o limite e procurando pelos contornos na imagem cortada
    ret, threshold = cv2.threshold(finalHSV, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print ("Numero de contours vermelhos %d" % len(contours))

    # pegando a posicao correta do centro do circulo

    index = 0
    while (index < len(contours)):
        cnt = contours[index]
        try:
            moments = cv2.moments(cnt)
            cy = int(moments['m10'] / moments['m00'])
            cx = int(moments['m01'] / moments['m00'])
            break
        except:
            index = index + 1

    # convertendo a coordenada em pixels para pixels/cm
    # 342 pixels/cm^2 mais ou menos -> 18 pixels/cm
    x = -1 * (cx / 18) + 23
    y = -1 * (cy / 18) + 15
    print ("Coordenada x do centro: %d" % x)
    print ("Coordenada y do centro: %d" % y)

    # desenhando os contornos encontrados e salvabndo a imagem final
    cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
    cv2.imwrite('Resultado.jpg', image)

    return [x, y]


contours_method()