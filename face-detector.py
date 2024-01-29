import cv2

imagePath = 'sample_data/human.jpg'

img = cv2.imread(imagePath)

if img is None:
    print("Error: Unable to load image from path:", imagePath)
else:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(face) > 0:
        print("Yes")
    else:
        print("No")