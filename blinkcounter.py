import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture('vvid.mp4')

if not cap.isOpened():
    print("Error: Video file not found or cannot be opened")
    exit()

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50])

idList = [22, 23, 24, 25, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    
    if not success or img is None:
        print("Error: Frame not captured")
        continue  # Skip processing if the frame isn't valid

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        leftUp, leftDown = face[159], face[23]
        leftEyel, LeftEyer = face[130], face[243]

        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftEyel, LeftEyer)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftEyel, LeftEyer, (0, 200, 0), 3)

        ratio = (lengthVer / lengthHor) * 100
        ratioList.append(ratio)

        if len(ratioList) > 3:
            ratioList.pop(0)

        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            color = (0, 200, 0)
            counter = 1

        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

        imgPlot = plotY.update(ratioAvg)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 1, 1)

    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 1, 1)

    cv2.imshow("Image Stack", imgStack)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
