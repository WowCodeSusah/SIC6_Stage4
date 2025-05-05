import cv2 as cv

ESP32_URL = ["http://192.168.1.26:81/stream", "http://192.168.1.27:81/stream"]

def getFrame(ESP32_URL):
    frames = []

    for ip in ESP32_URL:
        cap = cv.VideoCapture(ip)
        if not cap.isOpened():
            print("Failed to connect to camera stream")
            exit()

        # capture frame by frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to read a frame from the stream")
            exit()

        cv.imwrite(ip.replace("http://", "").replace("/strean", "") + "_img", frame)
        frames.append(frame)

        cap.release()
        cv.destroyAllWindows()

    return frames