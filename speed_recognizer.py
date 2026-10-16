import cv2
import numpy as np
import easyocr


READER = easyocr.Reader(['en'], gpu=True, model_storage_directory='./vendor/easyocr', user_network_directory='./vendor/easyocr')
IMAGE_PATH = "test_images/test_2.png"
VIDEO_PATH = "videoplayback (1).mp4"

def detect_speed(frame):
    full_frame = frame.copy()
    h, w = frame.shape[:2]
    frame = frame[3*h//5:h, w//4:w*3//4]
    largest_bbox = None
    speed = None
    # DETECT + RECOGNIZE
    results = READER.readtext(frame)
    for (bbox, text, prob) in results:
        if largest_bbox is None or (bbox[2][0] - bbox[0][0]) * (bbox[2][1] - bbox[0][1]) > (largest_bbox[2][0] - largest_bbox[0][0]) * (largest_bbox[2][1] - largest_bbox[0][1]):
            largest_bbox = bbox
            speed = text

    if largest_bbox is not None:
        pts = np.array(largest_bbox, dtype="float32")
        cv2.rectangle(full_frame, (int(pts[0][0] + w//4), int(pts[0][1] + 3*h//5)), (int(pts[2][0] + w//4), int(pts[2][1] + 3*h//5)), (0, 255, 0), 2)
        cv2.putText(full_frame, f"Detected Speed: {speed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    full_frame = cv2.resize(full_frame, (960, 540))
    return full_frame, speed

# image = cv2.imread(IMAGE_PATH)

# speed_frame, detected_speed = detect_speed(image)
# cv2.imshow("Speed Detection", speed_frame)


cap = cv2.VideoCapture(VIDEO_PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame, detected_speed = detect_speed(frame)
    cv2.imshow("Speed Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.waitKey(0)

cv2.destroyAllWindows()