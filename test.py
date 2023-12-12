# import os
# from ultralytics import YOLO
# import cv2
# # Load your image
# image = cv2.imread("C:\\Users\\badli\\OneDrive\\Desktop\\proiect dari\\images\\da2.jpeg")
#
# # Resize the image to (480, 640)
# resized_image = cv2.resize(image, (640, 640))
#
# # Now you can use the resized_image for inference
# IMAGES_DIR = os.path.join('.', 'images')
#
# image_path = os.path.join(IMAGES_DIR, 'da.jpg')
# output_path = 'output.jpg'
#
# image = cv2.imread(image_path)
#
# if image is not None:
#     H, W, _ = image.shape
#
#     model_path = "best.pt"
#     model = YOLO(model_path)  # load a custom model
#
#     threshold = 0.5
#
#     results = model(image)[0]
#
#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result
#
#         if score > threshold:
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
#
#     cv2.imwrite(output_path, image)
#     cv2.imshow('Result', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Error reading image.")
import os
import cv2
from ultralytics import YOLO

model_path = "best.pt"

model = YOLO(model_path)

threshold = 0.5

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()

    if not ret:
        print("Eroare la capturarea de la camera.")
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
