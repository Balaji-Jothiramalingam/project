### EX 12: Project - Object Detection
## Aim
To write a python program using OpenCV to do the following image manipulations. i) Extract ROI from an image. ii) Perform handwritting detection in an image. iii) Perform object detection with label in an image.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
## Step1:
Import necessary packages

## Step2:
Read the image and convert the image into RGB

## Step3:
Display the image

## Step4:
Set the pixels to display the ROI

## Step5:
Perform bit wise conjunction of the two arrays using bitwise_and

## Step6:
Display the segmented ROI from an image.

## II)Perform handwritting detection in an image
## Step1:
Import necessary packages

## Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.

## Step3:
Display the results.

III)Perform object detection with label in an image
## Step1:
Import necessary packages

## Step2:
Set and add the config_file,weights to ur folder.

## Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)

## Step4:
Create a classLabel and print the same

## Step5:
Display the image using imshow()

## Step6:
Set the model and Threshold to 0.5

## Step7:
Flatten the index,confidence.

## Step8:
Display the result.

## #Developed By: BALAJI J
## Reg No: 212221243001
# Perform ROI from an image:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'naturee.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([22, 93, 0])
upper_yellow = np.array([45, 255, 255])
mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
segmented_image = cv2.bitwise_and(img, img, mask=mask)
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
plt.imshow(segmented_image_rgb)
plt.title('Segmented Image (Yellow)')
plt.axis('off')
plt.show()
```
## II) Perform handwritting detection in an image:
```
get_ipython().system('pip install opencv-python numpy matplotlib')
import cv2
import numpy as np
import matplotlib.pyplot as plt
def detect_handwriting(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
image_path = 'hand1.jpg'
detect_handwriting(image_path)
III) Perform object detection with label in an image:
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

model=cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))
img=cv2.imread('bmw.png')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2=127.5
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (0, 0, 255), 2)
    text_position = (boxes[0] + 10, boxes[1] + 40)
    cv2.putText(img, classLabels[ClassInd - 1], (text_position[0] + 2, text_position[1] + 2), 
                font, fontScale=font_scale, color=(0, 0, 0), thickness=3) 
    cv2.putText(img, classLabels[ClassInd - 1], text_position, 
                font, fontScale=font_scale, color=(255, 0, 0), thickness=2) 
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
```
## Output::
## I) Perform ROI from an image:

![386212515-528f7bd7-7ab4-4695-a6d3-2831ea6a0fdc](https://github.com/user-attachments/assets/164f1fdf-088e-4a63-841e-655e0f8a0279)



![386212596-86a178b1-8c84-48b9-bf74-9f9731ecb324](https://github.com/user-attachments/assets/2fb0d232-d89e-4957-a281-0c53b0dda092)

## Perform handwritting detection in an image:

![386212920-72140c19-bc78-44dd-8371-ae36d8af057f](https://github.com/user-attachments/assets/083ddb57-9b6a-4fe4-b8f6-d3f937b3b7fc)

## III) Perform object detection with label in an image:


![386213112-43ea818c-b175-40cf-9561-4fcca3f67d46](https://github.com/user-attachments/assets/4949795d-fe5a-4fa9-8128-e51419d57c3e)


## Result:
Thus, a python program using OpenCV for following image manipulations is done successfully.

