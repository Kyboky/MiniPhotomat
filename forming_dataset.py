from cv2 import cv2
import numpy as np
import os

def preprocess_image(img, resize = 32, min_size = 60, padding = 4):
    height, width = img.shape
    if height > min_size or width > min_size:
        if height < width:
            new_img = np.zeros((width+2*padding,width+2*padding))
            offset = int((width-height)/2+padding)
            print(offset)
            new_img[ offset:offset+height, padding:padding+width] = img
        else:
            new_img = np.zeros((height+2*padding,height+2*padding))
            offset = int((height-width)/2+padding)
            print(offset)
            new_img[ padding:padding+height, offset:offset+width] = img
    else:
        new_img = np.zeros((min_size + 2 * padding, min_size + 2 * padding))
        offset_x = int((min_size + 2 * padding - width)/2)
        offset_y = int((min_size + 2 * padding - height)/2)
        new_img[offset_y:offset_y+height, offset_x:offset_x+width] = img
    cv2.threshold(new_img, 200, 255, cv2.THRESH_BINARY, dst=new_img)[1]
    new_img = cv2.resize(new_img,(resize,resize))
    return new_img

# Testing purposes
# img = cv2.imread(r"D:\Projects\Photomat\dataset\mul\36.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("bok", preprocess_image(img,resize = 20))
# cv2.waitKey(0)

def initialize_folders():
    if not os.path.isdir("dataset"): os.mkdir("dataset")
    if not os.path.isdir("dataset/0"): os.mkdir("dataset/0")
    if not os.path.isdir("dataset/1"): os.mkdir("dataset/1")
    if not os.path.isdir("dataset/2"): os.mkdir("dataset/2")
    if not os.path.isdir("dataset/3"): os.mkdir("dataset/3")
    if not os.path.isdir("dataset/4"): os.mkdir("dataset/4")
    if not os.path.isdir("dataset/5"): os.mkdir("dataset/5")
    if not os.path.isdir("dataset/6"): os.mkdir("dataset/6")
    if not os.path.isdir("dataset/7"): os.mkdir("dataset/7")
    if not os.path.isdir("dataset/8"): os.mkdir("dataset/8")
    if not os.path.isdir("dataset/9"): os.mkdir("dataset/9")
    if not os.path.isdir("dataset/add"): os.mkdir("dataset/add")
    if not os.path.isdir("dataset/sub"): os.mkdir("dataset/sub")
    if not os.path.isdir("dataset/div"): os.mkdir("dataset/div")
    if not os.path.isdir("dataset/mul"): os.mkdir("dataset/mul")
    if not os.path.isdir("dataset/right_bracket"): os.mkdir("dataset/right_bracket")
    if not os.path.isdir("dataset/left_bracket"): os.mkdir("dataset/left_bracket")
initialize_folders()

img_string = "mama_2"
prefix = img_string + '_'
cv2.namedWindow("Slika")
kernel = np.ones((3,3),np.uint8)
f = cv2.imread("learning_pictures/" + img_string + ".jpg")
(B,G,R) = cv2.split(f)

cv2.imshow("Slika",B)
cv2.waitKey(0)
f = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(B, 100, 255, cv2.THRESH_BINARY_INV)[1]
cv2.GaussianBlur(binary,(3,3),1,dst=binary)
cv2.threshold(binary, 70, 255, cv2.THRESH_BINARY,dst=binary)[1]
cv2.imshow("Slika",binary)
cv2.waitKey(0)
#binary = cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 30)

# cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,dst=binary,iterations=2)
# binary = cv2.rotate(binary,cv2.ROTATE_90_COUNTERCLOCKWISE)

ret, labels = cv2.connectedComponents(binary,connectivity=4)
count = 0
if True:
    for i in range(labels.max()):
        label_mask = np.where(labels==(i+1),255,0)
        label_mask = np.array(label_mask,dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(label_mask)
        if (w<10 and h<10):
            continue
        char = label_mask[y:y+h, x:x+w]

        cv2.imshow("Slika", char)
        cv2.waitKey(1)
        clasifier = input("koji je ovo znak")
        if clasifier=='0':
            cv2.imwrite("dataset/0/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='1':
            cv2.imwrite("dataset/1/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier == '2':
            cv2.imwrite("dataset/2/" + prefix + str(count) + '.jpg', preprocess_image(char))
            count += 1
        elif clasifier=='3':
            cv2.imwrite("dataset/3/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='4':
            cv2.imwrite("dataset/4/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='5':
            cv2.imwrite("dataset/5/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='6':
            cv2.imwrite("dataset/6/" + prefix + str(count) +'.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='7':
            cv2.imwrite("dataset/7/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='8':
            cv2.imwrite("dataset/8/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='9':
            cv2.imwrite("dataset/9/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='+':
            cv2.imwrite("dataset/add/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='/':
            cv2.imwrite("dataset/div/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='-':
            cv2.imwrite("dataset/sub/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='*':
            cv2.imwrite("dataset/mul/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier=='(':
            cv2.imwrite("dataset/left_bracket/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        elif clasifier==')':
            cv2.imwrite("dataset/right_bracket/" + prefix + str(count) + '.jpg',preprocess_image(char))
            count += 1
        else:
            continue
        cv2.destroyAllWindows()
binary = cv2.resize(binary,(1900,1100))
cv2.imshow("Slika",binary)
cv2.waitKey(0)
cv2.destroyAllWindows()