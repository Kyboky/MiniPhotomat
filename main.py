import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import model_from_json, Sequential
from cv2 import cv2

class Solver:
    def operator_solver(self, equation):
        operators = ("/", "*", "+", "-")
        numbers_list = []
        operators_list = []
        i = 0
        num = ""
        for i in range(len(equation)):
            if equation[i] in operators:
                operators_list.append(equation[i])
                if num != "":
                    numbers_list.append(float(num))
                num = ""
            else:
                num += equation[i]
            i += 1
        if num != "":
            numbers_list.append(float(num))
        count = 0
        while count < len(operators_list):
            if operators_list[count] == "/" or operators_list[count] == "*":
                current_op = operators_list.pop(count)
                if current_op == "/":
                    a = numbers_list.pop(count)
                    b = numbers_list.pop(count)
                    numbers_list.insert(count, (a / b))
                    continue
                if current_op == "*":
                    a = numbers_list.pop(count)
                    b = numbers_list.pop(count)
                    numbers_list.insert(count, (a * b))
                    continue
            count += 1
        count = 0
        while count < len(operators_list):
            if operators_list[count] == "+" or operators_list[count] == "-":
                current_op = operators_list.pop(count)
                if current_op == "+":
                    a = numbers_list.pop(count)
                    b = numbers_list.pop(count)
                    numbers_list.insert(count, (a + b))
                    continue
                if current_op == "-":
                    a = numbers_list.pop(count)
                    b = numbers_list.pop(count)
                    numbers_list.insert(count, (a - b))
                    continue
            count += 1
        return numbers_list[0]

    def solver(self, equation):
        close_pos = equation.find(")")
        while close_pos != -1:
            open_pos = equation.rfind("(", 0, close_pos)
            before_br = equation[0:open_pos]
            after_br = equation[close_pos + 1:]
            value = self.operator_solver(equation[open_pos + 1:close_pos])
            equation = before_br + str(value) + after_br
            close_pos = equation.find(")")
        return self.operator_solver(equation)

class Character:
    def __init__(self,x,y,img, value):
        self.posx = x
        self.posy = y
        self.img = img
        self.value = value

    def getImg(self):
        return self.img
    def get_value(self):
        return self.value
    def get_position(self, x_offset, y_offset):
        return (int(self.posx + x_offset), int(self.posy + y_offset))

class CNNModel:

    def __init__(self):
        self.str_value = ['0', '1', '2', '3','4','5','6','7','8','9','+','/','(','*',')','-']
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        self.model = loaded_model
        print("Loaded model from disk")
    def predict(self,x):
        if len(char_img.shape):
            x = x.reshape(1,32,32)
        return self.str_value[self.model.predict(x).argmax()]

def preprocess_image(img, resize = 32, min_size = 60, padding = 4):
    height, width = img.shape
    if height > min_size or width > min_size:
        if height < width:
            new_img = np.zeros((width+2*padding,width+2*padding))
            offset = int((width-height)/2+padding)
            # print(offset)
            new_img[ offset:offset+height, padding:padding+width] = img
        else:
            new_img = np.zeros((height+2*padding,height+2*padding))
            offset = int((height-width)/2+padding)
            # print(offset)
            new_img[ padding:padding+height, offset:offset+width] = img
    else:
        new_img = np.zeros((min_size + 2 * padding, min_size + 2 * padding))
        offset_x = int((min_size + 2 * padding - width)/2)
        offset_y = int((min_size + 2 * padding - height)/2)
        new_img[offset_y:offset_y+height, offset_x:offset_x+width] = img
    cv2.threshold(new_img, 200, 255, cv2.THRESH_BINARY, dst=new_img)[1]
    new_img = cv2.resize(new_img,(resize,resize))
    new_img = np.array(new_img,dtype=np.uint8)
    return new_img

test_img = "20211217_003053.jpg"
solve = Solver()
kernel = np.ones((3,3),np.uint8)
f = cv2.imread(test_img)
# (B,G,R) = cv2.split(f)
f = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
# f = f[240:1222, 151:832]
# R = np.array(R,dtype=np.uint8)
# binary = cv2.threshold(R, 100, 255, cv2.THRESH_BINARY_INV)[1]
binary = cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
cv2.GaussianBlur(binary,(5,5),1,dst=binary)
cv2.threshold(binary, 20, 255, cv2.THRESH_BINARY,dst=binary)[1]
cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,dst=binary,iterations=2)
cv2.imshow("Slika",binary)
cv2.waitKey(0)
# binary = cv2.rotate(binary,cv2.ROTATE_90_COUNTERCLOCKWISE)
ret, labels = cv2.connectedComponents(binary,connectivity=4)
list_of_chars = []
f = CNNModel()

for i in range(labels.max()):
    label_mask = np.where(labels==(i+1),255,0)
    label_mask = np.array(label_mask,dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(label_mask)
    if (w<30 and h<30):
        continue
    char_img = preprocess_image(label_mask[y:y+h, x:x+w])
    char = Character(x+w/2, y+h/2, char_img, f.predict(char_img))
    list_of_chars.append(char)

f = cv2.imread(test_img)
font = cv2.FONT_HERSHEY_SIMPLEX
# Debug plot

multiple_equations = False
if multiple_equations:
    list_of_chars.sort(key=lambda y: y.posy)
    last_posy = list_of_chars[0].posy
    curr = []
    for i in list_of_chars:
        if abs(last_posy - i.posy) > 30:
            curr.sort(key=lambda y: y.posx)
            equation = ''.join([i.get_value() for i in curr])
            equation_result = solve.solver(equation)
            cv2.putText(f, equation + "= " + str(equation_result), curr[0].get_position(50,-50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            last_posy = i.posy
            curr = [i]
        else:
            curr.append(i)
            last_posy = i.posy
    curr.sort(key=lambda y: y.posx)
    equation = ''.join([i.get_value() for i in curr])
    equation_result = solve.solver(equation)
    cv2.putText(f, equation + "= " + str(equation_result), curr[0].get_position(50, -50), font, 1, (0, 0, 255), 2,cv2.LINE_AA)

else:
    list_of_chars.sort(key=lambda x: x.posx)
    equation = ''.join([i.get_value() for i in list_of_chars])
    print(equation)
    equation_result = solve.solver(equation)
    cv2.putText(f, equation + "= " + str(equation_result), list_of_chars[0].get_position(0, -50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


# for char in list_of_chars:
#     cv2.putText(f,char.value,char.get_position(0, -50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("Slika",f)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for char in list_of_chars:
#     print(char.value)
#     cv2.imshow("Slika",char.img)
#     cv2.waitKey(0)
