import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import base64
import io
import imageio
from flask import Flask, render_template, request
import jyserver.Flask as jsf
import numpy as np
from keras.models import model_from_json, Sequential
from cv2 import cv2
from PIL import Image

class Solver:
    def __init__(self):
        self.operators = ("/", "*", "+", "-")

    def operator_solver(self,pos1 = -1,pos2 = -1):
        if pos1 != -1:
            equation = self.current_equation[pos1:pos2]
        else:
            equation = self.current_equation
        numbers_list = []
        operators_list = []
        if equation[0] == '-':
            is_negative = True
            equation = equation[1:]
        else:
            is_negative = False
        i = 0
        num = ""
        for i in range(len(equation)):
            if equation[i] in self.operators:
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
        if is_negative:
            numbers_list[0] *= (-1)
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

    def adding_mul(self):
        open_bracket_split = self.current_equation.split("(")
        for i in range(len(open_bracket_split) - 1):
            if i == 0 and i == "":
                continue
            if open_bracket_split[i][-1] not in self.operators:
                open_bracket_split[i] += "*"
        self.current_equation = "(".join(open_bracket_split)
        close_bracket_split = self.current_equation.split(")")
        for i in range(1,len(close_bracket_split)-1):
            if close_bracket_split[i] != '' and close_bracket_split[i][0] not in self.operators:
                close_bracket_split[i] = "*" + close_bracket_split[i]
        self.current_equation = ")".join(close_bracket_split)

    def solve(self, equation):
        self.current_equation = equation
        if self.current_equation.find(")") != -1:
            self.adding_mul()
        close_pos = self.current_equation.find(")")
        while close_pos != -1:
            open_pos = self.current_equation.rfind("(",0, close_pos)
            before_br = self.current_equation[0:open_pos]
            after_br = self.current_equation[close_pos + 1:]
            value = self.operator_solver(open_pos + 1 , close_pos)
            self.current_equation = before_br + str(value) + after_br
            close_pos = self.current_equation.find(")")
        return self.operator_solver()

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
    def predict(self,char_img):
        if len(char_img.shape) != 3:
            char_img = char_img.reshape(1,32,32)
        return self.str_value[self.model.predict(char_img).argmax()]

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

def solve_graphical_equation(img):
    # f = cv2.imread(test_img)
    f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binary = cv2.threshold(R, 100, 255, cv2.THRESH_BINARY_INV)[1]
    binary = cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    cv2.GaussianBlur(binary, (5, 5), 1, dst=binary)
    cv2.threshold(binary, 10, 255, cv2.THRESH_BINARY, dst=binary)[1]
    kernel = np.ones((3, 3), np.uint8)
    cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, dst=binary, iterations=2)
    cv2.imshow("Slika", binary)
    cv2.waitKey(0)
    ret, labels = cv2.connectedComponents(binary, connectivity=4)
    list_of_chars = []

    for i in range(labels.max()):
        label_mask = np.where(labels == (i + 1), 255, 0)
        label_mask = np.array(label_mask, dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(label_mask)
        if (w < 20 and h < 20):
            continue
        char_img = preprocess_image(label_mask[y:y + h, x:x + w])
        char = Character(x + w / 2, y + h / 2, char_img, cnn_model.predict(char_img))
        list_of_chars.append(char)
    f = cv2.imread(test_img)
    font = cv2.FONT_HERSHEY_SIMPLEX

    multiple_equations = False
    if multiple_equations:
        list_of_chars.sort(key=lambda y: y.posy)
        last_posy = list_of_chars[0].posy
        curr = []
        for i in list_of_chars:
            if abs(last_posy - i.posy) > 30:
                curr.sort(key=lambda y: y.posx)
                equation = ''.join([i.get_value() for i in curr])
                equation_result = solver.solve(equation)
                cv2.putText(f, equation + "= " + str(equation_result), curr[0].get_position(50, -50), font, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                last_posy = i.posy
                curr = [i]
            else:
                curr.append(i)
                last_posy = i.posy
        curr.sort(key=lambda y: y.posx)
        equation = ''.join([i.get_value() for i in curr])
        equation_result = solver.solve(equation)
        cv2.putText(f, equation + "= " + str(equation_result), curr[0].get_position(50, -50), font, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    else:
        list_of_chars.sort(key=lambda x: x.posx)
        equation = ''.join([i.get_value() for i in list_of_chars])
        print(equation)
        equation_result = solver.solve(equation)
        cv2.putText(f, equation + "= " + str(equation_result), list_of_chars[0].get_position(0, -50), font, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Slika", f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


capture=0
solver = Solver()
cnn_model = CNNModel()
camera = cv2.VideoCapture(0)
app = Flask(__name__,template_folder="./")

@jsf.use(app)
class App:
    def __init__(self):
        self.count=0
        self.eq = ""
        self.solution = None
    def solve(self, debug):
        print("hi")
        image = base64.b64decode(debug[22:])
        print(image)
        img = Image.open(io.BytesIO(image)).convert('RGB')
        open_cv_image = np.array(img)
        final_img = open_cv_image[:, :, ::-1].copy()
        cv2.imshow("hi",final_img)
        cv2.waitKey(0)
        print(final_img.shape)
        self.count += 1
        self.js.document.getElementById("count").innerHTML = self.count


@app.route('/')
def index():
    return App.render(render_template('index.html'))

# @app.route('/image/webp',methods=['GET'])
# def image():
#     return App.render(render_template('index.html'))

test_img = "zadatak_1.jpg"

if __name__ == '__main__':
    # solve_graphical_equation(cv2.imread(test_img))
    app.run(host="0.0.0.0",port="5000",ssl_context='adhoc')
    #socketio.run(app = app, host="0.0.0.0")


# for char in list_of_chars:
#     print(char.value)
#     cv2.imshow("Slika",char.img)
#     cv2.waitKey(0)
