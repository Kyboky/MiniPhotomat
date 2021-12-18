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
        last_operator = False
        for i in range(len(equation)):
            if equation[i] in self.operators:
                if last_operator:
                    num = equation[i]
                    last_operator = False
                    continue
                last_operator = True
                operators_list.append(equation[i])
                if num != "":
                    numbers_list.append(float(num))
                num = ""
            else:
                last_operator = False
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
        new_equation = self.current_equation
        open_bracket = new_equation.find("(")
        while open_bracket != -1:
            if open_bracket != 0 and new_equation[open_bracket - 1] not in ["(", "+", "-", "/", "*"]:
                before = new_equation[0:open_bracket]
                after = new_equation[open_bracket:]
                new_equation = before + "*" + after
            open_bracket = new_equation.find("(", open_bracket+1)
        closed_bracket = new_equation.find(")")
        while closed_bracket != -1:
            if (closed_bracket + 1) < len(new_equation) and new_equation[closed_bracket + 1] not in [")", "+", "-", "/","*"]:
                before = new_equation[0:closed_bracket+1]
                after = new_equation[closed_bracket+1:]
                new_equation = before + "*" + after
            closed_bracket = new_equation.find(")", closed_bracket + 1)
        self.current_equation = new_equation

    def solve(self, equation):
        self.current_equation = equation
        if self.current_equation.find(")") != -1:
            self.adding_mul()
        print(self.current_equation)
        close_pos = self.current_equation.find(")")
        while close_pos != -1:
            open_pos = self.current_equation.rfind("(",0, close_pos)
            before_br = self.current_equation[0:open_pos]
            after_br = self.current_equation[close_pos + 1:]
            value = self.operator_solver(open_pos + 1 , close_pos)
            self.current_equation = before_br + str(value) + after_br
            print(self.current_equation)
            close_pos = self.current_equation.find(")")
        return self.operator_solver()


solver = Solver()
equation = "(2+4(3-2)4)"
x = solver.solve(equation)
print(x)