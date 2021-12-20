def operator_solver(equation):
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
            num+=equation[i]
        i+=1
    if num != "":
        numbers_list.append(float(num))
    count = 0
    while count < len(operators_list):
        if operators_list[count] == "/" or operators_list[count] == "*":
            current_op = operators_list.pop(count)
            if current_op == "/":
                a = numbers_list.pop(count)
                b = numbers_list.pop(count)
                numbers_list.insert(count, (a/b))
                continue
            if current_op == "*":
                a = numbers_list.pop(count)
                b = numbers_list.pop(count)
                numbers_list.insert(count, (a*b))
                continue
        count+=1
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

def solver(equation):
    close_pos = equation.find(")")
    while  close_pos != -1:
        open_pos = equation.rfind("(",0,close_pos)
        before_br = equation[0:open_pos]
        after_br = equation[close_pos+1:]
        value = operator_solver(equation[open_pos+1:close_pos])
        equation = before_br + str(value) + after_br
        close_pos = equation.find(")")
    return operator_solver(equation)

problem = "((22+2)*2.3/23)-(2/2+2)/3"
solution = solver(problem)
print(solution)









