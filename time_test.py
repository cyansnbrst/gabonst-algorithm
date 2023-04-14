from main import *


def time_test(lang):
    times = []
    length, a, b = 0, 0, 0
    functions = [f1, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f14, f15]
    for function in functions:
        if function == f1:
            continue
            length = 10
            a = -1
            b = 1
        elif function == f3:
            length = 2
            a = -10
            b = 10
        elif function == f4:
            continue
            length = 10
            a = -5
            b = 5
        elif function == f5:
            continue
            length = 2
            a = 0
            b = 10
        elif function == f6:
            continue
            length = 256
            a = -5.12
            b = 5.12
        elif function == f7:
            continue
            length = 30
            a = -30
            b = 30
        elif function == f8:
            continue
            length = 30
            a = -1.28
            b = 1.28
        elif function == f9:
            continue
            length = 30
            a = -5.12
            b = 5.12
        elif function == f10:
            continue
            length = 128
            a = -32.768
            b = 32.768
        elif function == f11:
            continue
            length = 30
            a = -600
            b = 600
        elif function == f12:
            length = 2
            a = -2
            b = 2
        elif function == f14 or function == f15:
            length = 2
            a = -5
            b = 5
        if lang == "c":
            if function == f1:
                function = c1
            elif function == f3:
                function = c3
            elif function == f4:
                function = c4
            elif function == f5:
                function = c5
            elif function == f6:
                function = c6
            elif function == f7:
                function = c7
            elif function == f8:
                function = c8
            elif function == f9:
                function = c9
            elif function == f10:
                function = c10
            elif function == f11:
                function = c11
            elif function == f12:
                function = c12
            elif function == f14:
                function = c14
            elif function == f15:
                function = c15
            start_time = time.time()
            c_genetic_algorithm(length, 20, 20, 0.4, a, b, function_type(function))
            end_time = time.time()
        else:
            start_time = time.time()
            genetic_algorithm(length, 20, 20, 0.4, a, b, function)
            end_time = time.time()
        times.append(end_time - start_time)
        print(f"{function}")
    return times


x1 = np.arange(4) - 0.2
x2 = np.arange(4) + 0.2
y1 = time_test("c")
y2 = time_test("python")

fig, ax = plt.subplots()

ax.bar(x1, y1, width=0.4)
ax.bar(x2, y2, width=0.4)

ax.set_facecolor('seashell')
fig.set_figwidth(12)
fig.set_figheight(6)
fig.set_facecolor('floralwhite')

ax.legend(['С', 'Python + Numba'])

plt.show()
