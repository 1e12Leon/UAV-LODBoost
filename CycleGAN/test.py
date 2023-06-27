import time

import taichi as ti
start =time.time()
ti.init(arch=ti.cpu)
#@ti.func#通过@ti.fun可以定义一个函数,它不能在python的作用域里面被调用,只能在@ti.kernel中被调用:
@ti.kernel
def adas():

    for i in range(0,100000000):
        z=i+1
adas()
end = time.time()
last_time = end - start
print(last_time)

start =time.time()
for i in range(0,100000000):
    z=i+1

end = time.time()
last_time=end-start


print(last_time)