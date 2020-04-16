import ctypes
import sys
import os.path as path


cur_dir = path.dirname(path.realpath(__file__))

print(cur_dir)

print(sys.path)

sys.path.append("/Users/andy/src/mechanica/src")
sys.path.append("/Users/andy/src/mx-xcode/Debug/lib")
#sys.path.append("/Users/andy/src/mx-eclipse/src")

print(sys.path)


import _mechanica as m

print(m.Test)



t = m.Test()


o = t.test()

print(o)



