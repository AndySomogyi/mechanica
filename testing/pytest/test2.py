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



import mechanica as m
import _mechanica

print ("mechanica file: " + m.__file__, flush=True)
print("_mechanica file: " + _mechanica.__file__, flush=True)

print(dir(m))

s = m.Simulator()

print(dir(s))