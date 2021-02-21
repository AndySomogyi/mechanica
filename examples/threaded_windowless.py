import mechanica as m
import threading

m.Simulator(windowless=True, window_size=[1024,1024])

print(m.system.gl_info())

class Na (m.Particle):
    radius = 0.4
    style={"color":"orange"}

class Cl (m.Particle):
    radius = 0.25
    style={"color":"spablue"}

uc = m.lattice.bcc(0.9, [Na, Cl])

m.lattice.create_lattice(uc, [10, 10, 10])

def threaded_steps(steps):

    print('thread start')

    print("thread, calling context_has_current()")
    m.Simulator.context_has_current()\

    print("thread, calling context_make_current())")
    m.Simulator.context_make_current()

    m.step()

    with open('threaded.jpg', 'wb') as f:
        f.write(m.system.image_data())

    print("thread calling release")
    m.Simulator.context_release()

    print("thread done")


# m.system.image_data() is a jpg byte stream of the
# contents of the frame buffer.

print("main writing main.jpg")

with open('main.jpg', 'wb') as f:
    f.write(m.system.image_data())

print("main calling context_release()")
m.Simulator.context_release()

thread = threading.Thread(target=threaded_steps, args=(1,))

thread.start()

thread.join()


print("main thread context_has_current: ", m.Simulator.context_has_current())

m.Simulator.context_make_current()

m.step()


with open('main2.jpg', 'wb') as f:
    f.write(m.system.image_data())


print("all done")
