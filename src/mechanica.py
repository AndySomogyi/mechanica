import _mechanica

import ctypes

from _mechanica import Simulator
from _mechanica import CylinderModel
from _mechanica import ParticleType
from _mechanica import Particle
from _mechanica import Potential
from _mechanica import Universe

print(type(ParticleType))

try:
    import IPython
    import datetime
    import sys
    import time
    import signal
    from timeit import default_timer as clock

    # Frame per second : 60
    # Should probably be an IPython option
    mechanica_fps = 60


    ip = IPython.get_ipython()

    outfile = open("pylog.txt",  "a")


    def inputhook(context):
        """Run the event loop to process window events

        This keeps processing pending events until stdin is ready.  After
        processing all pending events, a call to time.sleep is inserted.  This is
        needed, otherwise, CPU usage is at 100%.  This sleep time should be tuned
        though for best performance.
        """

        try:
            t = clock()

            # Make sure the default window is set after a window has been closed

            while not context.input_is_ready():


                outfile.write(datetime.datetime.now().__str__() + "\n")
                outfile.flush()

                _mechanica.pollEvents()

                continue

                # We need to sleep at this point to keep the idle CPU load
                # low.  However, if sleep to long, GUI response is poor.  As
                # a compromise, we watch how often GUI events are being processed
                # and switch between a short and long sleep time.  Here are some
                # stats useful in helping to tune this.
                # time    CPU load
                # 0.001   13%
                # 0.005   3%
                # 0.01    1.5%
                # 0.05    0.5%
                used_time = clock() - t
                if used_time > 10.0:
                    # print 'Sleep for 1 s'  # dbg
                    time.sleep(1.0)
                elif used_time > 0.1:
                    # Few GUI events coming in, so we can sleep longer
                    # print 'Sleep for 0.05 s'  # dbg
                    time.sleep(0.05)
                else:
                    # Many GUI events coming in, so sleep only very little
                    time.sleep(0.001)

        except KeyboardInterrupt:
            pass

        outfile.write("user input ready, returning, " + datetime.datetime.now().__str__())



    def registerInputHook():
        """
        Registers the mechanica input hook with the ipython pt_inputhooks
        class.

        The ipython TerminalInteractiveShell.enable_gui('name') method
        looks in the registered input hooks in pt_inputhooks, and if it
        finds one, it activtes that hook.

        To acrtivate the gui mode, call:

        ip = IPython.get_ipython()
        ip.
        """
        import IPython.terminal.pt_inputhooks as pt_inputhooks
        pt_inputhooks.register("mechanica", inputhook)


    def enableGui():

        import IPython
        ip = IPython.get_ipython()
        registerInputHook()
        _mechanica.initializeGraphics()
        ip.enable_gui("mechanica")

    def createTestWindow():
        enableGui()
        _mechanica.createTestWindow()

    def destroyTestWindow():
        _mechanica.destroyTestWindow()



    ### Module Initialization ###

    registerInputHook()

except:
    pass
