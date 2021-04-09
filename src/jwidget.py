import mechanica as m
from ipywidgets import widgets
import numpy as np
import threading
import time
from ipyevents import Event
from IPython.display import display

flag = False
downflag = False
shiftflag = False

def init(*args, **kwargs):
    global flag, downflag, shiftflag

    w = widgets.Image(value=m.system.image_data(), width=600)
    d = Event(source=w, watched_events=['mousedown', 'mouseup', 'mousemove', 'keyup','keydown', 'wheel'])
    no_drag = Event(source=w, watched_events=['dragstart'], prevent_default_action = True)
    d.on_dom_event(listen_mouse)
    run = widgets.ToggleButton(
        value = False,
        description='Run',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='run the simulation',
        icon = 'play'
        )
    pause = widgets.ToggleButton(
        value = False,
        description='Pause',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='pause the simulation',
        icon = 'pause'
        )

    reset = widgets.ToggleButton(
        value = False,
        description='Reset',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='reset the simulation',
        icon = 'stop'
        )

    def onToggleRun(b):
        global flag
        if run.value:
            run.button_style = 'success'
            pause.value = False
            pause.button_style = ''
            reset.value = False
            reset.button_style = ''
            flag = True
        else:
            run.button_style=''
            flag = False

    def onTogglePause(b):
        global flag
        if pause.value:
            pause.button_style = 'success'
            run.value = False
            run.button_style = ''
            reset.value = False
            reset.button_style = ''
            flag = False
        else:
            pause.button_style=''
            flag = True

    def onToggleReset(b):
        global flag
        if reset.value:
            reset.button_style = 'success'
            pause.value = False
            pause.button_style = ''
            run.value = False
            run.button_style = ''
            flag = False
            m.Universe.reset()
        else:
            reset.button_style=''
            #w = create_simulation()

    buttons = widgets.HBox([run, pause, reset])
    run.observe(onToggleRun,'value')
    pause.observe(onTogglePause,'value')
    reset.observe(onToggleReset,'value')

    box = widgets.VBox([w, buttons])
    display(box)

    # the simulator initializes creating the gl context on the creating thread.
    # this function gets called on that same creating thread, so we need to
    # release the context before calling in on the background thread.
    m.system.context_release()

    def background_threading():
        global flag
        m.system.context_make_current()
        while True:
            if flag:
                m.step()
            w.value = m.system.image_data()
            time.sleep(0.01)

        # done with background thead, release the context.
        m.system.context_release()


    t = threading.Thread(target=background_threading)
    t.start()



def run(*args, **kwargs):
    global flag

    flag = True

    # return true to tell Mechanica to not run a simulation loop,
    # jwidget runs it's one loop.
    return True


def listen_mouse(event):
    global downflag, shiftflag
    if event['type'] == "mousedown":
        m.system.camera_init_mouse([event['dataX'], event['dataY']])
        downflag = True
    if event['type'] == "mouseup":
        downflag = False
    if event['type'] == "mousemove":
        if downflag and not shiftflag:
            m.system.camera_rotate_mouse([event['dataX'], event['dataY']])
        if downflag and shiftflag:
            m.system.camera_translate_mouse([event['dataX'], event['dataY']])

    if event['shiftKey'] == True:
        shiftflag = True
    if event['shiftKey'] == False:
        shiftflag = False
    if event['type'] == "wheel":
        m.system.camera_zoom_by(event['deltaY'])
    if event['type'] == "keydown" and event['code'] == "KeyR":
        m.system.camera_reset()
