

def init(*args, **kwargs):
    print("hello from mechanica.jwidget.init()")


def run(*args, **kwargs):
    print("hello from mechanica.jwidget.run()")

    # return true to tell Mechanica to not run a simulation loop,
    # jwidget runs it's one loop.
    return True
