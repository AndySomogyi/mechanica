import mechanica as m
from ipywidgets import widgets

m.Simulator(windowless=True, window_size=[1024,1024])

class Na (m.Particle):
    radius = 0.4
    style={"color":"orange"}

class Cl (m.Particle):
    radius = 0.25
    style={"color":"spablue"}

uc = m.lattice.bcc(0.9, [Na, Cl])

m.lattice.create_lattice(uc, [10, 10, 10])

w = widgets.Image(value=m.system.image_data(), width=600)

display(w)