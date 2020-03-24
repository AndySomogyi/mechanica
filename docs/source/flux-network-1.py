# create the three compartments, set thier volume to 1, and 
# define a species called "S1" in each comparment
n1 = Compartment(volume=1, species="S1")
n2 = Compartment(volume=1, species="S1")
n3 = Compartment(volume=1, species="S1")

# use the built-in passive (Fickian) flux to connect the 
# species between different comparments. The first argument
# is the source. If the source is a number, we treat that as 
# constant value (boundary condition). The second argument
# is the destination, and the last is the flux rate. The
# Fick flux is k * (input - output). 
j1 = PassiveFlux(1.5,   n1.S1, 0.1)
j2 = PassiveFlux(n1.S1, n2.S1, 0.2)
j3 = PassiveFlux(n1.S1, n3.S1, 0.3)
j4 = PassiveFlux(2.0,   n2.S1, 0.4)
j5 = PassiveFlux(n3.S1, 0.01,  0.5)




