# For example, say we create a type:
class Foo(Particle):
    mass = 1

# now we create a bunch of instances of that type:
for i in range(1,1000):
    Foo(x=numpy.random((1,3)))

# To get the temperature (or average kinetic energy) of that family, we simply do:
print(Foo.temperature)
print(Foo.kinetic_energy)


# To control the temperature, we can set the target_temperature property on the class:
Foo.target_temperature = 275

# Then we need a process that controls this temperature, so we add a temp process:
bind(forces.berendensen_thermostat(), Foo)

# we can do the same with other kinds of force process, like say a fricition, or random motion 
bind(forces.friction(), Foo)