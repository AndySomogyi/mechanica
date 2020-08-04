Writing Output Data With Events
-------------------------------


Writing output data is a frequently done task with simulations. The
built-in :any:`on_time` event provides a very convenient system to register an
output function that you want to get called at regular intervals. 

During event execution, the simulation is halted, so it's important to do
whatever processing quickly as no to slow down the simulation.

An example of an output function would be to iterate over the particles, and
write their positions to a file. Starting with out previous mitosis events
example, :ref:`mitosis-events-label`, we can add a function to write output
data, and add that to a time event. We use the Python csv writer to write
this to an output file::

  def write_data(time):
    print("time is now: ", time)

    positions = [list(p.position) for p in m.Universe.particles]

    with open(fname, "a") as f:
      writer = csv.writer(f)
      writer.writerow([time, positions])

Here we iterate over the particles, grab the particle positions, convert it to
a list, and concatenate that together with the time, and write that out as a row
to the output file. 

And we attach that function to the :any:`on_time` event. We only specify a
period, and do not give any distribution as we want this to be called at
regular intervals::

  m.on_time(write_data, period=0.05)

 
The complete simulation script is here, and can be downloaded here:

Download: :download:`this example script <../../examples/writedata_events.py>`::


