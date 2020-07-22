Object Creation:

In a metatype, consider the class::

  class Foo(m.Particle):

      print("hello")

      def thing(self, o):
          print("thing")


If we try something like::

  class Argon(m.Particle):

    print("creating argon type...")
      mass = 39.4

      print("in py, printing on_thing: " + on_thing)

      def on_thing(self, foo):
        print("on_thing")

We get::

  Traceback (most recent call last):
    File "argon.py", line 23, in <module>
      class Argon(m.Particle):
    File "argon.py", line 28, in Argon
      print("in py, printing on_thing: " + on_thing)
  NameError: name 'on_thing' is not defined
  void potential_dealloc(PyObject *)

because the ''on_thing'' has not been defined by when the ''print'' statement is
executed. 
