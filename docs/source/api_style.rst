Style
------

Events are the principle way users can attach connect thier own functions, and
built in methods event triggers. 


.. module::  mechanica


.. class:: Color3

   The class can store either a floating-point or integer representation of
   a linear RGB color. Colors in sRGB color space should not beused directly in
   calculations — they should be converted to linear RGB using fromSrgb(),
   calculation done on the linear representation and then converted back to sRGB
   using toSrgb().



   .. staticmethod:: from_srgb(srgb(int))

      constructs a color from a packed integer, i.e.::

        >>> c = m.Color3.from_srgb(0xffffff)
        >>> print(c)
        Vector(1, 1, 1)


