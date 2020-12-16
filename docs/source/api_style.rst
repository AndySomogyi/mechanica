Style
-----


.. class:: Style(object)

   The Style object contains all of the rendering visual style attribute of an
   object. Most simulation objects in Mechanica have a `style` attribute.

   .. attribute:: color

      ``Color4`` -- gets / sets the color of the style. This is actually a
      ``Color4`` object, but we can set it's color value either as a Color
      object, or via one of the color strings from :ref:`Colors`, i.e. ::

        >>> c = m.Style()
        >>> c.color = "DeepSkyBlue"

      The color string is not case sensitive.


   .. attribute:: visible

      ``boolean`` -- gets / sets the visible property. This toggles the
      visibility of any rendered object.




Colors
^^^^^^

You can construct a color object, (either a Color3 or Color4) using one of the
following web color names:

* "AliceBlue",
* "AntiqueWhite",
* "Aqua",
* "Aquamarine",
* "Azure",
* "Beige",
* "Bisque",
* "Black",
* "BlanchedAlmond",
* "Blue",
* "BlueViolet",
* "Brown",
* "BurlyWood",
* "CadetBlue",
* "Chartreuse",
* "Chocolate",
* "Coral",
* "CornflowerBlue",
* "Cornsilk",
* "Crimson",
* "Cyan",
* "DarkBlue",
* "DarkCyan",
* "DarkGoldenRod",
* "DarkGray",
* "DarkGreen",
* "DarkKhaki",
* "DarkMagenta",
* "DarkOliveGreen",
* "Darkorange",
* "DarkOrchid",
* "DarkRed",
* "DarkSalmon",
* "DarkSeaGreen",
* "DarkSlateBlue",
* "DarkSlateGray",
* "DarkTurquoise",
* "DarkViolet",
* "DeepPink",
* "DeepSkyBlue",
* "DimGray",
* "DodgerBlue",
* "FireBrick",
* "FloralWhite",
* "ForestGreen",
* "Fuchsia",
* "Gainsboro",
* "GhostWhite",
* "Gold",
* "GoldenRod",
* "Gray",
* "Green",
* "GreenYellow",
* "HoneyDew",
* "HotPink",
* "IndianRed",
* "Indigo",
* "Ivory",
* "Khaki",
* "Lavender",
* "LavenderBlush",
* "LawnGreen",
* "LemonChiffon",
* "LightBlue",
* "LightCoral",
* "LightCyan",
* "LightGoldenRodYellow",
* "LightGrey",
* "LightGreen",
* "LightPink",
* "LightSalmon",
* "LightSeaGreen",
* "LightSkyBlue",
* "LightSlateGray",
* "LightSteelBlue",
* "LightYellow",
* "Lime",
* "LimeGreen",
* "Linen",
* "Magenta",
* "Maroon",
* "MediumAquaMarine",
* "MediumBlue",
* "MediumOrchid",
* "MediumPurple",
* "MediumSeaGreen",
* "MediumSlateBlue",
* "MediumSpringGreen",
* "MediumTurquoise",
* "MediumVioletRed",
* "MidnightBlue",
* "MintCream",
* "MistyRose",
* "Moccasin",
* "NavajoWhite",
* "Navy",
* "OldLace",
* "Olive",
* "OliveDrab",
* "Orange",
* "OrangeRed",
* "Orchid",
* "PaleGoldenRod",
* "PaleGreen",
* "PaleTurquoise",
* "PaleVioletRed",
* "PapayaWhip",
* "PeachPuff",
* "Peru",
* "Pink",
* "Plum",
* "PowderBlue",
* "Purple",
* "Red",
* "RosyBrown",
* "RoyalBlue",
* "SaddleBrown",
* "Salmon",
* "SandyBrown",
* "SeaGreen",
* "SeaShell",
* "Sienna",
* "Silver",
* "SkyBlue",
* "SlateBlue",
* "SlateGray",
* "Snow",
* "SpringGreen",
* "SteelBlue",
* "Tan",
* "Teal",
* "Thistle",
* "Tomato",
* "Turquoise",
* "Violet",
* "Wheat",
* "White",
* "WhiteSmoke",
* "Yellow",
* "YellowGreen",

For example, to make some colors::
  >>> m.Color3("red")
  Vector(1, 0, 0)

  >>> m.Color3("MediumSeaGreen")
  Vector(0.0451862, 0.450786, 0.165132)

  >>> m.Color3("CornflowerBlue")
  Vector(0.127438, 0.300544, 0.846873)

  >>> m.Color3("this is total garbage")
  /usr/local/bin/ipython3:1: Warning: Warning, "this is total garbage" is not a valid color name.
  #!/usr/local/opt/python/bin/python3.7
  Vector(0, 0, 0)

As it's easy to make a mistake with color names, we simply issue a warnign,
instead of an error if the color name can't be found. 


.. class:: Color3

   The class can store either a floating-point or integer representation of
   a linear RGB color. Colors in sRGB color space should not beused directly in
   calculations — they should be converted to linear RGB using fromSrgb(),
   calculation done on the linear representation and then converted back to sRGB
   using toSrgb().

   You can construct a Color object 



   .. staticmethod:: from_srgb(srgb(int))

      constructs a color from a packed integer, i.e.::

        >>> c = m.Color3.from_srgb(0xffffff)
        >>> print(c)
        Vector(1, 1, 1)


