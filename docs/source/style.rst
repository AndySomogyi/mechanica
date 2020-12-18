.. _style-label:

Style
=====

All renderable objects in Mechanica have a `style` attribute. This style is
essentially like a CSS `.style` attribute in Javascript / HTML. The style object
behaves like a container for a variety of style descriptors. Our intent is to
eventually read and generate all of the style nodes from a CSS file, but for the
time being, we have to manually set style information. 

Like CSS, each instance of a object automatically inherits the style of it's
type, but users may override values.

Currently, the only supported style descriptor is a `color` value. This is an
instance of the 
