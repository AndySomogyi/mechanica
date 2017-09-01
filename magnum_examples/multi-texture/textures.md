***
#multi_texture.cpp is supposed to change the texture of given image.
***

##1. Fragment Shader
- I multiply vec4 to outColor. It will add some effect on the image. 
- Also, I changed the value to 1.5 from 0.5 to make puppy clearer.

	  	outColor = mix(texture(texKitten, Texcoord), 
	  				   texture(texPuppy, Texcoord), 1.5) * 
	  				   vec4(Color , 1.0);
  	 	
  	 	
			

##2. Magnum Application
- I have changed into "MAGNUM _ APPLICATION _ MAIN(MultiTexture)" It follows the typical structure of Manum Skeleton.

		MAGNUM_APPLICATION_MAIN(MultiTexture)
