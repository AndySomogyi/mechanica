***
#Transformation_exercise1.cpp is supposed to make the image upside down.
***

##1. Vertex Shader
- I add "uniform mat4 trans" here to rotate image. 
- Also, multiplying by "trans" will the change the window coordinates.

	  	layout(location=0) in vec2 position;
	    layout(location=1) in vec3 color;
	    layout(location=2) in vec2 texcoord;
	
		out vec3 Color;
		out vec2 Texcoord;
 
 		uniform mat4 trans;
 
 		void main() {
 	   		Color = color;
 	   		Texcoord = texcoord;
       		gl_Position = trans * vec4(position, 0.0, 1.0);
  	 	}
  	 	
  	 	
  
  
  	 	
##2. Textured Shader

- I add Private Integer value called transLoc. 

- "uniformLocation("trans");" will grab trans type mat4 from vertex shader.
-  .setTrans(...) will be use in Transformation1 class and it will find the location of a uniform variable called "trans"


		public:
			...
			...
			transLoc = uniformLocation("trans");
			...
			...
		
		void setTrans(const Matrix4& mat) {
 		      setUniform(transLoc, mat);
      	}
      	
      	private:
 			  int transLoc;
  		};

##3. Transformation1 class
- The first line of the code is actuall part that rotating image by 180 degree(3.141592(radian)). 
- shader.setTrans(trans); is taking trans type of Matrix4 and do work.

		Matrix4 trans = Matrix4::rotationZ(Rad{3.141592f});
		
   		shader.setTrans(trans);




##4. Magnum Application
- I have changed into "MAGNUM _ APPLICATION _ MAIN(transformation1)" It follows the typical structure of Manum Skeleton.

		MAGNUM_APPLICATION_MAIN(transformation1)




***
#Tranformation_exercise2.cpp is supposed to rotate the given image.
***

##1. Chrono
- I include chrono library to measure time.
		
		include <chrono>

##2. Vertex Shader
- I add one more variable named model of mat4 type.
- multiply model on gl_Position.
- It will make image rotate by specific period that we entered.

	  	layout(location=0) in vec2 position;
	    layout(location=1) in vec3 color;
	    layout(location=2) in vec2 texcoord;
	
		out vec3 Color;
		out vec2 Texcoord;
 
 		uniform mat4 trans;
 		uniform mat4 model;
 
 		void main() {
 	   		Color = color;
 	   		Texcoord = texcoord;
       		gl_Position = trans * model * vec4(position, 0.0, 1.0);
  	 	}
  	 	
  	 	
  
  
  	 	
##3. Textured Shader

- I add Private Integer value called modelLoc. 

- "uniformLocation("model");" will grab trans type mat4 from vertex shader.
-  .setModel(...) will be use in Transformation2 class and find the location of a uniform variable called "model"

		public:
			...
			

##8. Magnum Application
- I have changed into "MAGNUM _ APPLICATION _ MAIN(transformation2)" It follows the typical structure of Manum Skeleton.

		MAGNUM_APPLICATION_MAIN(transformation2)

***
#Transformation_3d.cpp is supposed to change the view point of image.
***


##1. Vertex Shader
- I add one more variable named "view" of mat4 type.
- Also, I changed "trans" variable to "proj"
- multiply "proj" and "view" on gl_Position.
- It will make image in dfferent view.

	  	uniform mat4 proj;
		uniform mat4 model;
		uniform mat4 view;

		void main() {
    		Color = color;
    		Texcoord = texcoord;
    
    		gl_Position = view * model * proj * vec4(position, 0.0, 1.0);
    	}
  	 	
  	 	
  
  
  	 	
##2. Textured Shader

- I add Private Integer value called projLoc and viewLoc. 

- "uniformLocation("proj");" and "uniformLocation("view");" will grab variable called "proj" and "view" type mat4 from vertex shader.
-  .setProj(...) will be use in Transformation2 class and find the location of a uniform variable called "model" 
-  Same principle apply to .setView(...).

		public:
			...
			...
			projLoc= uniformLocation("proj");
        	modelLoc = uniformLocation("model");
        	viewLoc = uniformLocation("view");
			...
			...
		
		 void setProj(const Matrix4& mat) {
        		setUniform(projLoc, mat);
    	 }
    
    	void setModel(const Matrix4& mat) {
       	   setUniform(modelLoc, mat);
    	}
    
    	void setView(const Matrix4& mat) {
        	   setUniform(viewLoc, mat);
    	}
      	
      	private:
 			  int projLoc;
    		  int modelLoc;
    	      int viewLoc;

##4. Transformation3D class
- I add Matrix4 variable called "view", it will change the location of camera which mean change the view point.
- I changed proj that can create perspective projection matrix.


		Matrix4 view = Matrix4::lookAt(Vector3{1.2f, 1.2f, 1.2f},
                                       Vector3{0.0f, 0.0f, 0.0f},
                                       Vector3{0.0f, 0.0f, 1.0f});
        shader.setView(view);
        
        Matrix4 proj = Matrix4::perspectiveProjection
        						(Rad{45.0f}, 800.0f / 600.0f, 1.0f, 10.0f);
        shader.setProj(proj);

##5. Magnum Application
- I have changed into "MAGNUM _ APPLICATION _ MAIN(transformation1)" It follows the typical structure of Manum Skeleton.

		MAGNUM_APPLICATION_MAIN(transformation3D)




