# MiniPhotomath

To build a docker image use this comand

	docker build https://github.com/Kyboky/MiniPhotomath.git -t miniphotomath
	
Create container with a image

	docker container create --name miniphotomath_container --publish 5000:5000 miniphotomath
  
To run container

	docker container start miniphotomath_container
	
If you want to stop container use 

	docker container stop miniphotomath_container
  


	
