# MiniPhotomath

To build a docker image use this comand

	docker build https://github.com/Kyboky/MiniPhotomath.git -t miniphotomath
	
Create container

	docker container create --name miniphotomath_container --publish 5000:5000 miniphotomath
  
To run image in container

	docker run --publish 5000:5000 miniphotomath
  
The container should be started now.

	
