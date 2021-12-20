# MiniPhotomath

To build a docker image use this comand

	docker build https://github.com/Kyboky/MiniPhotomath.git -t miniphotomath
	
Create container with a image

	docker container create --name miniphotomath_container --publish 5000:5000 miniphotomath
  
    docker container create --name miniphotomath_container --publish 443:5000 miniphotomath

To run container

	docker container start miniphotomath_container
	
After starting container use your phone or other device with camera, open browser and type

	https://[host-pc ip]:5000/

In case you have used second container create method with "--publish 443:5000" there is no need for specifying port ":5000"

    https://[host-pc ip]/

Where pcip is ip of a pc that is running a container.

If you want to stop container use 

	docker container stop miniphotomath_container
  


	
