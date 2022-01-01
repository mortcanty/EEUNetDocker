EEUNetDocker
===========
Source files for the Docker image mort/eeunetdocker

Docker image with Jupyter widget interface for exporting NAIP high-res images from GEE

UNet semantic classification on Earth Engine hires imagery

To be able to run the unet classifier, place the trained model
 
	unet_inria_model4.h5 

(available from the repo owner) in a local directory <my_model:folder>. Get help with

	%run scripts/unetclassify -h

Pull and/or run the container with

    sudo docker run -d -p 443:8888 -v <my_model_folder>:/home/  --name=eeunet mort/eeunetdocker  
    
Point your browser to http://localhost:8888 to see the Jupyter notebook home page. 
 
Open the Notebook 

    interface.ipynb 
    
Run the GEE authenticate script (once only)    

Stop the container with

    sudo docker stop eemad 
     
Re-start with

    sudo docker start eemad   