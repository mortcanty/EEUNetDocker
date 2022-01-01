EEUNetocker
===========
Source files for the Docker image mort/eeunetdocker
Docker image with Jupyter widget interface for 
UNet semantic classification on Earth Engine hires imagery

Pull and/or run the container with

    sudo docker run -d -p 443:8888 --name=eeunet mort/eeunetdocker  
    
Point your browser to http://localhost:443 to see the Jupyter notebook home page. 
 
Open the Notebook 

    interface.ipynb 

Stop the container with

    sudo docker stop eemad 
     
Re-start with

    sudo docker start eemad   