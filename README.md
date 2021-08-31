# neuronephys_v2

Neuronephys is a simulation framework for large-scale neural networks with realistic ionic dynamics. 

# Installation 
- Requirements:
  - An NVidia GPU 
  - Windows OS 
1. You first need to install CUDA 11.1 which can be done through their site.
2. Second, install visual studio 2019. 
3. Finally, clone this repository, open coupling_speed_test.sln in VS 2019, build in **Release**, and run!

# Requirements for mouse brain:
For the mouse brain simulations, the connectivity must and morphology files are needed. These can be retrieved at: 
[https://drive.google.com/file/d/1AZ0BQPchTO6ox4L7q46hkCfSDCEGNA3z/view?usp=sharing](https://drive.google.com/file/d/1AZ0BQPchTO6ox4L7q46hkCfSDCEGNA3z/view?usp=sharing). 
Download the files: cortex_connectivity_1m_1k_inh.h5 and cortex_connectivity_1m_1k_exc.h5 and put them in the neuronephys_v2 root directory. 

# Using the visualization
The visualization in OpenGL allows the user to get an understanding of the spatial patterns of behavior in the simulation. A number of controls are supported:
- Scrolling with the mouse wheel zooms in/out. 
- W and S move the model in the z direction (forwards/backwards), A and D move the brain in the x direction (left/right), E and Q move the brain in the Y direction (up/down).
- Holding the left mouse button allows rotation of the model.
- Pressing R turns on continuous model rotation. 
- Space pauses the simulation. 
- Esc ends the simulation.
