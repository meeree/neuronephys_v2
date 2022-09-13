# neuronephys_v2

Neuronephys is a simulation framework for large-scale neural networks with realistic ionic dynamics. 

# Windows Installation 
- Requirements:
  - An NVidia GPU 
  - Windows OS 
1. You first need to install CUDA 11.1 which can be done through their site. **WHEN INSTALLING** select custom and install everything. This include visual studio integration. 
2. Second, install visual studio 2019. 
3. Finally, clone this repository, open coupling_speed_test.sln in VS 2019, build in **Release**, and run!

# Linux Installation 
- Requirements: 
  - An NVidia GPU 
  - Linux OS 
1. Install CUDA 11.1 with visual studio integration. The version does not really matter. If on a cluster, load the CUDA module.
2. Cd into neuronephys_v2/linux_build using the terminal
3. Run "make" or "make -j<THREAD COUNT>" to make the library.
4. The executable "mainS" can be run to execute the code. More instructions in INSTRUCTIONS.txt

# Requirements for mouse brain:
For the mouse brain simulations, the connectivity must and morphology files are needed. These can be retrieved at: 
[https://drive.google.com/drive/folders/1Rg8oW7xpxMQzeGC2LgKM8gPe3o-xHbK9?usp=sharing]https://drive.google.com/drive/folders/1Rg8oW7xpxMQzeGC2LgKM8gPe3o-xHbK9?usp=sharing). 
Download the files: cortex_connectivity_1m_1k_inh.h5 and cortex_connectivity_1m_1k_exc.h5 and put them in the neuronephys_v2 root directory. 

# Using the visualization
The visualization in OpenGL allows the user to get an understanding of the spatial patterns of behavior in the simulation. A number of controls are supported:
- Scrolling with the mouse wheel zooms in/out. 
- W and S move the model in the z direction (forwards/backwards), A and D move the brain in the x direction (left/right), E and Q move the brain in the Y direction (up/down).
- Holding the left mouse button allows rotation of the model.
- Pressing R turns on continuous model rotation. 
- Space pauses the simulation. 
- Esc ends the simulation.
