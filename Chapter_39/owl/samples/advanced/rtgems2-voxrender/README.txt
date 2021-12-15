
dlacewell 2021
This is sample code to accompany our article "Ray Tracing Small Voxel Scenes", from Raytracing Gems 2.

We developed this on Windows 10 and the Windows Subsystem for Linux, using CMake 3.19.5 and
Visual Studio 2019 and the build environment.

Use -h to get a list of command line options.  If you build and run with no arguments, a small scene will
be loaded by default from a resource file.  But you can download vox files from the repos below and load
them by passing file paths on the command line.

The main option to be aware of is "--scenetype [user|userblocks|instanced|flat]", which builds different 
acceleration structure layouts as described in the paper.  This causes hostCode.cpp to be a little longer
than you might expect, but the different scene-building functions are orthogonal.  The default scene type 
is "instanced".  Consider switching to "user" or "userblocks" mode if loading very large scenes.


Resources for vox files:

Mini Mike's Metro Minis:
https://github.com/mikelovesrobots/mmmm

voxel-model @ ephtracy:
https://github.com/ephtracy/voxel-model



Here are camera arguments for some figures in the paper if you want to reproduce them.  
(All scenes from the Metro Minis repo)

Mechanic scene:
./voxrender.exe scene_mechanic2.vox --camera "-6.92038 -9.00064 6.3322"

Firetruck (use the '[' and ']' keys to move the clipping plane)
./voxrender.exe veh_fire.vox --camera "-9.42654 -19.2523 12.8643"

Hazmat (edit constants.h and switch to beveled bricks)
./voxrender.exe scene_hazmat.vox  --scenetype instanced --camera "3.17646 -5.70488 2.54116 -0.00587559 -0.00629854 0.0115683 0 0 1"

Scientist (this is the default scene in resource.cpp.  You can change the outlineDepthBias constant in deviceCode.cu)
./voxrender.exe --camera "-13.625 -16.9206 14.31 0.28517 0.34059 0.724736 0 0 1"

Big merged scene (load all the files in the Metro Mini's repo that start with scene_*)
./voxrender.exe --camera "7.93445 -2.90574 5.3478  0.0011816 0.079692 0.0421057" --scenetype user vox/scene_*.vox



