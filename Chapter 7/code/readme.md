# D3D12 Texture-coordinate Gradients Estimation for Ray Cones Sample

This sample modifies the [D3D12 Raytracing Library Subobject sample] as a basis to implement the technique of the chapter
'Texture-coordinate Gradients Estimation for Ray Cones' from the book Ray Tracing Gems II.

The sample assumes familiarity with Dx12 programming and DirectX Raytracing concepts introduced in the D3D12 Raytracing samples.


## Usage
D3D12RayconeGrads.exe

Additional arguments:
  * [-forceAdapter \<ID>] - create a D3D12 device on an adapter \<ID>. Defaults to adapter 0.


### UI
The title bar of the sample provides runtime information:
* Name of the sample
* Frames per second
* Million Primary Rays/s: a number of dispatched rays per second calculated based of FPS.
* GPU[ID]: name

In the sample, press [Backspace] to activate the tuning menu. Navigate with the arrows (up & down) to
move between the options. And use left & right arrows to change option values and to expand sub-menus.
Sub-menus are marked with a '+' sign when collapsed.

For this sample, please expand the 'Application' sub-menu, then 'Decals'. Options for controlling decal
ray-tracing are then offered and can be quickly changed while noticing the impact on performance.

The sample has multiple code paths to
showcase the differences between all the techniques referenced in this article. The scene is
drawn using either rasterization or ray-tracing, depending on the mode selected by
the user. The modes are:
* Mode 1: Draw using rasterization and use the standard Texture2D.Sample() HLSL instruction. This is the reference for mip-mapping quality. This is also the application's initial mode.
* Mode 2: Draw using ray-tracing and sample the texture using Texture2D.SampleLevel(), where the mip level index is fixed to level 0 (the most detailed level). The visual results suffer from heavy aliasing due to lack of mip-mapping.
* Mode 3: Draw using ray-tracing and sample the texture using Texture2D.SampleLevel(), where the mip level index is computed according to the original ray-cones implementation.
* Mode 4: Draw using ray-tracing and sample the texture using Texture2D.SampleGrad(), where the mip level index is computed following the new approach described in this article.

To toggle between the modes, simply press the mode's corresponding number on the keyboard (1-4).


## Requirements
* Windows 10 with the May 2019 update or higher.
* Consult the D3D12 Raytracing samples readme for further requirements.