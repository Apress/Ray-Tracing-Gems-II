# D3D12 Raytraced Decals Sample

This sample modifies the [D3D12 Raytracing Library Subobject sample] as a basis to implement decal support in a ray-tracing scene.
Details on the techniques used are described in the chapter 'Ray Tracing Decals' from the book Ray Tracing Gems II.

The sample assumes familiarity with Dx12 programming and DirectX Raytracing concepts introduced in the D3D12 Raytracing samples.


## Usage
D3D12RaytracedDecals.exe

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


## Requirements
* Windows 10 with the May 2019 update or higher.
* Consult the D3D12 Raytracing samples readme for further requirements.


## Notes
Due to a bug in latest Windows update, the Debug version of the sample
will cause a D3D12 device removal because it calls SetStablePowerState()
when developer mode is enabled. You can work around this in a few ways:
* Just run the Release build of the sample. But you lose stable timing measurements.
* Disable the call to SetStablePowerState() and run Debug build. You also lose stable timing measurements.
* Activate D3D12 experimental features before creating the D3D12 device. This works around the Windows bug for now.

Hopefully this bug is addressed by MS by the time you access this code!
