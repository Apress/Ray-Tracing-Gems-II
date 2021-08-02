# optix7course in OWL

iw, jan 8, 2020.

This is the OWL-version of the tutorial example I origianlly created
for the Siggrah 2019 course on OptiX 7 - not necessarily to demonstrate
the best-ever optix renderer/model viewer, but mostly intended as a 
step-by-step tutorial/show-and-tell of how to initialize optix, build
an SBT, build acceleration structures, etc. (original code is still
available here: https://github.com/ingowald/optix7course )

The original course samples used OptiX 7 directly, with its own,
"manual" setup of geometry, accel structures, SBT, etc; the purpose of
this OWL sample is to show how pretty much exactly the same ourcome
can be also achieved by using OWL, and much more simply. Most of the
actual app, viewer, and device-side rendering code is exactly the
same, in particular, the closest hit, raygen, etc programs are
*exactly* the same as before - but all the host code for setting up
the buffers, accel structs, and SBT, is now much simpler, and shorter.

As of this writing there are still a few things missing; in
particular 

a) I still use the original sample's CUDA-texture code (there is no
owl-specific support for textures, yet); but arguably, that is good
because it shows how easily OWL, optix, and CUDA can co-exist; and

b) I have currently disabled the denoiser that the original sample
already used - it *should* be possible to use that in the same way as
I use the CUDA textures, but I plan on natively supporting the
denoiser, anyway, so didn't want to go that route.

Unlike most other OWL samples this sample intentionally supports only
single-GPU rendering, primarily because the CUDA Texture setup code
taken from the original sample requires this. This would be easy to
fix, but since that problem will disappear the moment natively
supports textures, I leave this as is for now.
