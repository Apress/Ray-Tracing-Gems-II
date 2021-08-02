# `s07-rtow-multiGPU` : Illustrates Simple Way of Doing Multi-GPU in OWL

This sample illustrates how to do multi-GPU rendering with OWL; it is
based off the previous `rt06-rtow-mixedGeometries`with (intentionally)
minimal changes (ie, the diff between these examples is the minimum
required to make it do multi-GPU).

Doing Multi-GPU in OWL is actually pretty easy, as owl was designed
for Multi-GPU from the very start: The centerpiece of OWL - the
context - abstracts not a single device, but a *group* of devices
(which GPUs it is to use is specified during context creation); and
every operation (e.g., creating a buffer, launching a program etc)
will automatically be replicated across all devices. OWL will also
automatically build acceleration structures, SBTs, etc, on each
device.

OWL also automatically handles all "translation" from OWLBuffers,
OWLGroups, etc, to device-side addresses - ie, even if a given Buffer,
Group, etc will map to different addresses for different devices, OWL
will automatically handle that, and when buildinng the given device's
SBT, buffers, etc, will automatically fill it with the proper
addresses for the given device (in case you were wondering, this is
*exactly* why `OWLVariable`s exist: they allow to assign a single
handle on the host, and translate it to different addresses when the
per-device data are built). 

As such, all that is required to have different GPUs share in the same
work is to a) give them a means of combining their partial results (a
simple way of which is to use a host-mapped frame buffer, whose CUDA
Host-Pinned memory is uniformly accessible across both host and all
devices!), and b) to have the devices agree on which device renders
which pixels.

# Example: from `rtow-mixedGeometries` to `rtow-multiGPU`

In this (minimal example, there are only three small changes to the single-GPU example to make it work across multiple GPUs:

## 1) Create Context over mutiple GPUs

When creating the root owl context (`owlContextCreate`) one can pass a
list of CUDA device IDs that this context is supposed to be created
over. In the first few examples we always passed `(nullptr,1)`, which
is short-hand for "1 GPU, in default order" (ie, the first
CUDA-capable GPU in the system; though later versions may change that
to 'the most powerful one'). 

In this example, all we need to do is change this to `(nullptr,0)`,
which is short-hand for "anything you can find".

## 2) Make device code aware of number of device, and which one it runs on

OWL gives user code the *option* to do multi-GPU, but does not enforce
a specific way of how to do it. As such, it's the job of the user's
owl device code (typically, the RayGen program) to determine which
pixels each device wants to render. 

The user can do that in any way it wants (eg, by interleaving pixels
in a frame buffer, by rendering left-half vs right half, odd/even y
coordinates, etc); however, to decide that the programs need to know
a) how many devices there are total; and b) which device the current
code runs on.

For 'a', we allow the user to query the num devices on the host
`owlGetDeviceCount`; and it can then pass this to a variable of his
choice as a `OWL_INT` type). For `b`, OWL offers a special variable
type `OWL_DEVICE` that on each device will evaluate to and int that is
exactly the index of thise device (e.g., it will evaluate to 0 on the
first device, to 1 on the second, etc). Note that unlike other
variables this type will never get *set* on the host, but will always
evaluate to the right value on each device.

In this particular example, this is done as follows: 

First, in the `RayGenData` type we add two ints to hold device index and count:
```
    struct RayGenData {
	  ...
      int deviceIndex;
      int deviceCount;
	  ...
	}
```

Then when creating the raygen in the host code, we export these two variables, once as a int (to be set by the user), and once using the explicit `OWL_DEVICE` type:

```
  OWLVarDecl rayGenVars[] = {
    { "deviceIndex",   OWL_DEVICE, OWL_OFFSETOF(RayGenData,deviceIndex)},
    { "deviceCount",   OWL_INT,    OWL_OFFSETOF(RayGenData,deviceCount)},
    ...
  }
```

As described above the `OWL_DEVICE` type is fully implicit, and does
not/cannot be set by the user; but the device count does; we do this
in the host code when we create the actual raygen:
```
  int numGPUsFound = owlGetDeviceCount(context);
  owlRayGenSet1i(rayGen,"deviceCount",  numGPUsFound);
```

From now on, the raygen program will know both its device index, as
well as how many devices there are total.

# 3) Made Device Code Split the Rendering Work

As said above there are many ways of doing this; in this example, we
simply launch the full frame buffer size across all devices (ie, every
devices launches over all pixels), and then have each device simply
drop/skip some of the work. To do this we divide the frame buffer into
32-wide coluns pixels, and assign those round-robin to the different
devices; then in the raygen program we simply take the current launhc
index, compute who is respsonsible for the given pixels, and drop them
if these are to be rendered by a different GPU:

```
  const vec2i pixelID = owl::getLaunchIndex();
  int deviceThatIsResponsible = (pixeID.x>>5) % self.deviceCount;
  if (self.deviceIndex != deviceThatIsResponsible)
    /* sombody else's job ... */ 
	return;
```

Since this sample already uses a host pinned frame buffer we do not
have do any extra work to merge the partial results; every pixel will
get produced by one of the GPUs, and will get written into the frame
buffer; so at the end of the launch every pixel in the frame buffer
will be properly set.
