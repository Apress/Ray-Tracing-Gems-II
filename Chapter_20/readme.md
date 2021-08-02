Sample program for the RTG2 article on Multiple Importance Sampling.
Focus is on making a simple-to-read path tracer and support the article itself.
All the pseudo-code snippets in the article are taken from the source code in order to unit test the article.
Because of this .f has been omitted on all floats which means that the path tracer runs using doubles instead of floats.
The source code is also weirdly organized in order to support the article.
All code by Anders Lindqvist unless mentioned otherwise.

Building
========
Run cmake in the root directory.
OpenMP will be used when available for multi-threading.

Dependencies
============
* [PCG random number generator](https://www.pcg-random.org/index.html).
* [STB_image_write](https://github.com/nothings/stb)
* Ray/Sphere and Ray/Cube intersection by Inigo Quilez https://www.iquilezles.org/www/articles/intersectors/intersectors.htm

PCG was modified slightly in order for it to compile on Windows.