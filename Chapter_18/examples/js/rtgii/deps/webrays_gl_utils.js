"use strict";

//
// creates a shader of the given type, uploads the source and compiles it.
//
function wrays_gl_utils_compile_shader(shaderSource, shaderType)
{
  // Create the shader object
  let shader = gl.createShader(shaderType);
     
  // Set the shader source code.
  gl.shaderSource(shader, shaderSource);
     
  // Compile the shader
  gl.compileShader(shader);
     
  // Check if it compiled
  const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (!success) {

    // Delete shader
    const info = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader);
    
    // Something went wrong during compilation; get the error
    throw "An error occurred compiling the shader:" + info;

  }

  wrays_gl_utils_error();

  return shader;
}

//
// create a shader program based on vertex and fragment shaders. 
//
function wrays_gl_utils_create_program(vertexShader, fragmentShader)
{
  if(!vertexShader || !fragmentShader)
    return null;

  // create a program.
  let program = gl.createProgram();
 
  // attach the shaders.
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
 
  // link the program.
  gl.linkProgram(program);
 
  // validate the program.
  //gl.validateProgram(program);

  // Check if it linked.
  var success = gl.getProgramParameter(program, gl.LINK_STATUS);// & gl.getProgramParameter(program, gl.VALIDATE_STATUS);
  if (!success) {

    // Detach shaders
      gl.detachShader(program, vertexShader);
      gl.detachShader(program, fragmentShader);

      // Delete program
      const info = gl.getProgramInfoLog(program)
      gl.deleteProgram(program);

      // something went wrong with the link
      throw ("program failed to link:" + info);

  }
 
  //gl.deleteShader(vertexShader);  // [TODO] Maybe a different program needs them
	//gl.deleteShader(fragmentShader);

  wrays_gl_utils_error();

  return program;
};

//
// create an empty 2D texture. 
//
function wrays_gl_utils_texture_2d_alloc(internal_format, width, height)
{
  let texture = gl.createTexture();

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internal_format, width, height);
  
  wrays_gl_utils_error();

  return texture;
}

//
// create an 2D texture. 
//
function wrays_gl_utils_texture_2d_create(internal_format, width, height, format, filter, type, pixels, has_mipmaps = false)
{
  let texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  if(has_mipmaps)
  {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal_format, width, height, 0, format, type, pixels);
    gl.generateMipmap(gl.TEXTURE_2D);
    /*gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
	  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);*/
  }
  else
  {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
	  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
	  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	  gl.texImage2D(gl.TEXTURE_2D, 0, internal_format, width, height, 0, format, type, pixels);
  }	

  wrays_gl_utils_error();

	return texture;
}

//
// create an 2D texture array.
//
function wrays_gl_utils_texture_2darray_create(internal_format, width, height, layers, format, filter, type, pixels)
{
  let texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
	gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, filter);
	gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, filter);
	gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.REPEAT);
	gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, internal_format, width, height, layers, 0, format, type, pixels);

  wrays_gl_utils_error();

	return texture;
}

//
// create an 2D texture using an Image.
//
function wrays_gl_utils_texture_from_image(image, width, height, has_mipmaps = false)
{
  //document.body.appendChild(image);
  
  let placeholder_canvas    = document.createElement("canvas");
  let placeholder_context   = placeholder_canvas.getContext("2d");
  placeholder_canvas.height = height;
  placeholder_canvas.width  = width;
            
  /* Invert Image inside context
   * Some say that this might be slow
   */
  placeholder_context.translate(0, height);
  placeholder_context.scale(1, -1);
  placeholder_context.drawImage(image, 0, 0, width, height);
  
  let rgba = placeholder_context.getImageData(0, 0, width, height).data;
 
	return wrays_gl_utils_texture_2d_create(gl.RGBA, width, height, gl.RGBA, gl.LINEAR, gl.UNSIGNED_BYTE, rgba, has_mipmaps);
}

//
// create an 2D texture using Images.
//
function wrays_gl_utils_texture_from_images_scaled(images, flip = true) {
  
  //console.log(images);

  const reducer  = (accumulator, currentValue) => Math.max(accumulator, Math.max(currentValue.width, currentValue.height));
  const max_size = Math.min(images.reduce(reducer, 0), 4096);
  //console.log("Max Texture Size: "+max_size);
  const width    = max_size;
  const height   = max_size;
  const wh4      = width * height * 4;

  let placeholder_canvas    = document.createElement("canvas", {width:width+'px', height:height+"px"});
  placeholder_canvas.height = height;
  placeholder_canvas.width  = width;
  let placeholder_context   = placeholder_canvas.getContext("2d");
            
  /* Invert Image inside context
   * Some say that this might be slow
   */
  var rgba = new Uint8ClampedArray(wh4 * images.length );
  var rgba_length = 0;
  if(flip)
  {
    placeholder_context.translate(0, height);
    placeholder_context.scale(1, -1);
  }
  for(let img of images)
  {    
    placeholder_context.drawImage(img, 0, 0, width, height);
    rgba.set(placeholder_context.getImageData(0, 0, width, height).data, rgba_length);
    rgba_length += wh4;
  }
  
	return wrays_gl_utils_texture_2darray_create(gl.RGBA, width, height, images.length, gl.RGBA, gl.LINEAR, gl.UNSIGNED_BYTE, rgba);
}

//
// Create asynchronously query for performance measurement (Double)
//
function wrays_gl_utils_create_double_buffered_timer(ext)
{
  return {
    timers: [gl.createQuery(), gl.createQuery()],
    current: 0,
    firstTime: true
  };
}

//
// Begin asynchronously query for performance measurement (Double)
//
function wrays_gl_utils_begin_double_buffered_timer(ext, query)
{
  gl.beginQuery(ext.TIME_ELAPSED_EXT, query.timers[query.current]);
}

//
// End asynchronously query for performance measurement (Double)
//
function wrays_gl_utils_end_double_buffered_timer(ext, query)
{
  gl.endQuery(ext.TIME_ELAPSED_EXT);
  query.current = (query.current + 1) % 2;
}

//
// Get asynchronously query for performance measurement (Double)
//
function wrays_gl_utils_get_double_buffered_timer(ext, query)
{
  if(query.firstTime)
  {
    query.firstTime = false;
    return 0.0;
  }

  const prev    = (query.current + 0) % 2;
  let available = gl.getQueryParameter(query.timers[prev], gl.QUERY_RESULT_AVAILABLE);
  let disjoint  = gl.getParameter(ext.GPU_DISJOINT_EXT);
  console.log('Disjoint ', available, disjoint);

  //while(!available)
  // available = gl.getQueryParameter(query.timers[prev], gl.QUERY_RESULT_AVAILABLE);

  if (available){// && !disjoint) {
    // See how much time the rendering of the object took in nanoseconds.
    let timeElapsed = gl.getQueryParameter(query.timers[prev], gl.QUERY_RESULT);

    // Do something useful with the time.  Note that care should be
    // taken to use all significant bits of the result, not just the
    // least significant 32 bits.
    return timeElapsed;
    //console.log('DR timeElapsed:', timeElapsed / 1000000.0 );
  }
  return gl.getQueryParameter(query.timers[prev], gl.QUERY_RESULT);
}

//
// Create asynchronously query for performance measurement (Single)
//
function wrays_gl_utils_create_single_buffered_timer(ext)
{
  return {
    timer: gl.createQuery(),
    isSubmitted: false,
    isResolved: true
  };
}

//
// Begin asynchronously query for performance measurement (Single)
//
function wrays_gl_utils_begin_single_buffered_timer(ext, query)
{
  if(query.isResolved)
    gl.beginQuery(ext.TIME_ELAPSED_EXT, query.timer);
}

//
// End asynchronously query for performance measurement (Single)
//
function wrays_gl_utils_end_single_buffered_timer(ext, query)
{
  if(!query.isResolved)
    return;
  
  gl.endQuery(ext.TIME_ELAPSED_EXT);
  query.isSubmitted = true;
  query.isResolved  = false;
}

//
// GEt asynchronously query for performance measurement (Single)
//
function wrays_gl_utils_get_single_buffered_timer(ext, query)
{
  if(!query.isSubmitted)
  {
    return 0.0;
  }
  
  query.isResolved = gl.getQueryParameter(query.timer, gl.QUERY_RESULT_AVAILABLE);
  let disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
  //console.log('Disjoint ', query.isResolved, disjoint);

  if (query.isResolved){// && !disjoint) {
    // See how much time the rendering of the object took in nanoseconds.
    let timeElapsed = gl.getQueryParameter(query.timer, gl.QUERY_RESULT);

    // Do something useful with the time.  Note that care should be
    // taken to use all significant bits of the result, not just the
    // least significant 32 bits.
    return timeElapsed;
    //console.log('DR timeElapsed:', timeElapsed / 1000000.0 );
  }
  return 0.0;
}

//
// Check for WebGL general errors.
//
function wrays_gl_utils_error()
{
  const error = gl.getError();
  if(error != 0) {
    throw ("An WebGL error occured:" + error);
  }
}

//
// Check for WebGL framebuffer errors.
//
function wrays_gl_utils_fbo_error()
{
  const error = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if(error != gl.FRAMEBUFFER_COMPLETE) {
    throw ("An WebGL FBO error occured:" + error);
  }
}
