"use strict";

WebRaysModule['onRuntimeInitialized'] = wrays_viewer_run;

var WebRaysViewer;
if (!WebRaysViewer) WebRaysViewer = (typeof WebRaysViewer !== 'undefined' ? WebRaysViewer : null) || {};

var wr;
var gl;
var gl_timer_ext;

var blas;

function wrays_viewer_run() {
  /* Canvas CSS dimension differ from WebGL viewport dimensions so we
   * make them match here
   * @see https://webglfundamentals.org/webgl/lessons/webgl-resizing-the-canvas.html
   */

  WebRaysViewer.canvas = document.getElementById('webrays-main-canvas');
  WebRaysViewer.canvas.width = WebRaysViewer.canvas.clientWidth;
  WebRaysViewer.canvas.height = WebRaysViewer.canvas.clientHeight;
  WebRaysViewer.context_handle = 0;
  WebRaysViewer.webgl_context_attribs = { 'majorVersion' : 2, 'minorVersion' : 0 };

  /* @see https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/getContext */
  WebRaysViewer.gl = gl = WebRaysViewer.canvas.getContext("webgl2", {
    stencil: false,
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true, /* Needed for thumbnail extraction */
    depth: true
  });

  // If we don't have a GL context, give up now
  // Only continue if WebGL is available and working
  if (!gl) {
    throw ("Unable to initialize WebGL. Your browser or machine may not support it.");
  }

  var ext = gl.getExtension('Ext_color_buffer_float');
  if (!ext) {
    throw ("Unable to initialize WebGL ext: Ext_color_buffer_float.");
  }

  gl_timer_ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');

  /* We need this if we want to initialize the WebGL context from
   * the Javascript side
   */
  WebRaysModule.preinitializedWebGLContext = WebRaysViewer.context;

  WebRaysViewer.context_handle = GL.createContext(WebRaysViewer.canvas, WebRaysViewer.webgl_context_attribs);
  GL.makeContextCurrent(WebRaysViewer.context_handle);

  wrays_viewer_init()
}

function wrays_viewer_resize() {

  // compute new projection matrix
  const fieldOfView = WebRaysViewer.fov * Math.PI / 180;   // in radians
  const aspectRatio = gl.canvas.clientWidth / gl.canvas.clientHeight;
  const aspect      = aspectRatio > 0 ? aspectRatio : gl.canvas.clientWidth / gl.canvas.clientHeight;

  // note: glmatrix.js always has the first argument
  // as the destination to receive the result.
  WebRaysViewer.projection = glMatrix.mat4.perspective(glMatrix.mat4.create(),
                fieldOfView,
                aspect,
                WebRaysViewer.znear, WebRaysViewer.zfar);

  // Initialize Textures

  var buffer_info = {};

  if (gl.isFramebuffer(WebRaysViewer.preview_FBO))
    gl.deleteFramebuffer(WebRaysViewer.preview_FBO);

  if (gl.isFramebuffer(WebRaysViewer.final_FBO))
    gl.deleteFramebuffer(WebRaysViewer.final_FBO);

  if (gl.isTexture(WebRaysViewer.final_texture))
    gl.deleteTexture(WebRaysViewer.final_texture);

  WebRaysViewer.final_texture = wrays_gl_utils_texture_2d_alloc(gl.RGBA32F, gl.canvas.clientWidth, gl.canvas.clientHeight);

  if (gl.isTexture(WebRaysViewer.shadow_ray_texture))
    gl.deleteTexture(WebRaysViewer.shadow_ray_texture);

  if (gl.isFramebuffer(WebRaysViewer.ray_FBOs[0]))
    gl.deleteFramebuffer(WebRaysViewer.ray_FBOs[0]);
  if (gl.isFramebuffer(WebRaysViewer.ray_FBOs[1]))
    gl.deleteFramebuffer(WebRaysViewer.ray_FBOs[1]);

  if (gl.isTexture(WebRaysViewer.ray_directions_textures[0]))
    gl.deleteTexture(WebRaysViewer.ray_directions_textures[0]);
  if (gl.isTexture(WebRaysViewer.ray_directions_textures[1]))
    gl.deleteTexture(WebRaysViewer.ray_directions_textures[1]);

  if (gl.isTexture(WebRaysViewer.ray_origins_textures[0]))
    gl.deleteTexture(WebRaysViewer.ray_origins_textures[0]);
  if (gl.isTexture(WebRaysViewer.ray_origins_textures[1]))
    gl.deleteTexture(WebRaysViewer.ray_origins_textures[1]);

  if (gl.isTexture(WebRaysViewer.ray_accumulation_textures[0]))
    gl.deleteTexture(WebRaysViewer.ray_accumulation_textures[0]);
  if (gl.isTexture(WebRaysViewer.ray_accumulation_textures[1]))
    gl.deleteTexture(WebRaysViewer.ray_accumulation_textures[1]);

    
  buffer_info = wr.IntersectionBufferRequirements([WebRaysViewer.tile_width, WebRaysViewer.tile_height]);
  if (gl.TEXTURE_2D == buffer_info.Target) {
    WebRaysViewer.isect_texture = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
  }

  buffer_info = wr.OcclusionBufferRequirements([WebRaysViewer.tile_width, WebRaysViewer.tile_height]);
  if (gl.TEXTURE_2D == buffer_info.Target) {
    WebRaysViewer.occlusion_texture = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
  }

  buffer_info = wr.RayBufferRequirements([WebRaysViewer.tile_width, WebRaysViewer.tile_height]);
  if (gl.TEXTURE_2D == buffer_info.Target) {
    WebRaysViewer.ray_accumulation_textures[0] = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
    WebRaysViewer.ray_accumulation_textures[1] = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
    WebRaysViewer.ray_payloads_textures[0] = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
    WebRaysViewer.ray_payloads_textures[1] = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
  }

  buffer_info = wr.RayDirectionBufferRequirements([WebRaysViewer.tile_width, WebRaysViewer.tile_height]);
  if (gl.TEXTURE_2D == buffer_info.Target) {
    WebRaysViewer.ray_directions_textures[0] = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
    WebRaysViewer.ray_directions_textures[1] = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);

    WebRaysViewer.shadow_ray_texture = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
  }

  buffer_info = wr.RayOriginBufferRequirements([WebRaysViewer.tile_width, WebRaysViewer.tile_height]);
  if (gl.TEXTURE_2D == buffer_info.Target) {
    WebRaysViewer.ray_origins_textures[0] = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
    WebRaysViewer.ray_origins_textures[1] = wrays_gl_utils_texture_2d_alloc(buffer_info.InternalFormat, buffer_info.Width, buffer_info.Height);
  }

  WebRaysViewer.ray_FBOs[0] = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, WebRaysViewer.ray_FBOs[0]);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, WebRaysViewer.ray_accumulation_textures[0], 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, WebRaysViewer.ray_origins_textures[0], 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, WebRaysViewer.ray_directions_textures[0], 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, WebRaysViewer.ray_payloads_textures[0], 0);
  if (gl.FRAMEBUFFER_COMPLETE != gl.checkFramebufferStatus(gl.FRAMEBUFFER)) {
    console.log('error FBO');
  }

  WebRaysViewer.ray_FBOs[1] = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, WebRaysViewer.ray_FBOs[1]);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, WebRaysViewer.ray_accumulation_textures[1], 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, WebRaysViewer.ray_origins_textures[1], 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, WebRaysViewer.ray_directions_textures[1], 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, WebRaysViewer.ray_payloads_textures[1], 0);
  
  if (gl.FRAMEBUFFER_COMPLETE != gl.checkFramebufferStatus(gl.FRAMEBUFFER)) {
    console.log('error FBO');
  }

  WebRaysViewer.preview_texture = wrays_gl_utils_texture_2d_alloc(gl.RGBA32F, gl.canvas.clientWidth, gl.canvas.clientHeight);

  WebRaysViewer.preview_FBO = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, WebRaysViewer.preview_FBO);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, WebRaysViewer.preview_texture, 0);
  if (gl.FRAMEBUFFER_COMPLETE != gl.checkFramebufferStatus(gl.FRAMEBUFFER)) {
    console.log('error preview FBO');
  }

  WebRaysViewer.final_FBO = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, WebRaysViewer.final_FBO);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, WebRaysViewer.final_texture, 0);
  if (gl.FRAMEBUFFER_COMPLETE != gl.checkFramebufferStatus(gl.FRAMEBUFFER)) {
    console.log('error final FBO');
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function wrays_viewer_wr_scene_init() {
  blas = wr.CreateAds();
  
  const total_faces = WebRaysViewer.mesh.indicesPerMaterial.reduce((accumulator, currentValue) => {return accumulator + currentValue.length / 3}, 0);
  var faces = new Int32Array(total_faces * 4);
  let face_offset = 0;
  const num_materials = WebRaysViewer.mesh.indicesPerMaterial.length;
  for (let mat_index = 0; mat_index < num_materials; mat_index ++) {
    const face_count = WebRaysViewer.mesh.indicesPerMaterial[mat_index].length / 3;
    for (let face_index = 0; face_index < face_count; face_index++) {
      faces[(face_index + face_offset) * 4 + 0] = WebRaysViewer.mesh.indicesPerMaterial[mat_index][face_index * 3 + 0];
      faces[(face_index + face_offset) * 4 + 1] = WebRaysViewer.mesh.indicesPerMaterial[mat_index][face_index * 3 + 1];
      faces[(face_index + face_offset) * 4 + 2] = WebRaysViewer.mesh.indicesPerMaterial[mat_index][face_index * 3 + 2];
      faces[(face_index + face_offset) * 4 + 3] = mat_index;
    }
    face_offset += face_count;
  }

  wr.AddShape(blas, new Float32Array (WebRaysViewer.mesh.vertices), 3,
                    new Float32Array(WebRaysViewer.mesh.vertexNormals), 3,
                    new Float32Array(WebRaysViewer.mesh.textures), 2,
                    new Int32Array(faces));
}

function wrays_viewer_gl_scene_init() {
  
  let vao = null;
  let vbo = null;
  let ibo = null;
  let attr_count = WebRaysViewer.mesh.vertices.length / 3;

  vao = gl.createVertexArray();
  vbo = gl.createBuffer();
  ibo = gl.createBuffer();

  let vertices = [];
  const hasTextures = WebRaysViewer.mesh.textures !== null && WebRaysViewer.mesh.textures.length !== 0;
  for (var attr_index = 0; attr_index < attr_count; attr_index ++) {
    let va = WebRaysViewer.mesh.vertices[attr_index * 3 + 0];
    let vb = WebRaysViewer.mesh.vertices[attr_index * 3 + 1];
    let vc = WebRaysViewer.mesh.vertices[attr_index * 3 + 2];
    let na = WebRaysViewer.mesh.vertexNormals[attr_index * 3 + 0];
    let nb = WebRaysViewer.mesh.vertexNormals[attr_index * 3 + 1];
    let nc = WebRaysViewer.mesh.vertexNormals[attr_index * 3 + 2];
    let ta = (hasTextures)? WebRaysViewer.mesh.textures[attr_index * 2 + 0] : 0.0;
    let tb = (hasTextures)? WebRaysViewer.mesh.textures[attr_index * 2 + 1] : 0.0;

    vertices.push(va); vertices.push(vb); vertices.push(vc);
    vertices.push(na); vertices.push(nb); vertices.push(nc);
    vertices.push(ta); vertices.push(tb);
  }

  let parts = [];
  const num_materials = WebRaysViewer.mesh.indicesPerMaterial.length;
  let indices_offset = 0;
  for (let mat_index = 0; mat_index < num_materials; mat_index++) {
    let off = WebRaysViewer.mesh.indicesPerMaterial[mat_index][0];
    off = indices_offset;
    let len = WebRaysViewer.mesh.indicesPerMaterial[mat_index].length / 3;
    indices_offset += len;
    parts.push({offset: off, count: len}); 
  }

  /* 3 * vec4 
  {
    type : 0 - perfect specular, 1 - fresnel specular, 2 - CT GGX, 3 - Lambert, 
    baseColor.rgb

    nIndex: index of refraction
    metallic:
    roughness: 
    reflectance: [0 1] translates to [0 0.16]

    baseColorTexture: -1 not exists
    MetallicRoughnessTexture: -1 not exists
    NormalsTexture: -1 not exists
    UnusedTexture: -1 not exists
  }*/
 
  gl.bindVertexArray(vao);
          
  gl.enableVertexAttribArray(0);
  gl.enableVertexAttribArray(1);
  gl.enableVertexAttribArray(2);
  
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  let indices = WebRaysViewer.mesh.indicesPerMaterial.flat();
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Int32Array(indices), gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

  let vertex_position_offset = 0 * Float32Array.BYTES_PER_ELEMENT;
  let vertex_normal_offset   = 3 * Float32Array.BYTES_PER_ELEMENT;
  let vertex_texcoord_offset = 6 * Float32Array.BYTES_PER_ELEMENT;
  let vertex_stride          = 8 * Float32Array.BYTES_PER_ELEMENT;

  gl.vertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, vertex_stride, vertex_position_offset);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, gl.FALSE, vertex_stride, vertex_normal_offset);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, gl.FALSE, vertex_stride, vertex_texcoord_offset);

  WebRaysViewer.gl_mesh = {vao: vao, vbo: vbo, ibo: ibo, attr_count: attr_count, parts: parts};

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  /* GL Programs */
  var vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.accum_program_vs_source, gl.VERTEX_SHADER);
  var fragment_shader = wrays_gl_utils_compile_shader(WebRaysViewer.accum_program_fs_source, gl.FRAGMENT_SHADER);
  WebRaysViewer.accum_program = wrays_gl_utils_create_program(vertex_shader, fragment_shader);

  vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.pp_program_vs_source, gl.VERTEX_SHADER);
  fragment_shader = wrays_gl_utils_compile_shader(WebRaysViewer.pp_program_fs_source, gl.FRAGMENT_SHADER);
  WebRaysViewer.pp_program = wrays_gl_utils_create_program(vertex_shader, fragment_shader);

  wrays_viewer_gl_scene_lights_init();
  wrays_viewer_gl_scene_materials_init();
}

function wrays_viewer_gl_scene_materials_init() {
  let materialTexture = gl.createTexture();
  let num_materials = WebRaysViewer.materials.length / 12;
  gl.bindTexture(gl.TEXTURE_2D, materialTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 3, num_materials, 0, gl.RGBA, gl.FLOAT, new Float32Array(WebRaysViewer.materials));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.bindTexture(gl.TEXTURE_2D, null);

  // create the textures
  const resource_urls = WebRaysViewer.textures;
  Promise.all(resource_urls.map(u=>fetch(u))).then(responses => {
    //var decoded_imgs = [];
    for (let rsrc_index = 0; rsrc_index < responses.length; rsrc_index++) {
      if (false == responses[rsrc_index].ok) {
        return;
      }
    }

    Promise.all(responses.map(res => res.blob())).then(buffers => {
      let decoded_imgs = [];
      for (let rsrc_index = 0; rsrc_index < buffers.length; rsrc_index++) {
        let image = document.createElement('img');
        image.src = URL.createObjectURL(buffers[rsrc_index]);
        decoded_imgs.push(image);
      }
      Promise.allSettled(buffers.map(blob=>blob.arrayBuffer())).then(responses => {
        let imgs = [];
        for (let rsrc_index = 0; rsrc_index < responses.length; rsrc_index++) {
          const img = UPNG.decode(responses[rsrc_index].value);
          imgs.push(img);
        }

        const reducer  = (accumulator, currentValue) => Math.max(accumulator, Math.max(currentValue.width, currentValue.height));
        const max_size = Math.min(imgs.reduce(reducer, 0), 4096);
        const width    = max_size;
        const height   = max_size;
        const wh4      = width * height * 4;
        var rgba = new Uint8ClampedArray(wh4 * imgs.length);
        var rgba_length = 0;

        for (let rsrc_index = 0; rsrc_index < imgs.length; rsrc_index++) {
          const img = imgs[rsrc_index];
          const img_data = img.data;
          var   img_rgba = new Uint8ClampedArray(wh4);

          var channels = 3;
          if (img.ctype == 6) {
            channels = 4;
          }
          for (var i = 0; i < width; i++) {
            for (var j = 0; j < height; j++) {
              var index3 = ((height - j - 1) * width + i) * channels;
              var index4 = (j * width + i) * 4;
              img_rgba[index4 + 0] =                   img_data[index3 + 0];
              img_rgba[index4 + 1] =                   img_data[index3 + 1];
              img_rgba[index4 + 2] =                   img_data[index3 + 2];
              img_rgba[index4 + 3] = (channels == 4) ? img_data[index3 + 3] : 0;
            }
          }
          rgba.set(img_rgba, rgba_length);
          rgba_length += wh4;
        }

        WebRaysViewer.texturesBuffer = wrays_gl_utils_texture_2darray_create(gl.RGBA, width, height, imgs.length, gl.RGBA, gl.LINEAR, gl.UNSIGNED_BYTE, rgba);

        return;
      });
    });
  }); 

  WebRaysViewer.materialTexture = materialTexture;
}

function flatten(ary) {
  return ary.reduce(function(a, b) {
    if (Array.isArray(b) || ArrayBuffer.isView(b)) {
      return a.concat(flatten(b))
    }
    return a.concat(b)
  }, [])
}

function wrays_viewer_gl_scene_lights_init() {
  const light_count = WebRaysViewer.lights.length;
  for (let light_index = 0; light_index < light_count; light_index++) {
    let light = WebRaysViewer.lights[light_index];
    let vao = null;
    let vbo = null;
    let ibo = null;
    let attr_count = light.vertices.length / 3;
    let face_count = light.faces.length / 3;
  
    vao = gl.createVertexArray();
    vbo = gl.createBuffer();
    ibo = gl.createBuffer();
  
    let vertices = [];
    const hasTextures = WebRaysViewer.mesh.textures !== null && WebRaysViewer.mesh.textures.length !== 0;
    for (var attr_index = 0; attr_index < attr_count; attr_index ++) {
      let va = light.vertices[attr_index * 3 + 0];
      let vb = light.vertices[attr_index * 3 + 1];
      let vc = light.vertices[attr_index * 3 + 2];
      let na = light.normals[attr_index * 3 + 0];
      let nb = light.normals[attr_index * 3 + 1];
      let nc = light.normals[attr_index * 3 + 2];
      let ta = light.uvs[attr_index * 2 + 0];
      let tb = light.uvs[attr_index * 2 + 1];
  
      vertices.push(va); vertices.push(vb); vertices.push(vc);
      vertices.push(na); vertices.push(nb); vertices.push(nc);
      vertices.push(ta); vertices.push(tb);
    }
    let parts = [{offset: 0, count: 2}];
    
    gl.bindVertexArray(vao);
            
    gl.enableVertexAttribArray(0);
    gl.enableVertexAttribArray(1);
    gl.enableVertexAttribArray(2);
    
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Int32Array(light.faces), gl.STATIC_DRAW);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Int32Array(light.faces), gl.STATIC_DRAW);
  
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  
    let vertex_position_offset = 0 * Float32Array.BYTES_PER_ELEMENT;
    let vertex_normal_offset   = 3 * Float32Array.BYTES_PER_ELEMENT;
    let vertex_texcoord_offset = 6 * Float32Array.BYTES_PER_ELEMENT;
    let vertex_stride          = 8 * Float32Array.BYTES_PER_ELEMENT;
  
    gl.vertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, vertex_stride, vertex_position_offset);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, gl.FALSE, vertex_stride, vertex_normal_offset);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, gl.FALSE, vertex_stride, vertex_texcoord_offset);
  
    light.gl_mesh = {vao: vao, vbo: vbo, ibo: ibo, attr_count: attr_count, face_count: face_count };

    /*parts: parts, materialTexture: materialTexture, texturesBuffer: textures};*/
    light.gl_light = {
      position: glMatrix.vec4.fromValues(light.position[0], light.position[1], light.position[2], light.type + 0.5),
      power: glMatrix.vec4.fromValues(light.power[0], light.power[1], light.power[2], 0),
      up: glMatrix.vec4.fromValues(0, 2, 0, 0),
      right: glMatrix.vec4.fromValues(2, 0, 0, 0)
    };
  }

  let gl_lights_flat = [];
  for (let light_index = 0; light_index < light_count; light_index++) {
    const light = WebRaysViewer.lights[light_index];
    const gl_light_flat = flatten(Object.values(light.gl_light));
    let gl_light_vec4_count = gl_light_flat.length / 4;
    gl_lights_flat = gl_lights_flat.concat(gl_light_flat);
  }

  let gl_lights_flat_f32 = new Float32Array(gl_lights_flat);
  let gl_light_vec4_count = gl_lights_flat_f32.length / (light_count * 4);

  let lightTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, lightTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, gl_light_vec4_count, light_count, 0, gl.RGBA, gl.FLOAT,gl_lights_flat_f32);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  WebRaysViewer.lightTexture = lightTexture;

  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
}

function wrays_viewer_scene_lights_init() {
  let vertices = new Float32Array ([ -1, -1, -1,  1, -1, -1, 1,  1, -1, -1,  1, -1 ]);
  let normals = new Float32Array([ 0, 0, 1,  0, 0, 1, 0, 0, 1, 0, 0, 1 ]);
  let uvs = new Float32Array([ 0, 0, 1, 0, 1, 1, 0, 1 ]);
  let faces = new Int32Array([ 0, 1, 2, 0, 2, 3 ]);
  let attr_count = 4;
  let face_count = 2;
  
  WebRaysViewer.lights.push({
    type: 1,
    position: glMatrix.vec3.fromValues(1.4573, 2.43, 2.56),
    power: glMatrix.vec3.fromValues(0, 0, 0),
    faces: faces,
    normals: normals,
    uvs: uvs,
    vertices: vertices
  });
}

function wrays_viewer_scene_init() {
  wrays_viewer_scene_lights_init();

  wrays_viewer_wr_scene_init();
  wrays_viewer_gl_scene_init();
}

function wrays_viewer_framebuffer_init() {
  WebRaysViewer.vao = gl.createVertexArray();
  gl.bindVertexArray(WebRaysViewer.vao);
  var screen_fill_triangle = new Float32Array(
                          [ -4.0, -4.0, 0.0,
                             4.0, -4.0, 0.0, 
                             0.0, 4.0, 0.0 ]
                          );
  
  WebRaysViewer.vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, WebRaysViewer.vbo);
  gl.bufferData(gl.ARRAY_BUFFER, screen_fill_triangle, gl.STATIC_DRAW);  
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); 
 
  wrays_viewer_resize();

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function wrays_viewer_key_down_event(key) {
  const code = key.keyCode;
  switch (code) {
    case 65:  glMatrix.vec3.sub(WebRaysViewer.camera_pos, WebRaysViewer.camera_pos, WebRaysViewer.camera_right); WebRaysViewer.frame_counter = 0; break;
    case 87:  glMatrix.vec3.add(WebRaysViewer.camera_pos, WebRaysViewer.camera_pos, WebRaysViewer.camera_front); WebRaysViewer.frame_counter = 0; break;
    case 83:  glMatrix.vec3.sub(WebRaysViewer.camera_pos, WebRaysViewer.camera_pos, WebRaysViewer.camera_front);WebRaysViewer.frame_counter = 0;  break;
    case 68:  glMatrix.vec3.add(WebRaysViewer.camera_pos, WebRaysViewer.camera_pos, WebRaysViewer.camera_right); WebRaysViewer.frame_counter = 0; break;
    case 37:  break; //Left key
    case 38:  break; //Up key
    case 39:  break; //Right key
    case 40:  break; //Down key
    default:  break; //Everything else
  }

  const at = glMatrix.vec3.add(glMatrix.vec3.create(), WebRaysViewer.camera_front, WebRaysViewer.camera_pos);
  WebRaysViewer.view = glMatrix.mat4.lookAt( 
    glMatrix.mat4.create(),
    WebRaysViewer.camera_pos, // pos
    at, // at
    WebRaysViewer.camera_up // up
  ); // view is [right, up, forward, -pos]^T;
}

function wrays_viewer_touch_start_event(touch) {
  WebRaysViewer.frame_counter = 0;
  WebRaysViewer.arcball_on = true;
  WebRaysViewer.mouse = WebRaysViewer.last_mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
}

function wrays_viewer_touch_end_event(touch) {
  WebRaysViewer.arcball_on = false;
}

function wrays_viewer_touch_move_event(touch) {
  WebRaysViewer.frame_counter = 0;
  if (WebRaysViewer.arcball_on) {
    WebRaysViewer.depth = 3;
    WebRaysViewer.mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
	}
}
  
function wrays_viewer_mouse_move_event(mouse) {
  if (WebRaysViewer.arcball_on) {
    WebRaysViewer.frame_counter = 0;
    WebRaysViewer.depth = 2;
    WebRaysViewer.mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
	}
}

function wrays_viewer_mouse_up_event(mouse) {
  switch( mouse.button ) {
    case 0:
      WebRaysViewer.arcball_on = false;
      WebRaysViewer.depth = 3;
      break;
    case 1:
      break;
    case 2:
      break;  
    default:
      break;
  }
}

function wrays_viewer_mouse_down_event(mouse) {
  switch( mouse.button ) {
    case 0:
      WebRaysViewer.arcball_on = true;
			WebRaysViewer.mouse = WebRaysViewer.last_mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
      break;
    case 1:
      break;
    case 2:
      break;
    default:
      break;
  }
}

function wrays_viewer_init() {
  /* Initialize things (This could very well be a constructor) */
  WebRaysViewer.mouse           = glMatrix.vec2.fromValues(0, 0);
  WebRaysViewer.last_mouse      = glMatrix.vec2.fromValues(0, 0);
  WebRaysViewer.arcball_on      = false;

  // Viewer Properties
  WebRaysViewer.tile_width    = 256;
  WebRaysViewer.tile_height   = 256;
  WebRaysViewer.depth         = 3;
  WebRaysViewer.frame_counter = 1;

  // Camera Properties
  WebRaysViewer.fov         = 90.0;
  WebRaysViewer.znear       = 0.05;
  WebRaysViewer.zfar        = 100.0;
  WebRaysViewer.camera_pos  = glMatrix.vec3.fromValues(3.0, 1.5, 6.0);

  // View Matrix
  WebRaysViewer.view = glMatrix.mat4.lookAt( 
    glMatrix.mat4.create(),
    WebRaysViewer.camera_pos, // pos
    glMatrix.vec3.fromValues( 0, 1.5, -0.4 ), // at
    glMatrix.vec3.fromValues( 0, 1, 0 ) // up
  );
  
  // Decode Camera View Axis
  WebRaysViewer.camera_right = glMatrix.vec3.fromValues( WebRaysViewer.view[0], WebRaysViewer.view[4], WebRaysViewer.view[8] );
  WebRaysViewer.camera_up    = glMatrix.vec3.fromValues( WebRaysViewer.view[1], WebRaysViewer.view[5], WebRaysViewer.view[9] );
  WebRaysViewer.camera_front = glMatrix.vec3.fromValues( -WebRaysViewer.view[2], -WebRaysViewer.view[6], -WebRaysViewer.view[10] );
  
  // FrameBuffer
  WebRaysViewer.preview_FBO = null;
  WebRaysViewer.preview_texture = null;
  WebRaysViewer.final_FBO = null;
  WebRaysViewer.final_texture = null;

  WebRaysViewer.shadow_ray_texture = null;
  WebRaysViewer.ray_FBOs = [null, null];
  WebRaysViewer.ray_directions_textures = [null, null];
  WebRaysViewer.ray_origins_textures = [null, null];
  WebRaysViewer.ray_accumulation_textures = [null, null];
  WebRaysViewer.ray_payloads_textures = [null, null];
  WebRaysViewer.isect_texture = null;
  WebRaysViewer.occlusion_texture = null;

  // Textures
  WebRaysViewer.env_map_texture = null;

  // Rasterization
  WebRaysViewer.fbo = null;
  WebRaysViewer.vao = null;
  WebRaysViewer.vbo = null;
  WebRaysViewer.texture = null;

  /* Program sources */
  WebRaysViewer.isect_program_fs_source = null;
  WebRaysViewer.isect_program_vs_source = null;

  WebRaysViewer.accum_program_fs_source = null;
  WebRaysViewer.accum_program_vs_source = null;

  /* GL Program handles */
  WebRaysViewer.pp_program = null;
  WebRaysViewer.generate_program = null;
  WebRaysViewer.isect_program = null;
  WebRaysViewer.accum_program = null;

  WebRaysViewer.mesh = null;
  WebRaysViewer.gl_mesh = null;
  WebRaysViewer.textures = [];
  WebRaysViewer.lightTexture = null;
  WebRaysViewer.materialTexture = null;
  WebRaysViewer.texturesBuffer = null;
  
  WebRaysViewer.enable_timers = false;

  WebRaysViewer.timer = null;
  if(gl_timer_ext)
      wrays_gl_utils_create_single_buffered_timer();

  /* Light System */
  WebRaysViewer.lights = [];
  
  /* Resources to be loaded async */
  WebRaysViewer.shader_urls = [
                               'kernels/screen_fill.vs.glsl', 'kernels/rtgii/wrays_viewer_rtg_ss_generate.glsl',
                               'kernels/screen_fill.vs.glsl', 'kernels/rtgii/wrays_viewer_rtg_ss_intersection_preview.glsl',
                               'kernels/screen_fill.vs.glsl', 'kernels/rtgii/wrays_viewer_rtg_ss_accumulate.glsl',
                               'kernels/screen_fill.vs.glsl', 'kernels/rtgii/wrays_viewer_rtg_ss_post_process.glsl',
                              ];

  /* Set Viewer callbacks */
  var call_during_capture = true;
  WebRaysViewer.canvas.addEventListener("mousemove", wrays_viewer_mouse_move_event, call_during_capture);
  WebRaysViewer.canvas.addEventListener("mouseup", wrays_viewer_mouse_up_event, call_during_capture);
  WebRaysViewer.canvas.addEventListener("mousedown", wrays_viewer_mouse_down_event, call_during_capture);
  WebRaysViewer.canvas.addEventListener("touchstart", wrays_viewer_touch_start_event, false);
  WebRaysViewer.canvas.addEventListener("touchend", wrays_viewer_touch_end_event, false);
  //WebRaysViewer.canvas.addEventListener("touchcancel", handleCancel, false);
  WebRaysViewer.canvas.addEventListener("touchmove", wrays_viewer_touch_move_event, false);
  window.addEventListener('keydown', wrays_viewer_key_down_event, call_during_capture);

  /* Async shader load */
  Promise.all(WebRaysViewer.shader_urls.map(url =>
    fetch(url, {cache: "no-store"}).then(resp => resp.text())
  )).then(texts => {
    var vertex_shader = wrays_gl_utils_compile_shader(texts[0], gl.VERTEX_SHADER);
    var fragment_shader = wrays_gl_utils_compile_shader(texts[1], gl.FRAGMENT_SHADER);
    WebRaysViewer.generate_program = wrays_gl_utils_create_program(vertex_shader, fragment_shader);

    WebRaysViewer.isect_program_vs_source = texts[2];
    WebRaysViewer.isect_program_fs_source = texts[3];

    WebRaysViewer.accum_program_vs_source = texts[4];
    WebRaysViewer.accum_program_fs_source = texts[5];          
      
    WebRaysViewer.pp_program_vs_source = texts[6];
    WebRaysViewer.pp_program_fs_source = texts[7];

    wr = new WebRays.WebGLIntersectionEngine();

    webrays_load_obj("obj/spacestation/Spacestation.obj", "obj/spacestation/Spacestation.mtl").then(res => {
       if(res === null)
         throw new Error('Loading mesh FAILED.');
       WebRaysViewer.mesh = res[0];
       const materials = res[1].mats;
       const textures = res[1].texures;
       for (var texture_index = 0; texture_index < textures.length; texture_index++) {
        WebRaysViewer.textures.push('obj/spacestation/' + textures[texture_index]);      

       }
       let materials_flat = [];
       for (var material_index = 0; material_index < materials.length; material_index++) {
         const material = materials[material_index];
         materials_flat = materials_flat.concat([
          3,    material.baseColor[0],   material.baseColor[1],   material.baseColor[2],
          1.52, 0,   0, 0.5,
          material.baseColorTexture + 0.5,   material.MetallicRoughnessTexture + 0.5,  material.NormalsTexture + 0.5,  -1
         ]);
       }
       WebRaysViewer.materials = materials_flat;
       wrays_viewer_framebuffer_init();
       wrays_viewer_scene_init();

       // Begin rendering 
       requestAnimationFrame(wrays_viewer_render);
     }).catch(error => {
       console.log("Error ", error);
     });
  });                     
}

function wrays_radians(a) {
  const degree = Math.PI / 180;
  return a * degree;
}

function wrays_viewer_camera_update() {
  if (WebRaysViewer.mouse[0] == WebRaysViewer.last_mouse[0] && 
      WebRaysViewer.mouse[1] == WebRaysViewer.last_mouse[1])
    return;
    
  var delta = glMatrix.vec2.fromValues(1, 1);
  const mouse_delta = glMatrix.vec2.sub(glMatrix.vec2.create(), WebRaysViewer.last_mouse, WebRaysViewer.mouse);
  delta = glMatrix.vec2.mul(glMatrix.vec2.create(), delta, mouse_delta);
  delta[0] *= 0.004; // NEED to multiply with dt
  delta[1] *= 0.002;// NEED to multiply with dt

  WebRaysViewer.last_mouse[0] = WebRaysViewer.mouse[0];
  WebRaysViewer.last_mouse[1] = WebRaysViewer.mouse[1];
        
  let camera_at = glMatrix.vec3.clone(WebRaysViewer.camera_front);
  let rot_x = glMatrix.mat4.fromRotation(glMatrix.mat4.create(), delta[0], WebRaysViewer.camera_up);
  let rot_y = glMatrix.mat4.fromRotation(glMatrix.mat4.create(), delta[1], WebRaysViewer.camera_right);
  let rot = glMatrix.mat4.multiply(glMatrix.mat4.create(), rot_x, rot_y);
  
  camera_at = glMatrix.vec3.transformMat4(glMatrix.vec3.create(), camera_at, rot);
    
  camera_at = glMatrix.vec3.add(glMatrix.vec3.create(), camera_at, WebRaysViewer.camera_pos);
  WebRaysViewer.view = glMatrix.mat4.lookAt( 
    glMatrix.mat4.create(),
    WebRaysViewer.camera_pos, // pos
    camera_at, // at
    WebRaysViewer.camera_up // up
  ); // view is [right, up, -forward, -pos]^T;

  WebRaysViewer.camera_right = glMatrix.vec3.fromValues( WebRaysViewer.view[0], WebRaysViewer.view[4], WebRaysViewer.view[8] );
  WebRaysViewer.camera_up = glMatrix.vec3.fromValues( WebRaysViewer.view[1], WebRaysViewer.view[5], WebRaysViewer.view[9] );
  WebRaysViewer.camera_front = glMatrix.vec3.fromValues( -WebRaysViewer.view[2], -WebRaysViewer.view[6], -WebRaysViewer.view[10] );
    
  //WebRaysViewer.last_mouse = WebRaysViewer.mouse;
}

function wrays_viewer_update() {
  wrays_viewer_camera_update();

  var update_flags = wr.Update();
  if(0 == update_flags)
    return;

  var scene_accessor_string = wr.GetSceneAccessorString();

  var fragment_source = scene_accessor_string + WebRaysViewer.isect_program_fs_source;
  var vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.isect_program_vs_source, gl.VERTEX_SHADER);
  var fragment_shader = wrays_gl_utils_compile_shader(fragment_source, gl.FRAGMENT_SHADER);
  WebRaysViewer.isect_program = wrays_gl_utils_create_program(vertex_shader, fragment_shader);
}

function wrays_viewer_post_process(texture) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, WebRaysViewer.canvas.width, WebRaysViewer.canvas.height);

  gl.useProgram(WebRaysViewer.pp_program);

  let index = gl.getUniformLocation(WebRaysViewer.pp_program, "accumulated_texture");

  gl.activeTexture(gl.TEXTURE0 + 0);
	gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(index, 0);

  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function wrays_viewer_accumulate(tile_index_x, tile_index_y, texture) {
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, WebRaysViewer.preview_FBO);

  gl.viewport(0, 0, WebRaysViewer.canvas.width, WebRaysViewer.canvas.height);
      
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.useProgram(WebRaysViewer.accum_program);

  let index = gl.getUniformLocation(WebRaysViewer.accum_program, "blend_factor");
  gl.uniform1f(index, 1.0 / WebRaysViewer.frame_counter);
  index = gl.getUniformLocation(WebRaysViewer.accum_program, "accumulation_texture");
  gl.activeTexture(gl.TEXTURE0 + 0);
	gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(index, 0);

  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.disable(gl.BLEND);
}

function wrays_viewer_blit(tile_index_x, tile_index_y, src_fbo) {
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, WebRaysViewer.final_FBO);
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, src_fbo);

  gl.readBuffer(gl.COLOR_ATTACHMENT0);

  var dst_viewport = [
    (tile_index_x + 0) * WebRaysViewer.tile_width,
    (tile_index_y + 0) * WebRaysViewer.tile_height,
    Math.min((tile_index_x + 1) * WebRaysViewer.tile_width, WebRaysViewer.canvas.width),
    Math.min((tile_index_y + 1) * WebRaysViewer.tile_height, WebRaysViewer.canvas.height)
  ];
  var src_viewport = [
    0,0,
    Math.min(WebRaysViewer.tile_width, WebRaysViewer.tile_width + WebRaysViewer.canvas.width - (tile_index_x + 1) * WebRaysViewer.tile_width),
    Math.min(WebRaysViewer.tile_height, WebRaysViewer.tile_height + WebRaysViewer.canvas.height - (tile_index_y + 1) * WebRaysViewer.tile_height)
  ];

  gl.blitFramebuffer(
    src_viewport[0], src_viewport[1], src_viewport[2], src_viewport[3],
    dst_viewport[0], dst_viewport[1], dst_viewport[2], dst_viewport[3],
    gl.COLOR_BUFFER_BIT, gl.NEAREST);
}

function wrays_viewer_isects_handle(tile_index_x, tile_index_y, depth) {
  const ray_directions    = WebRaysViewer.ray_directions_textures[(depth + 0) & 1];
  const ray_accumulations = WebRaysViewer.ray_accumulation_textures[(depth + 0) & 1];
  const ray_payloads      = WebRaysViewer.ray_payloads_textures[(depth + 0) & 1];
  const ray_origins       = WebRaysViewer.ray_origins_textures[(depth + 0) & 1];
  const ray_fbo           = WebRaysViewer.ray_FBOs[(depth + 1) & 1];
  const ray_intersections = WebRaysViewer.isect_texture;
  const ray_occlusions    = WebRaysViewer.occlusion_texture;
   
  //wr.QueryIntersection([ray_origins, ray_directions], ray_intersections, [WebRaysViewer.tile_width, WebRaysViewer.tile_height]);

  gl.bindFramebuffer(gl.FRAMEBUFFER, ray_fbo);
  gl.drawBuffers([ gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3 ]);

  gl.viewport(0, 0, WebRaysViewer.tile_width, WebRaysViewer.tile_height);
      
  gl.useProgram(WebRaysViewer.isect_program);

  var index = gl.getUniformLocation(WebRaysViewer.isect_program, "tile");
  gl.uniform4iv(index, [tile_index_x, tile_index_y, WebRaysViewer.tile_width, WebRaysViewer.tile_height]);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "frame");
  gl.uniform2iv(index, [WebRaysViewer.canvas.width, WebRaysViewer.canvas.height]);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "seed");
  gl.uniform2uiv(index, random_uvec2());
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "sample_index");
  gl.uniform1i(index,  WebRaysViewer.frame_counter);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "depth");
  gl.uniform1i(index,  depth);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "ads");
  gl.uniform1i(index, blas);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "light_count");
  gl.uniform1i(index, WebRaysViewer.lights.length);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "ray_directions");
  gl.activeTexture(gl.TEXTURE0 + 0);
	gl.bindTexture(gl.TEXTURE_2D, ray_directions);
  gl.uniform1i(index, 0);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "ray_origins");
  gl.activeTexture(gl.TEXTURE0 + 1);
	gl.bindTexture(gl.TEXTURE_2D, ray_origins);
  gl.uniform1i(index, 1);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "env_map");
  gl.activeTexture(gl.TEXTURE0 + 2);
	gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.env_map_texture);
  gl.uniform1i(index, 2);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "intersections");
  gl.activeTexture(gl.TEXTURE0 + 3);
	gl.bindTexture(gl.TEXTURE_2D, ray_intersections);
  gl.uniform1i(index, 3);
  // Material Properties
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "u_materialBuffer");
  gl.activeTexture(gl.TEXTURE0 + 4);
  gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.materialTexture);
  gl.uniform1i(index, 4);
  // Object Textures
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "u_texturesBuffer");
  gl.activeTexture(gl.TEXTURE0 + 5);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, WebRaysViewer.texturesBuffer);
  gl.uniform1i(index, 5);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "u_lightBuffer");
  gl.activeTexture(gl.TEXTURE0 + 6);
  gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.lightTexture);
  gl.uniform1i(index, 6);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "ray_accumulations");
  gl.activeTexture(gl.TEXTURE0 + 7);
  gl.bindTexture(gl.TEXTURE_2D, ray_accumulations);
  gl.uniform1i(index, 7);
  index = gl.getUniformLocation(WebRaysViewer.isect_program, "ray_payloads");
  gl.activeTexture(gl.TEXTURE0 + 8);
  gl.bindTexture(gl.TEXTURE_2D, ray_payloads);
  gl.uniform1i(index, 8);
  var bindings = wr.Bindings;
  var next_texture_unit = 9;
  for (var binding_index = 0; binding_index < bindings.length; ++binding_index) {		
    var binding = bindings[binding_index];

    /* if UBO */
    if (binding.Type == 1) {
      
    /* if Texture 2D */
    } else if (binding.Type == 2) {
      index = gl.getUniformLocation(WebRaysViewer.isect_program, binding.Name);
      gl.activeTexture(gl.TEXTURE0 + next_texture_unit);
      gl.bindTexture(gl.TEXTURE_2D, binding.Texture);
      gl.uniform1i(index, next_texture_unit);
      next_texture_unit++;
    /* if Texture Array 2D */
    } else if (binding.Type == 3) {
      index = gl.getUniformLocation(WebRaysViewer.isect_program, binding.Name);
      gl.activeTexture(gl.TEXTURE0 + next_texture_unit);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, binding.Texture);
      gl.uniform1i(index, next_texture_unit);
      next_texture_unit++;
    }
  }

  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function wrays_viewer_rays_generate(tile_index_x, tile_index_y, fbo) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  
  gl.drawBuffers([ gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3 ]);
  gl.viewport(0, 0, WebRaysViewer.tile_width, WebRaysViewer.tile_height);

  gl.useProgram(WebRaysViewer.generate_program);

  var index = gl.getUniformLocation(WebRaysViewer.generate_program, "tile");
  gl.uniform4iv(index, [tile_index_x, tile_index_y, WebRaysViewer.tile_width, WebRaysViewer.tile_height]);
  index = gl.getUniformLocation(WebRaysViewer.generate_program, "frame");
  gl.uniform2iv(index, [WebRaysViewer.canvas.width, WebRaysViewer.canvas.height]);
  index = gl.getUniformLocation(WebRaysViewer.generate_program, "seed");
  gl.uniform2uiv(index, random_uvec2());
  index = gl.getUniformLocation(WebRaysViewer.generate_program, "camera_pos");
  gl.uniform3fv(index, WebRaysViewer.camera_pos);
  index = gl.getUniformLocation(WebRaysViewer.generate_program, "camera_up");
  gl.uniform3fv(index, WebRaysViewer.camera_up);
  index = gl.getUniformLocation(WebRaysViewer.generate_program, "camera_front");
  gl.uniform3fv(index, WebRaysViewer.camera_front);
  index = gl.getUniformLocation(WebRaysViewer.generate_program, "camera_right");
  gl.uniform3fv(index, WebRaysViewer.camera_right);
  
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function wrays_viewer_render_rt(now) {
  var tile_count_x = ((WebRaysViewer.canvas.width - 1) / WebRaysViewer.tile_width) + 1;
	var tile_count_y = ((WebRaysViewer.canvas.height - 1) / WebRaysViewer.tile_height) + 1;
 
  gl.bindVertexArray(WebRaysViewer.vao);

  WebRaysViewer.frame_counter += 1;
  
  if(WebRaysViewer.enable_timers && gl_timer_ext)
  {
    wrays_gl_utils_begin_single_buffered_timer(gl_timer_ext, WebRaysViewer.timer);
  }
  let rays = {directions: WebRaysViewer.ray_directions_textures, origins: WebRaysViewer.ray_origins_textures};
  let isects = WebRaysViewer.isect_texture;
  let targets = WebRaysViewer.ray_FBOs;
  let target = null;
  for (var tile_index_x = 0; tile_index_x < tile_count_x; tile_index_x++) {
		for (var tile_index_y = 0; tile_index_y < tile_count_y; tile_index_y++) {
      target = targets[0];
      wrays_viewer_rays_generate(tile_index_x, tile_index_y, target);

      for (var depth = 0; depth < WebRaysViewer.depth; depth++  ) {
        target = targets[(depth + 1) & 1];
        const directions    = rays.directions[depth & 1];
        const origins       = rays.origins[depth & 1];
        const intersections = isects;
         
        wr.QueryIntersection([origins, directions], intersections, [WebRaysViewer.tile_width, WebRaysViewer.tile_height]);

        wrays_viewer_isects_handle(tile_index_x, tile_index_y, depth);
      }

      wrays_viewer_blit(tile_index_x, tile_index_y, target);

    }
  }

  wrays_viewer_accumulate(tile_index_x, tile_index_y, WebRaysViewer.final_texture);

  wrays_viewer_post_process(WebRaysViewer.preview_texture);

  if(WebRaysViewer.enable_timers && gl_timer_ext)
  {
    wrays_gl_utils_end_single_buffered_timer(gl_timer_ext, WebRaysViewer.timer);
    const ellapsedTime = wrays_gl_utils_get_single_buffered_timer(gl_timer_ext, WebRaysViewer.timer) / 1000000.0; // ms
    if(ellapsedTime !== 0.0)
    console.log('RT :'+ellapsedTime+' ms');
  
  }
}

function randomIntInc(low, high) {
  return Math.floor(Math.random() * (high - low + 1) + low)
}

function random_uvec2() {
  var u1 = randomIntInc(0, 4294967295);
  var u2 = randomIntInc(0, 4294967295);
  //return new Uint32Array([u1, u2]);
  //return [Math.floor(Math.random() * Math.pow(2, 32)), Math.floor(Math.random() * Math.pow(2, 32))];
  return [Math.floor(Math.random() * 10000), Math.floor(Math.random() * 10000)];
}

function wrays_viewer_render(now) {
  wrays_viewer_update();

  wrays_viewer_render_rt();

  gl.bindVertexArray(null);
  gl.useProgram(null);

  requestAnimationFrame(wrays_viewer_render);
}

function webrays_parse_mtl(text)
{
  const NEWMTL_RE = /^newmtl\s/;
  const SHININESS_RE = /^Ns\s/; // shininess
  const KD_RE = /^Kd\s/;
  const KS_RE = /^Ks\s/;
  const KE_RE = /^Ke\s/; // emissive
  const D_RE = /^d\s/; // opacity. Format alternatively support Tr which is (1 - d)
  const IOR_RE = /^Ni\s/;
  const ILLUM_MODEL_RE = /^illum\s/; // [0-2] Phong, [3,8,9] Reflection, 4 Glass, 5 Fresnel Reflection, [6,7] Refraction
  const MAP_KD_RE = /^map_Kd\s/;
  const MAP_KS_RE = /^map_Ks\s/;
  const MAP_NS_RE = /^map_Ns\s/;
  const MAP_NORMAL_RE = /^map_Bump\s/i;
  const WHITESPACE_RE = /\s+/;  

  const MATERIAL_SYSTEM_WEBRAYS = 1;  
  const MATERIAL_SYSTEM_UNITY = 2;  
  const material_system = MATERIAL_SYSTEM_UNITY;  

  let materials = [];
  let currentMaterial = null;

  const lines = text.split("\n");
  for (let line of lines) {
    line = line.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const elements = line.split(WHITESPACE_RE);
    elements.shift();
    if (NEWMTL_RE.test(line)) {
      if(currentMaterial !== null)
        materials.push(currentMaterial);      
      currentMaterial = { 
        name: elements[0],
        type: 3, //type : 0 - perfect specular, 1 - fresnel specular, 2 - CT GGX, 3 - Lambert, 
        Kd: [0, 0, 0],
        Ks: [0,0,0],
        Ns: 0,
        ior: 1,
        map_Kd: null,
        map_Ks: null,
        map_Ns: null,
        map_bump: null,
      }
    }
    else if (KD_RE.test(line)) {
      currentMaterial.Kd = [...elements];
    }
    else if (KS_RE.test(line)) {
      currentMaterial.Ks = [...elements];
    }
    else if (SHININESS_RE.test(line)) {
      const shininess = parseFloat(elements[0]);
      currentMaterial.Ns = shininess;
    }
    else if (IOR_RE.test(line)) {
      currentMaterial.ior = parseFloat(elements[0]);
    }
    else if (MAP_KD_RE.test(line)) {
      currentMaterial.map_Kd = elements[0];
    }
    else if (MAP_KS_RE.test(line)) {
      currentMaterial.map_Ks = elements[0];
    }
    else if (MAP_NORMAL_RE.test(line)) {
      currentMaterial.map_bump = elements[0];
    }
    else if (ILLUM_MODEL_RE.test(line)) {
      // [0-2] Phong, [3,8,9] Reflection, 4 Glass, 5 Fresnel Reflection, [6,7] Refraction
      //type : 0 - perfect specular, 1 - fresnel specular, 2 - CT GGX, 3 - Lambert, 
      let model = parseInt(elements[0]);
      switch(model)
      {
        case 0:
        case 1:
        case 2:
          currentMaterial.type = 2;
          break;
        case 3:
        case 8:
        case 9:
          currentMaterial.type = 0;
          break;
        case 4:
        case 5:
        case 6:
        case 7:
          currentMaterial.type = 1;
          break;
      }
    }
  }
  if(currentMaterial !== null)
    materials.push(currentMaterial);

  let textures = [];
  for(let element of materials)
  {
    if(element.map_Kd)
    {
      let index = textures.indexOf(element.map_Kd);
      if(index === -1)
      {
        index = textures.length;
        textures.push(element.map_Kd);
      }
      element.map_Kd = index;
    }
    else
      element.map_Kd = -1;

    if(element.map_Ks)
    {
      let index = textures.indexOf(element.map_Ks);
      if(index === -1)
      {
        index = textures.length;
        textures.push(element.map_Ks);
      }
      element.map_Ks = index;
    }
    else
      element.map_Ks = -1;

    if(element.map_bump)
    {
      let index = textures.indexOf(element.map_bump);
      if(index === -1)
      {
        index = textures.length;
        textures.push(element.map_bump);
      }
      element.map_bump = index;
    }
    else
      element.map_bump = -1;
  }

  // convert to our system
  materials = materials.map(element => {

    let max_diff = Math.max(element.Kd);
	  let max_spec = Math.max(element.Ks);
    let metallic = max_spec / (max_spec + max_diff);
    
    let color = [Math.max(element.Kd[0], element.Ks[0]),
    Math.max(element.Kd[1], element.Ks[1]),
    Math.max(element.Kd[2], element.Ks[2])];

    //currentMaterial.roughness = 1.0 - glm::clamp(prev_mat.shininess / 128, 0.f, 1.f);
    let roughness = Math.sqrt(2.0 / (element.Ns + 2.0)); // from graphics rants UE4
    roughness = Math.sqrt(roughness); // from graphics rants UE4
    return {
      type: element.type,
      baseColor: color,

      nIndex: element.ior,
      metallic:metallic,
      roughness: roughness,
      reflectance: 0.5,

      baseColorTexture: element.map_Kd,
      MetallicRoughnessTexture: (material_system == MATERIAL_SYSTEM_UNITY) ? element.map_Ks : -1,
      NormalsTexture: element.map_bump,
      UnusedTexture: -1
    }
  });
  return {texures: textures, mats: materials};
}

function webrays_load_obj(url, mat_url = null)
{
  const objPromise = fetch(url).then(res => {
    return res.text();
  }).then(res => {
    let mesh = new OBJ.Mesh(res);
    return mesh;
  }).catch(error => {
    console.log("Error ", error);
    return null;
  });

  const mtlPromise = fetch(mat_url).then(res => {
    return res.text();
  }).then(res => {
    let mtls = webrays_parse_mtl(res);
    return mtls;
  }).catch(error => {
    console.log("Error ", error);
    return null;
  });


  return Promise.all([objPromise, mtlPromise]).then(res => {
    return res;
  });  
};
