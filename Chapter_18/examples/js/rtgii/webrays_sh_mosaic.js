"use strict";

const MaterialType = {
  PERFECT_SPECULAR : 0,
  FRESNEL_SPECULAR : 1,
  CT_GGX : 2,
  LAMBERT : 3,  
  DISNEY : 4, // ??
  THICK_GLASS : 5 // ??
};

const IOR_VALUES = {
  n_index_air : 1.000293,
  n_index_water : 1.333,
  n_index_ice : 1.31,
  n_index_glass : 1.52,
  n_index_diamond : 2.417,
  n_index_amber : 1.55,
  n_index_sapphire : 1.77
}

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

  WebRaysViewer.camera = new FreeRoamPerspectiveCamera3(WebRaysViewer.canvas.width ,WebRaysViewer.canvas.height);
  //WebRaysViewer.camera = new OrbitPerspectiveCamera(WebRaysViewer.canvas.width ,WebRaysViewer.canvas.height);
  WebRaysViewer.camera.attachControls(WebRaysViewer.canvas);

  window.addEventListener('keydown', (key) => {
    const code = key.key;

    if(code === 'r' || code === 'R')
    {
      Promise.all(WebRaysViewer.shader_urls.map(url =>
        fetch(url, {cache: "no-store"}).then(resp => resp.text())
      )).then(texts => {                   
        WebRaysViewer.gbuffer_program_vs_source = texts[0];
        WebRaysViewer.gbuffer_program_fs_source = texts[1];
        
        WebRaysViewer.indirect_illum_program_vs_source = texts[2];
        WebRaysViewer.indirect_illum_program_fs_source = texts[3];  

        WebRaysViewer.pp_program_vs_source = texts[4];
        WebRaysViewer.pp_program_fs_source = texts[5];  

        let scene_accessor_string = wr.GetSceneAccessorString();      

        // Compile and Link Programs throw on error. So we do not need extra code
        let fragment_source = scene_accessor_string + WebRaysViewer.indirect_illum_program_fs_source;
        let vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.indirect_illum_program_vs_source, gl.VERTEX_SHADER);
        let fragment_shader = wrays_gl_utils_compile_shader(fragment_source, gl.FRAGMENT_SHADER);
        let program = wrays_gl_utils_create_program(vertex_shader, fragment_shader);

        vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.gbuffer_program_vs_source, gl.VERTEX_SHADER);
        fragment_shader = wrays_gl_utils_compile_shader(WebRaysViewer.gbuffer_program_fs_source, gl.FRAGMENT_SHADER);
        let program2 = wrays_gl_utils_create_program(vertex_shader, fragment_shader);

        let program3;
        let program4;

        vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.pp_program_vs_source, gl.VERTEX_SHADER);
        fragment_shader = wrays_gl_utils_compile_shader(WebRaysViewer.pp_program_fs_source, gl.FRAGMENT_SHADER);
        let program5 = wrays_gl_utils_create_program(vertex_shader, fragment_shader);
        
        // No need to check
        if(program !== null)
        {
          WebRaysViewer.indirect_illum_program = program;
          WebRaysViewer.gbuffer_program = program2;
          WebRaysViewer.pp_program = program5;
          WebRaysViewer.frame_count = 0;
        }

      });
    }
  }, true);

  WebRaysViewer.lightManager = LightManager();

  let dir = DirectionalLight([0,1,0], [128,256,-1]);
  WebRaysViewer.lightManager.addLight(dir);
  dir.intensity = 1000.0;
  let quad = QuadLight([-5.1, 0, -2], [0, 0, 4], [0, 2, 0]);
  WebRaysViewer.lightManager.addLight(quad);
  WebRaysViewer.lights_ubo_bind_location = 0;
  WebRaysViewer.lights_ubo = gl.createBuffer();
  gl.bindBuffer(gl.UNIFORM_BUFFER, WebRaysViewer.lights_ubo);
  gl.bufferData(gl.UNIFORM_BUFFER, WebRaysViewer.lightManager.getGPUBuffer(), gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  gl.bindBufferBase(gl.UNIFORM_BUFFER, WebRaysViewer.lights_ubo_bind_location, WebRaysViewer.lights_ubo);

  wrays_viewer_init()
}

function wrays_viewer_resize() {

  WebRaysViewer.camera.setSize(gl.canvas.clientWidth, gl.canvas.clientHeight);

  // Initialize Textures

  // Delete previous state
  if(WebRaysViewer.hybrid_buffer !== undefined)
  {
    gl.deleteFramebuffer(WebRaysViewer.hybrid_buffer.FBO);
    
    gl.deleteTexture(WebRaysViewer.hybrid_buffer.depth_texture);
    gl.deleteTexture(WebRaysViewer.hybrid_buffer.color_texture);
    gl.deleteTexture(WebRaysViewer.hybrid_buffer.normals_texture);
    gl.deleteTexture(WebRaysViewer.hybrid_buffer.materials_texture);
  }

  /* HYBRID RENDERER */

  WebRaysViewer.hybrid_buffer = {
    FBO: gl.createFramebuffer(),
    depth_texture: wrays_gl_utils_texture_2d_alloc(gl.DEPTH_COMPONENT32F, WebRaysViewer.canvas.width ,WebRaysViewer.canvas.height),
    color_texture: wrays_gl_utils_texture_2d_alloc(gl.R11F_G11F_B10F , WebRaysViewer.canvas.width ,WebRaysViewer.canvas.height), // unsigned unormalized : R11F_G11F_B10F
    normals_texture: wrays_gl_utils_texture_2d_alloc(gl.RGB10_A2 , WebRaysViewer.canvas.width ,WebRaysViewer.canvas.height), // unsigned normalized : RGB10_A2
    materials_texture: wrays_gl_utils_texture_2d_alloc(gl.RGB10_A2 , WebRaysViewer.canvas.width ,WebRaysViewer.canvas.height),// unsigned normalized : RGB10_A2

    FBO_RT: gl.createFramebuffer(),
    rt_accum_texture: wrays_gl_utils_texture_2d_alloc(gl.RGBA32F , WebRaysViewer.canvas.width ,WebRaysViewer.canvas.height),
  };  

  gl.bindFramebuffer(gl.FRAMEBUFFER, WebRaysViewer.hybrid_buffer.FBO);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, WebRaysViewer.hybrid_buffer.depth_texture, 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, WebRaysViewer.hybrid_buffer.color_texture, 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, WebRaysViewer.hybrid_buffer.normals_texture, 0);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, WebRaysViewer.hybrid_buffer.materials_texture, 0);
  if (gl.FRAMEBUFFER_COMPLETE != gl.checkFramebufferStatus(gl.FRAMEBUFFER)) {
    console.log('error Hybrid FBO');
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, WebRaysViewer.hybrid_buffer.FBO_RT);
  gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, WebRaysViewer.hybrid_buffer.rt_accum_texture, 0);
  if (gl.FRAMEBUFFER_COMPLETE != gl.checkFramebufferStatus(gl.FRAMEBUFFER)) {
    console.log('error Hybrid FBO RT');
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
  for (let attr_index = 0; attr_index < attr_count; attr_index ++) {
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
  
  let materialTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, materialTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 3, num_materials, 0, gl.RGBA, gl.FLOAT, new Float32Array(WebRaysViewer.mesh.materials));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.bindTexture(gl.TEXTURE_2D, null);

  // create the textures
  let textures = null;
  const resource_urls = WebRaysViewer.mesh.materialTextureURLs;
  Promise.all(resource_urls.map(u=> {
    if(u instanceof HTMLImageElement)
      return u;
    else
      return fetch(u).then(response => {
        if(!response.ok)
        {
          console.error("Could not fetch: ", response);
          return;
        }
        return response.blob();
      }).then(response => {
        let image = document.createElement('img');
        image.src = URL.createObjectURL(response);
        return image;
      });
  })).then(decoded_imgs => {
    // Async image load 
    Promise.all(decoded_imgs.map(img=>img.decode())).then(responses => {
      WebRaysViewer.gl_mesh.texturesBuffer = wrays_gl_utils_texture_from_images_scaled(decoded_imgs, WebRaysViewer.mesh.flip_textures);
    }).catch(err => {
      console.error(err);
    });
  });

  gl.bindVertexArray(vao);
          
  gl.enableVertexAttribArray(0);
  gl.enableVertexAttribArray(1);
  gl.enableVertexAttribArray(2);
  
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  //gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Int32Array(WebRaysViewer.mesh.indices), gl.STATIC_DRAW);
  let indices = [];
  /*for (let mat_index = 0; mat_index < WebRaysViewer.mesh.indicesPerMaterial.length; mat_index ++) {
    indices = indices.concat(WebRaysViewer.mesh.indicesPerMaterial[mat_index]);
  }*/
  indices = WebRaysViewer.mesh.indicesPerMaterial.flat();
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

  WebRaysViewer.gl_mesh = {vao: vao, vbo: vbo, ibo: ibo, attr_count: attr_count, parts: parts, materialTexture: materialTexture, texturesBuffer: textures};

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  /* GL Programs */
  let vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.gbuffer_program_vs_source, gl.VERTEX_SHADER);
  let fragment_shader = wrays_gl_utils_compile_shader(WebRaysViewer.gbuffer_program_fs_source, gl.FRAGMENT_SHADER);
  WebRaysViewer.gbuffer_program = wrays_gl_utils_create_program(vertex_shader, fragment_shader);

  vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.pp_program_vs_source, gl.VERTEX_SHADER);
  fragment_shader = wrays_gl_utils_compile_shader(WebRaysViewer.pp_program_fs_source, gl.FRAGMENT_SHADER);
  WebRaysViewer.pp_program = wrays_gl_utils_create_program(vertex_shader, fragment_shader);
}

function wrays_viewer_scene_init() {
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

function wrays_viewer_init() {
  /* Initialize things (This could very well be a constructor) */
 
  // Viewer Properties
  WebRaysViewer.tile_width  = 256;
  WebRaysViewer.tile_height = 256;
  WebRaysViewer.depth       = 1;
  WebRaysViewer.frame_count = 0;

  // Textures
  WebRaysViewer.env_map_texture = null;
  WebRaysViewer.blue_noise_texture = null;

  // Rasterization
  WebRaysViewer.fbo = null;
  WebRaysViewer.vao = null;
  WebRaysViewer.vbo = null;
  WebRaysViewer.texture = null;

  /* Program sources */
  WebRaysViewer.indirect_illum_program_fs_source = null;
  WebRaysViewer.indirect_illum_program_vs_source = null;

  WebRaysViewer.gbuffer_program_fs_source = null;
  WebRaysViewer.gbuffer_program_vs_source = null;

  WebRaysViewer.pp_program_fs_source = null;
  WebRaysViewer.pp_program_vs_source = null;

  /* GL Program handles */
  WebRaysViewer.indirect_illum_program = null;
  WebRaysViewer.gbuffer_program = null;
  WebRaysViewer.pp_program = null;


  WebRaysViewer.mesh = null;
  WebRaysViewer.gl_mesh = null;

  /* Timers */
  WebRaysViewer.enable_timers = false;
  if(gl_timer_ext) {
    WebRaysViewer.dr_timer = wrays_gl_utils_create_single_buffered_timer();
    WebRaysViewer.rt_timer = wrays_gl_utils_create_single_buffered_timer();
  }
  
  /* Resources to be loaded async */
  WebRaysViewer.shader_urls = [
                               'kernels/gbuffer.vs.glsl', 'kernels/gbuffer_gk.fs.glsl',
                               'kernels/screen_fill.vs.glsl', 'kernels/rtgii/wrays_viewer_rtg_shadows2.glsl',
                               'kernels/screen_fill.vs.glsl', 'kernels/post_processing.fs.glsl',
                              ];
  WebRaysViewer.resource_urls = ['textures/park_2k.jpg', 'textures/HDR_RGBA_0.png'];

  /* Set Viewer callbacks */
  WebRaysViewer.camera.attachControls(WebRaysViewer.canvas);
  
  Promise.all(WebRaysViewer.resource_urls.map(u=>fetch(u))).then(responses => {
    var decoded_imgs = [];
    for (var rsrc_index = 0; rsrc_index < WebRaysViewer.resource_urls.length; rsrc_index++) {
      if (false == responses[rsrc_index].ok) {
        return;
      }
    }

    Promise.all(responses.map(res => res.blob())).then(buffers => {
      var decoded_imgs = [];
      for (var rsrc_index = 0; rsrc_index < WebRaysViewer.resource_urls.length; rsrc_index++) {
        var image = document.createElement('img');
        image.src = URL.createObjectURL(buffers[rsrc_index]);
        decoded_imgs.push(image);
      }

      /* Async image load */
      Promise.all(decoded_imgs.map(img=>img.decode())).then(responses => {
        /* Index zero always has the env map */
        var env_map_image = decoded_imgs[0];
        WebRaysViewer.env_map_texture = wrays_gl_utils_texture_from_image(env_map_image, env_map_image.width, env_map_image.height);
        const blue_noise_image = decoded_imgs[1];
        WebRaysViewer.blue_noise_texture = wrays_gl_utils_texture_from_image(blue_noise_image, blue_noise_image.width, blue_noise_image.height);
        
        /* Async shader load */
        Promise.all(WebRaysViewer.shader_urls.map(url =>
          fetch(url, {cache: "no-store"}).then(resp => resp.text())
        )).then(texts => {
                     
          WebRaysViewer.gbuffer_program_vs_source = texts[0];
          WebRaysViewer.gbuffer_program_fs_source = texts[1];
          
          WebRaysViewer.indirect_illum_program_vs_source = texts[2];
          WebRaysViewer.indirect_illum_program_fs_source = texts[3];

          WebRaysViewer.pp_program_vs_source = texts[4];
          WebRaysViewer.pp_program_fs_source = texts[5];
          
          wr = new WebRays.WebGLIntersectionEngine();
          const gltf_file = 'gltf/arealightexample1/example1.gltf';
          webrays_load_gltf(gltf_file).then(res => {
            WebRaysViewer.mesh = res;

            {
              let minValue = [Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE];
              let maxValue = [-Number.MAX_VALUE, -Number.MAX_VALUE, -Number.MAX_VALUE];
              for(let ii = 0; ii < WebRaysViewer.mesh.vertices.length; ++ii)
              {
                minValue[ii % 3] = Math.min(minValue[ii % 3], WebRaysViewer.mesh.vertices[ii]);
                maxValue[ii % 3] = Math.max(maxValue[ii % 3], WebRaysViewer.mesh.vertices[ii]);
              }
              let center = [0.5 * (minValue[0] + maxValue[0]), 0.5 * (minValue[1] + maxValue[1]), 0.5 * (minValue[2] + maxValue[2])];

              //const position = [center[0], center[1], maxValue[2] + 1.0];
              let position = [center[0], maxValue[1] - 1.0, maxValue[2] + 1.0];

              // custom camera
              if(gltf_file === 'gltf/arealightexample1/example1.gltf')
              {
                position[0] = 2.101961612701416;
                position[1] = 0.6843416690826416;
                position[2] = 1.4356122016906738;

                center[0] = position[0] - 0.8233922719955444;
                center[1] = position[1] - 0.27443304657936096;
                center[2] = position[2] - 0.4967007637023926;
              }
              
              WebRaysViewer.camera.setPosition(...position);
              WebRaysViewer.camera.setTarget(...center);              
              WebRaysViewer.camera.setSceneBBOX(minValue, maxValue);
            }

            wrays_viewer_framebuffer_init();
            wrays_viewer_scene_init();
      
            // Begin rendering 
            requestAnimationFrame(wrays_viewer_render);
          });
        });
      });
    });
  });                       
}

function wrays_radians(a) {
  const degree = Math.PI / 180;
  return a * degree;
}

function wrays_viewer_update() {
  if(WebRaysViewer.camera.update())
    WebRaysViewer.frame_count = 0;

  var update_flags = wr.Update();
  if(0 == update_flags)
    return;

  let scene_accessor_string = wr.GetSceneAccessorString();      

  let fragment_source = scene_accessor_string + WebRaysViewer.indirect_illum_program_fs_source;
  let vertex_shader = wrays_gl_utils_compile_shader(WebRaysViewer.indirect_illum_program_vs_source, gl.VERTEX_SHADER);
  let fragment_shader = wrays_gl_utils_compile_shader(fragment_source, gl.FRAGMENT_SHADER);
  WebRaysViewer.indirect_illum_program = wrays_gl_utils_create_program(vertex_shader, fragment_shader);
}

function wrays_viewer_blit(tile_index_x, tile_index_y) {
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, WebRaysViewer.preview_FBO);

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

function wrays_viewer_render_rt(now) {
  var tile_count_x = ((WebRaysViewer.canvas.width - 1) / WebRaysViewer.tile_width) + 1;
	var tile_count_y = ((WebRaysViewer.canvas.height - 1) / WebRaysViewer.tile_height) + 1;
 
  gl.bindVertexArray(WebRaysViewer.vao);

  for (var tile_index_x = 0; tile_index_x < tile_count_x; tile_index_x++) {
		for (var tile_index_y = 0; tile_index_y < tile_count_y; tile_index_y++) {
      for (var depth = 0; depth < WebRaysViewer.depth; depth++  ) {
        wrays_viewer_rays_generate(tile_index_x, tile_index_y, depth);
        
        wrays_viewer_isects_handle(tile_index_x, tile_index_y, depth);

      }
      wrays_viewer_blit(tile_index_x, tile_index_y);
    }
  }
}

function wrays_viewer_render_rt_hybrid() {
  var tile_count_x = ((WebRaysViewer.canvas.width - 1) / WebRaysViewer.tile_width) + 1;
  var tile_count_y = ((WebRaysViewer.canvas.height - 1) / WebRaysViewer.tile_height) + 1;
  tile_count_x = 1;
  tile_count_y = 1;
  
  gl.bindFramebuffer(gl.FRAMEBUFFER, WebRaysViewer.hybrid_buffer.FBO_RT);

  gl.viewport(0, 0, WebRaysViewer.canvas.width, WebRaysViewer.canvas.height);
  gl.disable(gl.DEPTH_TEST); 
  
  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  //gl.blendFunc(gl.ONE, gl.ONE);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  
  //gl.drawBuffers([ gl.COLOR_ATTACHMENT0 ]);
  //gl.viewport(0, 0, WebRaysViewer.tile_width, WebRaysViewer.tile_height);
 
  gl.bindVertexArray(WebRaysViewer.vao);

  for (var tile_index_x = 0; tile_index_x < tile_count_x; tile_index_x++) {
		for (var tile_index_y = 0; tile_index_y < tile_count_y; tile_index_y++) {
      
      gl.useProgram(WebRaysViewer.indirect_illum_program);

      var index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "tile");
      gl.uniform4iv(index, [tile_index_x, tile_index_y, WebRaysViewer.tile_width, WebRaysViewer.tile_height]);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "frame");
      gl.uniform2iv(index, [WebRaysViewer.canvas.width, WebRaysViewer.canvas.height]);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "frame_count");
      gl.uniform1ui(index, WebRaysViewer.frame_count);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "seed");
      gl.uniform2ui(index, Math.floor(Math.random() * 10000), Math.floor(Math.random() * 10000));
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "camera_pos");
      gl.uniform3fv(index, WebRaysViewer.camera.camera_pos);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "camera_up");
      gl.uniform3fv(index, WebRaysViewer.camera.camera_up);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "camera_front");
      gl.uniform3fv(index, WebRaysViewer.camera.camera_front);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "camera_right");
      gl.uniform3fv(index, WebRaysViewer.camera.camera_right);

      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "ads");
      gl.uniform1i(index, blas);

      // Textures
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "env_map");
      gl.activeTexture(gl.TEXTURE0 + 0);
	    gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.env_map_texture);
      gl.uniform1i(index, 0);

      // Material Properties
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "u_materialBuffer");
      gl.uniform1i(index, 1);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.gl_mesh.materialTexture);
      // Object Textures
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "u_texturesBuffer");
      gl.uniform1i(index, 2);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, WebRaysViewer.gl_mesh.texturesBuffer);
      
      // GBuffer Textures
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "u_gBuffer_depth");
      gl.uniform1i(index, 3);
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.hybrid_buffer.depth_texture);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "u_gBuffer_material");
      gl.uniform1i(index, 4);
      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.hybrid_buffer.materials_texture);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "u_gBuffer_normal");
      gl.uniform1i(index, 5);
      gl.activeTexture(gl.TEXTURE5);
      gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.hybrid_buffer.normals_texture);

      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "u_blue_noise_map");
      gl.activeTexture(gl.TEXTURE0 + 6);
	    gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.blue_noise_texture);
      gl.uniform1i(index, 6);
      
      // note: glmatrix.js always has the first argument
      // as the destination to receive the result.
      var projectionInvMatrix = glMatrix.mat4.create();
      glMatrix.mat4.invert(projectionInvMatrix, WebRaysViewer.camera.projection);
      var viewInvMatrix = glMatrix.mat4.create();
      glMatrix.mat4.invert(viewInvMatrix, WebRaysViewer.camera.view);

      // Camera
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "u_ProjectionInvMatrix");
      gl.uniformMatrix4fv(index, false, projectionInvMatrix);
      index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, "u_ViewInvMatrix");
      gl.uniformMatrix4fv(index, false, viewInvMatrix);

      // Bind the lights UBO
      gl.uniformBlockBinding(WebRaysViewer.indirect_illum_program, gl.getUniformBlockIndex(WebRaysViewer.indirect_illum_program, "u_lightsArray"), WebRaysViewer.lights_ubo_bind_location);

      
      var bindings = wr.Bindings;
      var next_texture_unit = 7;
      for (var binding_index = 0; binding_index < bindings.length; ++binding_index) {		
        var binding = bindings[binding_index];

        /* if UBO */
        if (binding.Type == 1) {
      
        /* if Texture 2D */
        } else if (binding.Type == 2) {
          index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, binding.Name);
          gl.activeTexture(gl.TEXTURE0 + next_texture_unit);
          gl.bindTexture(gl.TEXTURE_2D, binding.Texture);
          gl.uniform1i(index, next_texture_unit);
          next_texture_unit++;
        /* if Texture Array 2D */
        } else if (binding.Type == 3) {
          index = gl.getUniformLocation(WebRaysViewer.indirect_illum_program, binding.Name);
          gl.activeTexture(gl.TEXTURE0 + next_texture_unit);
          gl.bindTexture(gl.TEXTURE_2D_ARRAY, binding.Texture);
          gl.uniform1i(index, next_texture_unit);
          next_texture_unit++;
        }
      }  
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      while(next_texture_unit >= 0)
      {
        gl.activeTexture(gl.TEXTURE0 + next_texture_unit);
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
        next_texture_unit = next_texture_unit - 1;
      }
    }
  }

  gl.disable(gl.BLEND);
}


function wrays_viewer_render_dr() {
  gl.disable(gl.CULL_FACE);
  gl.bindFramebuffer(gl.FRAMEBUFFER, WebRaysViewer.hybrid_buffer.FBO);
  gl.drawBuffers([ gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2 ]);

  gl.viewport(0, 0, WebRaysViewer.canvas.width, WebRaysViewer.canvas.height);
  gl.enable(gl.DEPTH_TEST); 

  gl.clearColor(0.0,0.0,0.0,0);
  gl.clearDepth(1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  
  gl.useProgram(WebRaysViewer.gbuffer_program);

  // Camera
  var index = gl.getUniformLocation(WebRaysViewer.gbuffer_program, "u_ProjectionMatrix");
  gl.uniformMatrix4fv(index, false, WebRaysViewer.camera.projection);
  index = gl.getUniformLocation(WebRaysViewer.gbuffer_program, "u_ViewMatrix");
  gl.uniformMatrix4fv(index, false, WebRaysViewer.camera.view);
  index = gl.getUniformLocation(WebRaysViewer.gbuffer_program, "u_camera_position");
  gl.uniform3f(index, WebRaysViewer.camera.camera_pos[0], WebRaysViewer.camera.camera_pos[1], WebRaysViewer.camera.camera_pos[2]);

  // Render Object  
  gl.bindVertexArray(WebRaysViewer.gl_mesh.vao);  
    
  // Material Properties
  index = gl.getUniformLocation(WebRaysViewer.gbuffer_program, "u_materialBuffer");
  gl.uniform1i(index, 0);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.gl_mesh.materialTexture);
  // Object Textures
  index = gl.getUniformLocation(WebRaysViewer.gbuffer_program, "u_texturesBuffer");
  gl.uniform1i(index, 1);
  gl.activeTexture(gl.TEXTURE0 + 1);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, WebRaysViewer.gl_mesh.texturesBuffer);

  // Environment Map
  index = gl.getUniformLocation(WebRaysViewer.gbuffer_program, "env_map");
  gl.activeTexture(gl.TEXTURE0 + 2);
	gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.env_map_texture);
  gl.uniform1i(index, 2);

  index = gl.getUniformLocation(WebRaysViewer.gbuffer_program, "u_blue_noise_map");
  gl.activeTexture(gl.TEXTURE0 + 3);
	gl.bindTexture(gl.TEXTURE_2D, WebRaysViewer.blue_noise_texture);
  gl.uniform1i(index, 3);
  
  index = gl.getUniformLocation(WebRaysViewer.gbuffer_program, "u_material_index");
  let part_index = 0;
  for(const part of WebRaysViewer.gl_mesh.parts)
  {
    gl.uniform1i(index, part_index);
    gl.drawElements(gl.TRIANGLES, part.count * 3, gl.UNSIGNED_INT, 3 * part.offset * Int32Array.BYTES_PER_ELEMENT);  
    part_index++;
  }

  gl.useProgram(null);
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, null)
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, null)
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null)
}

function wrays_viewer_render_post_processing(texture) {
  
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  //gl.drawBuffers([ gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2 ]);

  gl.viewport(0, 0, WebRaysViewer.canvas.width, WebRaysViewer.canvas.height);
    
  gl.useProgram(WebRaysViewer.pp_program);

  // Camera
  let index = gl.getUniformLocation(WebRaysViewer.pp_program, "u_inputBuffer");
  gl.uniform1i(index, 0);
  gl.activeTexture(gl.TEXTURE0 + 0);
  gl.bindTexture(gl.TEXTURE_2D, texture);

  index = gl.getUniformLocation(WebRaysViewer.pp_program, "frame_count");
  gl.uniform1ui(index, WebRaysViewer.frame_count);
  
  // Render Object  
  gl.bindVertexArray(WebRaysViewer.vao);  

  gl.drawArrays(gl.TRIANGLES, 0, 3);    

  gl.useProgram(null);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null)
}

function wrays_viewer_render(now) {
  wrays_viewer_update();

  // Jitter Camera
  let at = glMatrix.vec3.add(glMatrix.vec3.create(), WebRaysViewer.camera.camera_front, WebRaysViewer.camera.camera_pos);
  let rx = (Math.random() * 2.0 - 1.0) * (0.001);
  let ry = (Math.random() * 2.0 - 1.0) * (0.001); /// prev magic value was 0.05

  at[0] += WebRaysViewer.camera.camera_up[0] * ry + WebRaysViewer.camera.camera_right[0] * rx;
  at[1] += WebRaysViewer.camera.camera_up[1] * ry + WebRaysViewer.camera.camera_right[1] * rx;
  at[2] += WebRaysViewer.camera.camera_up[2] * ry + WebRaysViewer.camera.camera_right[2] * rx;

  WebRaysViewer.camera.view = glMatrix.mat4.lookAt( 
    glMatrix.mat4.create(),
    WebRaysViewer.camera.camera_pos, // pos
    at, // at
    WebRaysViewer.camera.camera_up // up
  ); // view is [right, up, forward, -pos]^T;


  WebRaysViewer.frame_count += 1;

  if(WebRaysViewer.enable_timers)
  {
    wrays_gl_utils_begin_single_buffered_timer(gl_timer_ext, WebRaysViewer.dr_timer);
    wrays_viewer_render_dr();
    wrays_gl_utils_end_single_buffered_timer(gl_timer_ext, WebRaysViewer.dr_timer);
  }
  else
    wrays_viewer_render_dr();
  
  if(WebRaysViewer.enable_timers)
  {
    wrays_gl_utils_begin_single_buffered_timer(gl_timer_ext, WebRaysViewer.rt_timer);
    wrays_viewer_render_rt_hybrid();
    wrays_gl_utils_end_single_buffered_timer(gl_timer_ext, WebRaysViewer.rt_timer);
  }
  else
    wrays_viewer_render_rt_hybrid();

  //wrays_viewer_render_rt();
  
  gl.bindVertexArray(null);
  gl.useProgram(null);

  const enablePP = true;
  if(enablePP)
  {
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, WebRaysViewer.hybrid_buffer.FBO_RT);

    wrays_viewer_render_post_processing(WebRaysViewer.hybrid_buffer.rt_accum_texture);
  }
  else
  {
    // Blit framebuffer
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, WebRaysViewer.hybrid_buffer.FBO_RT);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, WebRaysViewer.hybrid_buffer.FBO_RT);

    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    //gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

    var dst_viewport = [
      0, 0,
      WebRaysViewer.canvas.width,
      WebRaysViewer.canvas.height
    ];
    var src_viewport = [
      0, 0,
      WebRaysViewer.canvas.width,
      WebRaysViewer.canvas.height
    ];

    gl.blitFramebuffer(
      src_viewport[0], src_viewport[1], src_viewport[2], src_viewport[3],
      dst_viewport[0], dst_viewport[1], dst_viewport[2], dst_viewport[3],
      gl.COLOR_BUFFER_BIT, gl.NEAREST);
  }

  if(WebRaysViewer.enable_timers)
  {
    const ellapsedTime = wrays_gl_utils_get_single_buffered_timer(gl_timer_ext, WebRaysViewer.dr_timer) / 1000000.0; // ms
    if(ellapsedTime !== 0.0)
      console.log('DR :'+ellapsedTime+' ms');

    const ellapsedTime2 = wrays_gl_utils_get_single_buffered_timer(gl_timer_ext, WebRaysViewer.rt_timer) / 1000000.0; // ms
    if(ellapsedTime2 !== 0.0)
      console.log('RT :'+ellapsedTime2+' ms');
  }

  requestAnimationFrame(wrays_viewer_render);
}

function webrays_gltf_load_nodes(glTF, r_mesh, node, parentModel = null)
{
  let model = (node.matrix !== null)? glMatrix.mat4.copy(glMatrix.mat4.create(), node.matrix) : null;
  if(parentModel !== null && model !== null)
    model = glMatrix.mat4.multiply(glMatrix.mat4.create(), parentModel, model);
  
  for(let i = 0; i < node.children.length; i++)
  {
    if(node.children[i].mesh === null)
    {   
      webrays_gltf_load_nodes(glTF, r_mesh, node.children[i], model);
    }
    else
    {
      webrays_gltf_load_mesh(glTF, r_mesh, node.children[i], model);
    }
  }
}

const insertArray = (arr, ob) =>
{
  if(ob === null)
    return -1;
  let index = arr.indexOf(ob.index);
  if(index === -1)
  {
    index = arr.length;
    arr.push(ob.index);
  }
  return index;
}

var counter = 0;
function webrays_gltf_load_mesh(glTF, r_mesh, node, parentModel = null)
{
  /*if(counter > 1)
      return;
    counter++;*/  
  
  // GLTF loader returns matrix as an Float32Array. glMatrix.* are Float32Arrays
  let model = (node.matrix !== null)? glMatrix.mat4.copy(glMatrix.mat4.create(), node.matrix) : null;
  if(parentModel !== null)
    model = glMatrix.mat4.multiply(glMatrix.mat4.create(), parentModel, model);
  const normalMatrix = (node.matrix !== null)? glMatrix.mat3.normalFromMat4(glMatrix.mat3.create(), model) : null;

  const mesh = node.mesh;
  let indexOffset = 0;
  indexOffset = r_mesh.vertices.length / 3;
  for(let j = 0; j < mesh.primitives.length; j++)
  {
    const primitive = mesh.primitives[j];
    indexOffset = r_mesh.vertices.length / 3;

    if(primitive.indices !== null)
    {
      const accessor = glTF.accessors[primitive.indices];
      accessor.bufferView.data;
      accessor.byteOffset;
      accessor.count;
      accessor.type;
      accessor.componentType; // 5126 FLOAT, 5123 UNSIGNED_SHORT, 5125 UINT
      primitive.indicesOffset; // 0
      primitive.indicesLength; // 10920
      primitive.indicesComponentType; // 5123 USHORT
      primitive.mode; // 0 Point, 1 Line, 4 Triangle

      if(accessor.byteStride !== 0 && accessor.byteStride !== 6)
        console.error("ERRRRROR");

      const data = (primitive.indicesComponentType === 5123)? 
        new Uint16Array(accessor.bufferView.data, accessor.byteOffset, accessor.count) : 
        new Uint32Array(accessor.bufferView.data, accessor.byteOffset, accessor.count);
            
      r_mesh.indicesPerMaterial.push(Array.from(data).map(e => {return e + indexOffset}));
      //r_mesh.indicesPerMaterial.push(Array.from(data));
      if(primitive.material && primitive.material.pbrMetallicRoughness)
      {
        // TODO: FAST FIX. NEED TO BE FIXED IN A PROPER WAY
        let metallicRoughnessTextureIndex = -1;
        if(glTF.json.materials[primitive.material.materialID].pbrMetallicRoughness)
        {
          if(glTF.json.materials[primitive.material.materialID].pbrMetallicRoughness.metallicRoughnessTexture !== undefined)
          {
            const textureID = glTF.json.materials[primitive.material.materialID].pbrMetallicRoughness.metallicRoughnessTexture;
            metallicRoughnessTextureIndex = insertArray(r_mesh.materialTextureURLs, textureID);
          }
        }

        const pbr_mat = primitive.material.pbrMetallicRoughness;
        const baseColorTextureIndex = insertArray(r_mesh.materialTextureURLs, pbr_mat.baseColorTexture !== undefined? pbr_mat.baseColorTexture : null);
        //const metallicRoughnessTextureIndex = insertArray(r_mesh.materialTextureURLs, pbr_mat.metallicRoughnessTexture !== undefined? pbr_mat.metallicRoughnessTexture : null);
        const normalTextureIndex = insertArray(r_mesh.materialTextureURLs, pbr_mat.normalTexture !== undefined? pbr_mat.normalTexture : null);
        //const type = primitive.material.name.toLocaleLowerCase().includes('glass')? MaterialType.FRESNEL_SPECULAR : MaterialType.CT_GGX;
        //const type = primitive.material.name.toLocaleLowerCase().includes('Transluscent')? MaterialType.FRESNEL_SPECULAR : MaterialType.CT_GGX;
        //const type = primitive.material.name ==='Glass'? MaterialType.FRESNEL_SPECULAR : MaterialType.CT_GGX;
        // Fireplace
        let type = primitive.material.name ==='grey_and_white_roomaterial_Glass'? MaterialType.FRESNEL_SPECULAR : MaterialType.CT_GGX;
        type = primitive.material.name ==='grey_and_white_roomaterial_Mirror'? MaterialType.PERFECT_SPECULAR : type;
        
        r_mesh.materials.push({
          //type: MaterialType.CT_GGX,
          type: type,
          baseColor: pbr_mat.baseColorFactor,

          nIndex: 1.0,
          metallic: pbr_mat.metallicFactor,
          roughness: pbr_mat.roughnessFactor,
          reflectance: 0.5,

          baseColorTexture: baseColorTextureIndex,
          MetallicRoughnessTexture: metallicRoughnessTextureIndex,
          NormalsTexture: normalTextureIndex,
          UnusedTexture: -1
        });
      }
      else // default material
      {
        r_mesh.materials.push({
          type: MaterialType.CT_GGX,
          baseColor: [1, 1, 1, 1],

          nIndex: 1.0,
          metallic: 1.0,
          roughness: 1.0,
          reflectance: 0.5,

          baseColorTexture: -1,
          MetallicRoughnessTexture: -1,
          NormalsTexture: -1,
          UnusedTexture: -1
        });
      }
    }

    if(primitive.attributes.POSITION !== undefined) {
      const accessor = glTF.accessors[primitive.attributes.POSITION];
      const bv = accessor.bufferView;
      
      const stride = accessor.byteStride === 0 ? 3 : accessor.byteStride / 4; // stride in floats
      const data = new Float32Array(bv.data, accessor.byteOffset, stride * (accessor.count - 1) + 3);
      
      if(model !== null)
      {
        let pos2 = glMatrix.vec3.create();
        for(let index = 0; index < accessor.count; index++)
        {
          //let pos = new Float32Array(data, stride * index * 4, 3);
          let pos = glMatrix.vec3.fromValues(data[stride * index + 0], data[stride * index + 1], data[stride * index + 2]);
          pos = glMatrix.vec3.transformMat4(pos2, pos, model);
          
          r_mesh.vertices.push(pos[0]);
          r_mesh.vertices.push(pos[1]);
          r_mesh.vertices.push(pos[2]);
        }
      }
      else if(accessor.byteStride !== 0 && accessor.byteStride !== 12)
      {
        for(let index = 0; index < accessor.count; index++)
        {
          r_mesh.vertices.push(data[stride * index + 0]);
          r_mesh.vertices.push(data[stride * index + 1]);
          r_mesh.vertices.push(data[stride * index + 2]);
        }
      }
      else
        r_mesh.vertices = r_mesh.vertices.concat(Array.from(data));
    }
          
    if(primitive.attributes.NORMAL !== undefined) {
      const accessor = glTF.accessors[primitive.attributes.NORMAL];
      const bv = accessor.bufferView;

      const stride = accessor.byteStride === 0 ? 3 : accessor.byteStride / 4;
      const data = new Float32Array(bv.data, accessor.byteOffset, stride * (accessor.count - 1) + 3);
      
      if(model !== null)
      {     
        let pos2 = glMatrix.vec3.create();
        for(let index = 0; index < accessor.count; index++)
        {
          //let pos = new Float32Array(data, stride * index * 4, 3);
          let pos = glMatrix.vec3.fromValues(data[stride * index + 0], data[stride * index + 1], data[stride * index + 2]);
          pos = glMatrix.vec3.transformMat3(pos2, pos, normalMatrix);
          
          r_mesh.vertexNormals.push(pos[0]);
          r_mesh.vertexNormals.push(pos[1]);
          r_mesh.vertexNormals.push(pos[2]);
        }
      }
      else if(accessor.byteStride !== 0 && accessor.byteStride !== 12)
      {        
        for(let index = 0; index < accessor.count; index++)
        {
          r_mesh.vertexNormals.push(data[stride * index + 0]);
          r_mesh.vertexNormals.push(data[stride * index + 1]);
          r_mesh.vertexNormals.push(data[stride * index + 2]);
        }
      }
      else
        r_mesh.vertexNormals = r_mesh.vertexNormals.concat(Array.from(data));        
    }

    if(primitive.attributes.TEXCOORD_0 !== undefined) 
    {
      const accessor = glTF.accessors[primitive.attributes.TEXCOORD_0];
      const bv = accessor.bufferView;

      const stride = accessor.byteStride === 0 ? 2 : accessor.byteStride / 4;
      const data = new Float32Array(bv.data, accessor.byteOffset, stride * (accessor.count - 1) + 2);
      if(accessor.byteStride !== 0 && accessor.byteStride !== 8)
      {        
        for(let index = 0; index < accessor.count; index++)
        {
          r_mesh.textures.push(data[stride * index + 0]);
          r_mesh.textures.push(data[stride * index + 1]);
        }
      }
      else
        r_mesh.textures = r_mesh.textures.concat(Array.from(data));
    }
  }
}

function webrays_load_gltf(url) 
{
  const _gltfLoader = new glTFLoader(null);

  const update = (resolve, reject) => {
    _gltfLoader.load_GLTF(url, glTF => {
      let r_mesh = {
        vertices: [],
        vertexNormals: [],
        textures: [],
        indicesPerMaterial: [],
        materials: [],
        materialTextureURLs: []
      };

      let scene = glTF.defaultScene === undefined ? glTF.scenes[0] : glTF.scenes[glTF.defaultScene];
      // Get the First Scene TODO       

      for(let i = 0; i < scene.nodes.length; i++)
      {
        if(scene.nodes[i].mesh === null)
          webrays_gltf_load_nodes(glTF, r_mesh, scene.nodes[i]);
        else
          webrays_gltf_load_mesh(glTF, r_mesh, scene.nodes[i]);
      }

      //convert materials
      r_mesh.materials = r_mesh.materials.flatMap(e => [
        e.type, e.baseColor[0], e.baseColor[1], e.baseColor[2], 
        e.nIndex, e.metallic, e.roughness, e.reflectance,
        e.baseColorTexture, e.MetallicRoughnessTexture, e.NormalsTexture, e.UnusedTexture]);

      r_mesh.materialTextureURLs = [...new Set(r_mesh.materialTextureURLs)];
      r_mesh.materialTextureURLs = r_mesh.materialTextureURLs.map(e => glTF.images[e].currentSrc);
      /*for(let e of r_mesh.materialTextureURLs)
        e.style.transform = 'scaleY(-1)';*/
      r_mesh.flip_textures = false;
      
      resolve(r_mesh);
    }
  );
  }
  
  return new Promise(update);  
}

