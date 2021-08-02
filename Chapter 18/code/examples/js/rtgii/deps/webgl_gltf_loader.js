/*!
The MIT License (MIT)

Copyright (c) 2016 Shuai Shao (shrekshao) and Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

var Type2NumOfComponent = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16
};

class Scene {
    constructor(s, sceneID, nodes) {
        this.id    = sceneID;
        this.nodes = new Array(s.nodes.length);
        for (var i = 0; i < s.nodes.length; ++i) {
            this.nodes[i] = nodes[s.nodes[i]];
        }
    }
}

class Camera {
    constructor(c) {
        this.name = c.name !== undefined ? c.name : null;
        this.type = c.type; 

        this.othographic = c.othographic === undefined ? null : c.othographic; 
        this.perspective = c.perspective === undefined ? null : {
            yfov:        c.perspective.yfov,
            znear:       c.perspective.znear,
            zfar:        c.perspective.zfar        !== undefined ? c.perspective.zfar        : null,
            aspectRatio: c.perspective.aspectRatio !== undefined ? c.perspective.aspectRatio : null
        };

        this.extensions = c.extensions !== undefined ? c.extensions : null;
        this.extras     = c.extras     !== undefined ? c.extras     : null;
    }
}

class TextureInfo {
    constructor(tex){
        this.index    = tex.index;
        this.texCoord = tex.texCoord !== undefined ? tex.texCoord : 0 ;
    }
};

class PbrMetallicRoughness {
    constructor(pbr) {
        this.baseColorFactor    = pbr.baseColorFactor  !== undefined ? pbr.baseColorFactor : [1, 1, 1, 1];
        this.baseColorTexture   = pbr.baseColorTexture !== undefined ? new TextureInfo(pbr.baseColorTexture): null;
        this.metallicFactor     = pbr.metallicFactor   !== undefined ? pbr.metallicFactor  : 1;
        this.roughnessFactor    = pbr.roughnessFactor  !== undefined ? pbr.roughnessFactor : 1;
        //this.metallicRoughnessTexture = pbr.metallicRoughnessTexture !== undefined ? new TextureInfo(pbr.metallicRoughnessTexture): null;
        
        this.extensions = pbr.extensions !== undefined ? pbr.extensions : null;
        this.extras     = pbr.extras     !== undefined ? pbr.extras     : null;
    }
}

class Texture {
    constructor(t, glTF) {
        this.name       = t.name        !== undefined ? t.name : null;
        this.sampler    = t.sampler     !== undefined ? glTF.samplers[t.sampler] : glTF.defaultSampler;
        this.source     = t.source      !== undefined ? glTF.images[t.source]    : null;
        this.extensions = t.extensions  !== undefined ? t.extensions : null;
        this.extras     = t.extras      !== undefined ? t.extras     : null;

        this.texture = null;
    }

    createTexture(gl) {
        this.texture = gl.createTexture();
        gl.bindTexture   (gl.TEXTURE_2D, this.texture);
        gl.texImage2D    (gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.source);
        if (isPowerOf2(this.source.width) && isPowerOf2(this.source.height)) {
            gl.generateMipmap(gl.TEXTURE_2D);
         }        
        gl.bindTexture   (gl.TEXTURE_2D, null);
    }
}

class Sampler {
    constructor(s) {
        if(s){
            this.magFilter  = s.magFilter   !== undefined ? s.magFilter : gl.LINEAR;
            this.minFilter  = s.minFilter   !== undefined ? s.minFilter : gl.NEAREST_MIPMAP_LINEAR;
            this.wrapS      = s.wrapS       !== undefined ? s.wrapS     : gl.REPEAT;
            this.wrapT      = s.wrapT       !== undefined ? s.wrapT     : gl.REPEAT;
            this.extensions = s.extensions  !== undefined ? s.extensions: null;
            this.extras     = s.extras      !== undefined ? s.extras    : null;
   
        } 
        else{
            this.magFilter  = gl.LINEAR;
            this.minFilter  = gl.NEAREST_MIPMAP_LINEAR;
            this.wrapS      = gl.REPEAT;
            this.wrapT      = gl.REPEAT;
            this.extensions = null;
            this.extras     = null;
        }

        this.sampler = null;
    }

    createSampler(gl) {
        this.sampler = gl.createSampler();
        gl.samplerParameteri(this.sampler, gl.TEXTURE_MIN_FILTER, this.minFilter);
        gl.samplerParameteri(this.sampler, gl.TEXTURE_MAG_FILTER, this.magFilter);
        gl.samplerParameteri(this.sampler, gl.TEXTURE_WRAP_S    , this.wrapS);
        gl.samplerParameteri(this.sampler, gl.TEXTURE_WRAP_T    , this.wrapT);
    }
}

class Material {
    constructor(m, matID) {
        this.name       = m.name !== undefined ? m.name : null;
        this.materialID = matID;

        this.pbrMetallicRoughness = m.pbrMetallicRoughness !== undefined ? new PbrMetallicRoughness(m.pbrMetallicRoughness) : new PbrMetallicRoughness({
            baseColorFactor: [1, 1, 1, 1],
            metallicFactor:   1,
            roughnessFactor:  0,
            metallicRoughnessTexture: 1
        });

        // this.normalTexture      = m.normalTexture    !== undefined ? new NormalTextureInfo(m.normalTexture)       : null;
        // this.occlusionTexture   = m.occlusionTexture !== undefined ? new OcclusionTextureInfo(m.occlusionTexture) : null;
        // this.emissiveTexture    = m.emissiveTexture  !== undefined ? new TextureInfo(m.emissiveTexture)           : null;

        this.emissiveFactor = m.emissiveFactor !== undefined ? m.emissiveFactor : [0, 0, 0];
        this.alphaMode      = m.alphaMode      !== undefined ? m.alphaMode      : "OPAQUE";
        this.alphaCutoff    = m.alphaCutoff    !== undefined ? m.alphaCutoff    : 0.5;
        this.doubleSided    = m.doubleSided    ||  false;

        this.extensions = m.extensions !== undefined ? m.extensions : null;
        this.extras     = m.extras !== undefined ? m.extras : null;
    }
}

class SceneNode {
    constructor (n, nodeID, glTF) {
        this.name   = n.name !== undefined ? n.name : null;
        this.nodeID = nodeID;

        this.matrix = glMatrix.mat4.create();
        if (n.hasOwnProperty('matrix')) {
            for(var i = 0; i < 16; ++i) {
                this.matrix[i] = n.matrix[i];
            }

            this.translation = glMatrix.vec3.create();
            glMatrix.mat4.getTranslation(this.translation, this.matrix);

            this.rotation    = glMatrix.quat.create()
            glMatrix.mat4.getRotation(this.rotation, this.matrix);

            this.scale       = glMatrix.vec3.create();
            glMatrix.mat4.getScaling(this.scale, this.matrix);
        } else {
            this.getTransformMatrixFromTRS(n.translation, n.rotation, n.scale);
        }
        this.camera   = n.camera !== undefined ? glTF.cameras[n.camera] : null;
        this.mesh     = n.mesh   !== undefined ? glTF.meshes[n.mesh] : null;
        this.children = n.children || []; 

        this.extensions = n.extensions  !== undefined ? n.extensions : null;
        this.extras     = n.extras      !== undefined ? n.extras     : null;
    }
    
    getTransformMatrixFromTRS = function(translation, rotation, scale) {

        this.translation = translation !== undefined ? glMatrix.vec3.fromValues(translation[0], translation[1], translation[2])           : glMatrix.vec3.fromValues(0, 0, 0);
        this.rotation    = rotation    !== undefined ? glMatrix.vec4.fromValues(rotation[0]   , rotation[1]   , rotation[2], rotation[3]) : glMatrix.vec4.fromValues(0, 0, 0, 1);
        this.scale       = scale       !== undefined ? glMatrix.vec3.fromValues(scale[0]      , scale[1]      , scale[2])                 : glMatrix.vec3.fromValues(1, 1, 1);
    
        this.updateMatrixFromTRS();
    };

    updateMatrixFromTRS = function() {
        var TRSMatrix = glMatrix.mat4.create();
        glMatrix.mat4.fromRotationTranslation(TRSMatrix, this.rotation, this.translation);
        glMatrix.mat4.scale(this.matrix, TRSMatrix, this.scale);
    };
};

class Mesh {
    constructor(mesh, meshID, glTF) {

        this.name   = mesh.name !== undefined ? mesh.name : null;
        this.meshID = meshID;
        
        this.primitives = new Array(mesh.primitives.length);
        for (var i = 0; i < mesh.primitives.length; ++i) {
            this.primitives[i] = new Primitive(mesh.primitives[i], glTF);
        }

        this.extensions = mesh.extensions !== undefined ? mesh.extensions : null;
        this.extras     = mesh.extras     !== undefined ? mesh.extras     : null;
    }
}

class Primitive {
    constructor(p, glTF) {
        this.attributes = p.attributes;
        this.indices    = p.indices !== undefined ? p.indices : null;

        if (this.indices !== null) {
            this.indicesComponentType =  glTF.accessors[this.indices].componentType;
            this.indicesLength        =  glTF.accessors[this.indices].count;
            this.indicesOffset        = (glTF.accessors[this.indices].byteOffset || 0);
        }
        else {
            // assume 'POSITION' is there
            this.drawArraysCount  =  glTF.accessors[this.attributes.POSITION].count;
            this.drawArraysOffset = (glTF.accessors[this.attributes.POSITION].byteOffset || 0);
        }

        this.material   = p.material   !== undefined ? glTF.materials[p.material] : null; 
        this.mode       = p.mode       !== undefined ? p.mode       : gl.TRIANGLES; 
        this.vao        = null;
        this.extensions = p.extensions !== undefined ? p.extensions : null;
        this.extras     = p.extras     !== undefined ? p.extras     : null;
    }
}

class Accessor {
    constructor(a, bufferViewObject) {
        this.bufferView     = bufferViewObject;
        this.componentType  = a.componentType;
        this.byteOffset     = a.byteOffset !== undefined ? a.byteOffset : 0;
        this.byteStride     = bufferViewObject.byteStride;
        this.normalized     = a.normalized !== undefined ? a.normalized : false;
        this.count          = a.count; 
        this.type           = a.type; 
        this.size           = Type2NumOfComponent[this.type];

        this.extensions     = a.extensions !== undefined ? a.extensions : null;
        this.extras         = a.extras !== undefined ? a.extras : null;
    }

    prepareVertexAttrib(gl, location) {
        gl.vertexAttribPointer(
            location,
            this.size,
            this.componentType,
            this.normalized,
            this.byteStride,
            this.byteOffset
        );
        gl.enableVertexAttribArray(location);
    }
}

class BufferView {
    constructor(bf, bufferData) {
        this.byteLength = bf.byteLength; 
        this.byteOffset = bf.byteOffset !== undefined ? bf.byteOffset : 0;
        this.byteStride = bf.byteStride !== undefined ? bf.byteStride : 0;
        this.target     = bf.target     !== undefined ? bf.target     : null;

        this.data       = bufferData.slice(this.byteOffset, this.byteOffset + this.byteLength);

        this.extensions = bf.extensions !== undefined ? bf.extensions   : null;
        this.extras     = bf.extras     !== undefined ? bf.extras       : null;

        this.buffer = null;
    }

    createBuffer(gl) {
        this.buffer = gl.createBuffer();
    }
    
    bindBuffer(gl) {
        if (this.target) {
            gl.bindBuffer(this.target, this.buffer);
        }
    }
    unbindBuffer(gl) {
        if (this.target) {
            gl.bindBuffer(this.target, null);
        }
    }
    bufferData(gl) {
        if (this.target) {
            gl.bufferData(this.target, this.data, gl.STATIC_DRAW);
        }
    }
}

class glTFModel {
    constructor(gltf) {

        this.json    = gltf;
        this.version = Number(gltf.asset.version);

        if (gltf.accessors) {
            this.accessors = new Array(gltf.accessors.length);
        }

        if (gltf.bufferViews) {
            this.bufferViews = new Array(gltf.bufferViews.length);
        }

        if (gltf.scenes) {
            this.scenes = new Array(gltf.scenes.length); 
        }

        if (gltf.nodes) {
            this.nodes = new Array(gltf.nodes.length); 
        }

        if (gltf.meshes) {
            this.meshes = new Array(gltf.meshes.length); 
        }

        if (gltf.materials) {
            this.materials = new Array(gltf.materials.length); 
        }

        if (gltf.textures) {
            this.textures = new Array(gltf.textures.length);
        }

        if (gltf.samplers) {
            this.samplers = new Array(gltf.samplers.length);
        }

        if (gltf.images) {
            this.images = new Array(gltf.images.length);
        }

        if (gltf.skins) {
            this.skins = new Array(gltf.skins.length);
        }

        if (gltf.animations) {
            this.animations = new Array(gltf.animations.length);
        }

        if (gltf.cameras) {
            this.cameras = new Array(gltf.cameras.length);
        }

        this.extensions = gltf.extensions   !== undefined ? gltf.extensions : null;
        this.extras     = gltf.extras       !== undefined ? gltf.extras     : null;
    }
}

class glTFLoader {
    
    constructor(gl) {
        // Set context
        this.gl = gl !== undefined ? gl : null;
        
        // Init
        this._init();
        this.glTF = null;
    }

    _init = function() {
  
        this._bufferRequested   = 0;
        this._bufferLoaded      = 0;
        this._buffers           = [];
        this._bufferTasks       = {};
    
        this._shaderRequested   = 0;
        this._shaderLoaded      = 0;
    
        this._imageRequested    = 0;
        this._imageLoaded       = 0;
    
        this._pendingTasks      = 0;
        this._finishedPendingTasks = 0;

        this._loadDone  = false;
        this.onload     = null;
    };

    load_GLTF = function(uri, callback) {

        // init
        this._init();

        // set onload function
        this.onload = callback || function(glTF_file) {
            console.log('glTF file loaded.');
            console.log(glTF_file);
        };
        
        this.baseUri = _getBaseUri(uri);

        var loader = this;
        _loadJSON(uri, function (response)
        {
            // Parse JSON string into object
            var json    = JSON.parse(response);

            // Constructr gltfModel from json
            loader.glTF = new glTFModel(json);

            // load buffers
            var bid;
            var loadArrayBufferCallback = function (resource) {
                loader._bufferLoaded++;
                loader._buffers[bid] = resource;
                if (loader._bufferTasks[bid]) {
                    var i, len;
                    for (i = 0, len = loader._bufferTasks[bid].length; i < len; ++i) {
                        (loader._bufferTasks[bid][i])(resource);
                    }
                }
                loader._checkComplete();
            };

            if (json.buffers) {
                for (bid in json.buffers) {
                    loader._bufferRequested++;
                    _loadArrayBuffer(loader.baseUri + json.buffers[bid].uri, loadArrayBufferCallback);
                }
            }

            // load images
            var loadImageCallback = function (img, iid) {
                loader._imageLoaded++;
                loader.glTF.images[iid] = img;
                loader._checkComplete();
            };

            var iid;
            if (json.images) {
                for (iid in json.images) {
                    loader._imageRequested++;
                    _loadImage(loader.baseUri + json.images[iid].uri, iid, loadImageCallback);
                }
            }

            loader._checkComplete();
        });
    };

    _checkComplete = function ()
    {
        if (this._bufferRequested == this._bufferLoaded && 
            this._shaderRequested == this._shaderLoaded && 
            this._imageRequested  == this._imageLoaded 
            ) {
            this._loadDone = true;
        }
    
        if (this._loadDone && this._pendingTasks == this._finishedPendingTasks) {
            this._postprocess();
            this.onload(this.glTF);
        }
    };
    
    _postprocess = function ()
    {
        //console.log('finish loading all assets, do a second pass postprocess');

        var i,j;
        // load bufferviews
        if (this.glTF.bufferViews) {
            for (i = 0; i < this.glTF.bufferViews.length; i++) {
                this.glTF.bufferViews[i] = new BufferView(this.glTF.json.bufferViews[i], this._buffers[ this.glTF.json.bufferViews[i].buffer ]);
            }
        }

        // load samplers 
        this.glTF.defaultSampler = new Sampler(null);
        if (this.glTF.samplers) {
            for (i = 0; i < this.glTF.samplers.length; i++) {
                this.glTF.samplers[i] = new Sampler(this.glTF.json.samplers[i]);
            } 
        }

        // load textures
        if (this.glTF.textures) {
            for (i = 0; i < this.glTF.textures.length; i++) {
                this.glTF.textures[i] = new Texture(this.glTF.json.textures[i], this.glTF);
            }
        }

        // load accessors
        if (this.glTF.accessors) {
            for (i = 0; i < this.glTF.accessors.length; i++) {
                this.glTF.accessors[i] = new Accessor(this.glTF.json.accessors[i], this.glTF.bufferViews[ this.glTF.json.accessors[i].bufferView ]);
            }
        }

        // load cameras
        if (this.glTF.cameras) {
            for (i = 0; i < this.glTF.cameras.length; i++) {
                this.glTF.cameras[i] = new Camera(this.glTF.json.cameras[i]);
            }
        }

        // load materials
        if (this.glTF.materials) {
            for (i = 0; i < this.glTF.materials.length; i++) {
                this.glTF.materials[i] = new Material(this.glTF.json.materials[i], i);
            }
        }

        // load meshes
        for (i = 0; i < this.glTF.meshes.length; i++) {
            this.glTF.meshes[i] = new Mesh(this.glTF.json.meshes[i], i, this.glTF);
        }

        // node
        for (i = 0; i < this.glTF.nodes.length; i++) {
            this.glTF.nodes[i] = new SceneNode(this.glTF.json.nodes[i], i, this.glTF);
        }

        // node children
        for (i = 0; i < this.glTF.nodes.length; i++)
            for (j = 0; j < this.glTF.nodes[i].children.length; j++) {
                this.glTF.nodes[i].children[j] = this.glTF.nodes[ this.glTF.nodes[i].children[j] ];
            }
        
        // scenes
        for (i = 0; i < this.glTF.scenes.length; i++) {
            this.glTF.scenes[i] = new Scene(this.glTF.json.scenes[i], i, this.glTF.nodes);
        }
    };
}

// for animation use
function _arrayBuffer2TypedArray(buffer, byteOffset, countOfComponentType, componentType) {
    switch(componentType) {
        // @todo: finish
        case 5122: return new Int16Array(buffer, byteOffset, countOfComponentType);
        case 5123: return new Uint16Array(buffer, byteOffset, countOfComponentType);
        case 5124: return new Int32Array(buffer, byteOffset, countOfComponentType);
        case 5125: return new Uint32Array(buffer, byteOffset, countOfComponentType);
        case 5126: return new Float32Array(buffer, byteOffset, countOfComponentType);
        default: return null; 
    }
}

function _getAccessorData(accessor) {
    return _arrayBuffer2TypedArray(
        accessor.bufferView.data, 
        accessor.byteOffset, 
        accessor.count * Type2NumOfComponent[accessor.type],
        accessor.componentType
        );
}

function _getBaseUri(uri) {
    
    // https://github.com/AnalyticalGraphicsInc/cesium/blob/master/Source/Core/getBaseUri.js
    var basePath = '';
    var i = uri.lastIndexOf('/');
    if(i !== -1) {
        basePath = uri.substring(0, i + 1);
    }   
    return basePath;
}

function _loadJSON(src, callback) {

    // native json loading technique from @KryptoniteDove:
    // http://codepen.io/KryptoniteDove/post/load-json-file-locally-using-pure-javascript

    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', src, true);
    xobj.onreadystatechange = function () {
        if (xobj.readyState == 4 && // Request finished, response ready
            xobj.status == "200") { // Status OK
            callback(xobj.responseText, this);
        }
    };
    xobj.send(null);
}

function _loadArrayBuffer(url, callback) {
    var xobj = new XMLHttpRequest();
    xobj.responseType = 'arraybuffer';
    xobj.open('GET', url, true);
    xobj.onreadystatechange = function () {
        if (xobj.readyState == 4 && // Request finished, response ready
            xobj.status == "200") { // Status OK
            var arrayBuffer = xobj.response;
            if (arrayBuffer && callback) {
                callback(arrayBuffer);
            }
        }
    };
    xobj.send(null);
}

function _loadImage(url, iid, onload) {
    var img = new Image();
    img.onload = function() {
        onload(img, iid);
    };
    img.crossOrigin = "Anonymous";
    img.src = url;
}

function isPowerOf2(value) {
    return (value & (value - 1)) == 0;
}