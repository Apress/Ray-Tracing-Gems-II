//import { glMatrix } from "./deps/gl-matrix"

/*export function webrays_create_FPS_camera()
{
  return {
    type: "FPS",
    fov_degrees: 60,
    znear: 0.01,
    zfar: 1000.0,
    camera_pos: [0,0,0],
    camera_direction: [0,0,-1],
    camera_right: [1,0,0],
    camera_up: [0, 1, 0],
    view: glMatrix.mat4.create()
  }
};*/

class FreeRoamPerspectiveCamera {
  constructor(width, height, fov=60.0)
  {
    // Camera Properties
    this.width = width;
    this.height = height;
    this.FOV = fov;
    this.zNear = 0.1;
    this.zFar = 1000.0;
    this.camera_pos = glMatrix.vec3.fromValues(0, 20, 50);
    this.camera_front = glMatrix.vec3.normalize(glMatrix.vec3.create(), glMatrix.vec3.fromValues(10, 0,-50));
    this.camera_up = glMatrix.vec3.fromValues(0,1,0);
    this.setSize(width, height);
    this.recomputeViewMatrix();
    this.recomputeViewMatrix();
    
    // Mouse States
    this.mouse           = glMatrix.vec2.fromValues(0, 0);
    this.last_mouse      = glMatrix.vec2.fromValues(0, 0);
    this.arcball_on      = false; 
    this.need_update     = true;
  }

  setSize(width, height)
  {
    this.width = width;
    this.height = height;
    this.aspectRatio = width / height;

    this.recomputePerspectiveMatrix();   
  }

  setNearFar(near, far) {
    this.zNear = near;
    this.zFar = far;
    recomputePerspectiveMatrix();
  }

  recomputePerspectiveMatrix() {
    const fieldOfView = this.FOV * Math.PI / 180;   // in radians
    
    this.projection = glMatrix.mat4.perspective(glMatrix.mat4.create(),
      fieldOfView,
      this.aspectRatio,
      this.zNear, this.zFar);
  }

  recomputeViewMatrix() {
    const at = glMatrix.vec3.add(glMatrix.vec3.create(), this.camera_front, this.camera_pos);
    // View Matrix
    this.view = glMatrix.mat4.lookAt( 
      glMatrix.mat4.create(),
      this.camera_pos, // pos
      at, // at
      this.camera_up // up
    ); // view is [right, up, forward, -pos]^T;

    // Decode Camera View Axis
    this.camera_right = glMatrix.vec3.fromValues( this.view[0], this.view[4], this.view[8] );
    this.camera_up    = glMatrix.vec3.fromValues( this.view[1], this.view[5], this.view[9] );
    this.camera_front = glMatrix.vec3.fromValues( -this.view[2], -this.view[6], -this.view[10] );
    this.need_update  = true;
  }

  mouseMove(mouse)
  {
    if (this.arcball_on) {
      this.mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
    }
  }

  mouseUp(mouse) 
  {
    switch( mouse.button ) {
      case 0:
        this.arcball_on = false;
        break;
      case 1:
        break;
      case 2:
        break;  
      default:
        break;
    }
  }

  mouseDown(mouse)
  {
    switch( mouse.button ) {
      case 0:
        this.arcball_on = true;
        this.mouse = this.last_mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
        break;
      case 1:
        break;
      case 2:
        break;
      default:
        break;
    }
  }

  touchStart(touch)
  {
    this.arcball_on = true;
    this.mouse = this.last_mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
  }

  touchEnd(touch) {
    this.arcball_on = false;
  }
  
  touchMove(touch) {
    if (this.arcball_on) {
      this.mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
    }
  }

  keyDown(key) {
    const code = key.keyCode;
    switch (code) {
      case 65:  glMatrix.vec3.sub(this.camera_pos, this.camera_pos, this.camera_right); break;
      case 87:  glMatrix.vec3.add(this.camera_pos, this.camera_pos, this.camera_front); break;
      case 83:  glMatrix.vec3.sub(this.camera_pos, this.camera_pos, this.camera_front); break;
      case 68:  glMatrix.vec3.add(this.camera_pos, this.camera_pos, this.camera_right); break;
      case 37:  break; //Left key
      case 38:  break; //Up key
      case 39:  break; //Right key
      case 40:  break; //Down key
      default:  break; //Everything else
    }
  
    this.recomputeViewMatrix();
  }

  update() {
    const ret_value = this.need_update;
    this.need_update     = false;
    if (this.mouse[0] == this.last_mouse[0] && 
        this.mouse[1] == this.last_mouse[1])
      return ret_value;
      
    var delta = glMatrix.vec2.fromValues(1, 1);
    const mouse_delta = glMatrix.vec2.sub(glMatrix.vec2.create(), this.last_mouse, this.mouse);
    delta = glMatrix.vec2.mul(glMatrix.vec2.create(), delta, mouse_delta);
    delta[0] *= 0.004; // NEED to multiply with dt
    delta[1] *= 0.002;// NEED to multiply with dt
  
    this.last_mouse[0] = this.mouse[0];
    this.last_mouse[1] = this.mouse[1];
  
    //console.log("Start");
          
    let camera_at = glMatrix.vec3.clone(this.camera_front);
    let rot_x = glMatrix.mat4.fromRotation(glMatrix.mat4.create(), delta[0], this.camera_up);
    let rot_y = glMatrix.mat4.fromRotation(glMatrix.mat4.create(), delta[1], this.camera_right);
    let rot = glMatrix.mat4.multiply(glMatrix.mat4.create(), rot_x, rot_y);
    
    camera_at = glMatrix.vec3.transformMat4(glMatrix.vec3.create(), camera_at, rot);
      
    camera_at = glMatrix.vec3.add(glMatrix.vec3.create(), camera_at, this.camera_pos);
    this.view = glMatrix.mat4.lookAt( 
      glMatrix.mat4.create(),
      this.camera_pos, // pos
      camera_at, // at
      this.camera_up // up
    ); // view is [right, up, -forward, -pos]^T;
    
    this.camera_right = glMatrix.vec3.fromValues( this.view[0], this.view[4], this.view[8] );
    this.camera_up = glMatrix.vec3.fromValues( this.view[1], this.view[5], this.view[9] );
    this.camera_front = glMatrix.vec3.fromValues( -this.view[2], -this.view[6], -this.view[10] );
      
    //WebRaysViewer.last_mouse = WebRaysViewer.mouse;
    return true;
  }


  attachControls(canvas) {
    const call_during_capture = true;
    canvas.addEventListener("mousemove", (ev) => this.mouseMove(ev), call_during_capture);
    canvas.addEventListener("mouseup", (ev) => {this.mouseUp(ev);}, call_during_capture);
    canvas.addEventListener("mousedown", (ev) => {this.mouseDown(ev);}, call_during_capture);
    canvas.addEventListener("touchstart", ev => {this.touchStart(ev);}, false);
    canvas.addEventListener("touchend", (ev) => {this.touchEnd(ev);}, false);
    //canvas.addEventListener("touchcancel", handleCancel, false);
    canvas.addEventListener("touchmove", (ev) => {this.touchMove(ev);}, false);
    window.addEventListener('keydown', (ev) => {this.keyDown(ev);}, call_during_capture);
  }
}

const FreeRoamPerspectiveCamera2 = (width, height, fov) => {

  let arcball_on = false;
  let mouse = null;

  const mouseMove = (ev) => {
    console.log(this, mouse);
    if (arcball_on) {
      mouse = glMatrix.vec2.fromValues(ev.offsetX, ev.offsetY);
    }
  }

  const mouseUp = (mouse) =>
  {
    console.log(this, mouse);
    switch( mouse.button ) {
      case 0:
        arcball_on = false;
        break;
      case 1:
        break;
      case 2:
        break;  
      default:
        break;
    }
  }

  return ({
    width : width,
    height : height,
    FOV : fov,
    zNear : 0.1,
    zFar : 1000.0,
    camera_pos : glMatrix.vec3.create(),
    camera_front : glMatrix.vec3.create(),
    camera_up : glMatrix.vec3.fromValues(0,1,0),
    view : glMatrix.mat4.create(),
  
    setSize(width, height)
    {
      width = width;
      height = height;
      aspectRatio = width / height;
      
      recomputePerspectiveMatrix();   
    },

    setNearFar(near, far) {
      zNear = near;
      zFar = far;
      recomputePerspectiveMatrix();
    },

    recomputePerspectiveMatrix() {
      projection = glMatrix.mat4.perspective(glMatrix.mat4.create(),
        FOV,
        aspectRatio,
        zNear, zFar);
    },

    recomputeViewMatrix() {
      const at = glMatrix.vec3.add(glMatrix.vec3.create(), camera_front, camera_pos);
      view = glMatrix.mat4.lookAt( 
        glMatrix.mat4.create(),
        camera_pos, // pos
        at, // at
        camera_up // up
      ); // view is [right, up, forward, -pos]^T;
    },

    mouseDown(mouse)
    {
      console.log(this, mouse);
      console.log(view);
      switch( mouse.button ) {
        case 0:
          arcball_on = true;
          mouse = last_mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
          break;
        case 1:
          break;
        case 2:
          break;
        default:
          break;
      }
    },

    touchStart(touch)
    {
      arcball_on = true;
      mouse = last_mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
    },

    touchEnd(touch) {
      arcball_on = false;
    },
  
    touchMove(touch) {
      if (arcball_on) {
        mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
      }
    },

    keyDown(key) {
      console.log(camera_pos);
      const code = key.keyCode;
      switch (code) {
        case 65:  glMatrix.vec3.sub(camera_pos, camera_pos, camera_right); break;
        case 87:  glMatrix.vec3.add(camera_pos, camera_pos, camera_front); break;
        case 83:  glMatrix.vec3.sub(camera_pos, camera_pos, camera_front); break;
        case 68:  glMatrix.vec3.add(camera_pos, camera_pos, camera_right); break;
        case 37:  break; //Left key
        case 38:  break; //Up key
        case 39:  break; //Right key
        case 40:  break; //Down key
        default:  break; //Everything else
      }
      
      recomputeViewMatrix();
    },

    attachControls(canvas) {
      const call_during_capture = true;
      canvas.addEventListener("mousemove", mouseMove, call_during_capture);
      canvas.addEventListener("mouseup", mouseUp, call_during_capture);
      canvas.addEventListener("mousedown", this.mouseDown, call_during_capture);
      canvas.addEventListener("touchstart", this.touchStart, false);
      /*canvas.addEventListener("touchend", touchEnd, false);
      //canvas.addEventListener("touchcancel", handleCancel, false);
      canvas.addEventListener("touchmove", touchMove, false);*/
      canvas.addEventListener('keydown', this.keyDown, call_during_capture);

    }
  });
}

class FreeRoamPerspectiveCamera3 {
  constructor(width, height, fov=60.0)
  {
    // Camera Properties
    this.width = width;
    this.height = height;
    this.FOV = fov;
    this.zNear = 0.1;
    this.zFar = 100.0;
    this.camera_pos = glMatrix.vec3.fromValues(0, 20, 50);
    this.camera_front = glMatrix.vec3.normalize(glMatrix.vec3.create(), glMatrix.vec3.fromValues(10, 0,-50));
    this.camera_up = glMatrix.vec3.fromValues(0,1,0);

    this.sceneBBOX = null; // Else {minV: [], maxV: []}
    this.angles = [-Math.atan2(this.camera_front[2], this.camera_front[0]), 0]; // Radians
    this.moveSpeed = 1.0;
    this.setSize(width, height);
    this.recomputeViewMatrix();    
    
    // Mouse States
    this.mouse           = glMatrix.vec2.fromValues(0, 0);
    this.last_mouse      = glMatrix.vec2.fromValues(0, 0);
    this.arcball_on      = false; 
    this.need_update     = true;
  }

  setSize(width, height)
  {
    this.width = width;
    this.height = height;
    this.aspectRatio = width / height;

    this.recomputePerspectiveMatrix();   
  }

  setNearFar(near, far) {
    this.zNear = near;
    this.zFar = far;
    this.recomputePerspectiveMatrix();
  }

  setSceneBBOX(minV, maxV)
  {
    this.sceneBBOX = { minV: minV, maxV: maxV};
    const diagonalV = [maxV[0] - minV[0], maxV[1] - minV[1], maxV[2] - minV[2]];
    const diagonal_length = Math.sqrt(diagonalV[0] * diagonalV[0] + diagonalV[1] * diagonalV[1] + diagonalV[2] * diagonalV[2]);

    this.moveSpeed = Math.max(diagonal_length, 1.0) / 150.0;
  }

  setPosition(x,y,z) {
    this.camera_pos = glMatrix.vec3.fromValues(x,y,z);
    this.recomputeViewMatrix();
  }

  setTarget(x,y,z) {
    const camera_target = glMatrix.vec3.fromValues(x,y,z);
    this.camera_front = glMatrix.vec3.sub(glMatrix.vec3.create(), camera_target, this.camera_pos);
    this.camera_front = glMatrix.vec3.normalize(this.camera_front, this.camera_front);

    const half_pi = 0.5 * Math.PI;
    const d2 = Math.abs(this.camera_front[1]) > 0.99? 
      Math.sign(this.camera_front[1]) * half_pi : 
      Math.sqrt(this.camera_front[0] * this.camera_front[0] + this.camera_front[2] * this.camera_front[2]);
		this.angles = [
      -Math.atan2(this.camera_front[2], this.camera_front[0]), 
      Math.atan(this.camera_front[1] / d2)
    ]; // Radians

    // The Right vector is on the right of the view vector
    const theta = this.angles[1];
    const phi = -this.angles[0];
	  const d = phi + 0.5 * Math.PI;
	  let right = glMatrix.vec3.fromValues(Math.cos(d), 0.0, Math.sin(d));
    right = glMatrix.vec3.normalize(right, right);
    // Compute the up vector
	  this.camera_up = glMatrix.vec3.cross(glMatrix.vec3.create(), right, this.camera_front);
    this.camera_up = glMatrix.vec3.normalize(this.camera_up, this.camera_up);

    this.recomputeViewMatrix();
  }

  recomputePerspectiveMatrix() {
    const fieldOfView = this.FOV * Math.PI / 180;   // in radians
    
    this.projection = glMatrix.mat4.perspective(glMatrix.mat4.create(),
      fieldOfView,
      this.aspectRatio,
      this.zNear, this.zFar);
  }

  recomputeViewMatrix() {
    const at = glMatrix.vec3.add(glMatrix.vec3.create(), this.camera_front, this.camera_pos);
    // View Matrix
    this.view = glMatrix.mat4.lookAt( 
      glMatrix.mat4.create(),
      this.camera_pos, // pos
      at, // at
      this.camera_up // up
    ); // view is [right, up, forward, -pos]^T;

    // Decode Camera View Axis
    this.camera_right = glMatrix.vec3.fromValues( this.view[0], this.view[4], this.view[8] );
    this.camera_up    = glMatrix.vec3.fromValues( this.view[1], this.view[5], this.view[9] );
    this.camera_front = glMatrix.vec3.fromValues( -this.view[2], -this.view[6], -this.view[10] );
    this.need_update  = true;
    this.fitNearFarPlane();
  }

  fitNearFarPlane() {
    if(this.sceneBBOX === null)
      return;
    const scene_size = [this.sceneBBOX.maxV[0] - this.sceneBBOX.minV[0], this.sceneBBOX.maxV[1] - this.sceneBBOX.minV[1], this.sceneBBOX.maxV[2] - this.sceneBBOX.minV[2]];
    const scene_centerA = [
      0.5 * (this.sceneBBOX.maxV[0] + this.sceneBBOX.minV[0]), 
      0.5 * (this.sceneBBOX.maxV[1] + this.sceneBBOX.minV[1]), 
      0.5 * (this.sceneBBOX.maxV[2] + this.sceneBBOX.minV[2])
    ];
    const scene_center = glMatrix.vec3.fromValues(scene_centerA[0], scene_centerA[1], scene_centerA[2]);
    const scene_radius = 0.5 * glMatrix.vec3.dot(scene_size, scene_size);
    const scene_max_side = Math.max(...scene_size);
    const units_scale = 0.1;

    let new_near = 0.13;
    let new_far = 133;

    const {minV, maxV} = this.sceneBBOX;

    let corner = glMatrix.vec3.create();
    let center_ecs = glMatrix.vec3.transformMat4(glMatrix.vec3.create(), this.camera_pos, this.view);

		let cur_far = -Number.MAX_VALUE;
		let cur_near = Number.MAX_VALUE;
		let changed = false;
		for (let i = 0; i < 8; ++i)
		{
      corner = glMatrix.vec3.fromValues(i & 1 ? maxV[0] : minV[0], i & 2 ? maxV[1] : minV[1], i & 4 ? maxV[2] : minV[2]);
			corner = glMatrix.vec3.transformMat4(glMatrix.vec3.create(), corner, this.view);
			
			if (corner[2] >= 0.0)
			{
				cur_near = units_scale;
				continue;
			}

			let f = -corner[2];
			if (f < cur_near)
			{
				cur_near = f;
				changed = true;
			}
			if (f > cur_far)
			{
				cur_far = f;
				changed = true;
			}
		}

		let center_box_ecs = glMatrix.vec3.transformMat4(glMatrix.vec3.create(), scene_center, this.view);
		let dist_to_bbox_center = glMatrix.vec3.distance(center_ecs, center_box_ecs);
		let inside_bounding_sphere = dist_to_bbox_center <= scene_radius;

		if (inside_bounding_sphere)
		{
			new_near = units_scale;
			new_far = new_near + cur_far;
		}
		else
		{
			if (changed)
			{
				new_near = Math.max(units_scale, cur_near);
				new_far = new_near + cur_far;
			}
			else
			{
				// if we are outside the bounding sphere looking at nothing, just set a default near far
				new_near = units_scale;
				new_far = new_near + 1.0;
			}
    }
    
    new_near -= 0.1 + 0.1 * scene_max_side;//getSceneBox().getMaxSide();
	  new_near = Math.max(new_near, 0.1 /*unit scales*/);

    new_far += 0.1;
    this.setNearFar(new_near, new_far);
  }

  mouseMove(mouse)
  {
    if (this.arcball_on) {
      this.mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
    }
  }

  mouseUp(mouse) 
  {
    switch( mouse.button ) {
      case 0:
        this.arcball_on = false;
        break;
      case 1:
        break;
      case 2:
        break;  
      default:
        break;
    }
  }

  mouseDown(mouse)
  {
    switch( mouse.button ) {
      case 0:
        this.arcball_on = true;
        this.mouse = this.last_mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
        break;
      case 1:
        break;
      case 2:
        break;
      default:
        break;
    }
  }

  touchStart(touch)
  {
    this.arcball_on = true;
    this.mouse = this.last_mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
  }

  touchEnd(touch) {
    this.arcball_on = false;
  }
  
  touchMove(touch) {
    if (this.arcball_on) {
      this.mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
    }
  }

  keyDown(key) {
    const code = key.keyCode;
    const rightOffset = glMatrix.vec3.multiply(glMatrix.vec3.create(), this.camera_right, glMatrix.vec3.fromValues(this.moveSpeed,this.moveSpeed,this.moveSpeed));
    const frontOffset = glMatrix.vec3.multiply(glMatrix.vec3.create(), this.camera_front, glMatrix.vec3.fromValues(this.moveSpeed,this.moveSpeed,this.moveSpeed));
    const upOffset = glMatrix.vec3.multiply(glMatrix.vec3.create(), this.camera_up, glMatrix.vec3.fromValues(this.moveSpeed,this.moveSpeed,this.moveSpeed));
    let updateMatrix = true;
    switch (code) {
      case 65:  glMatrix.vec3.sub(this.camera_pos, this.camera_pos, rightOffset); break;
      case 87:  glMatrix.vec3.add(this.camera_pos, this.camera_pos, frontOffset); break;
      case 83:  glMatrix.vec3.sub(this.camera_pos, this.camera_pos, frontOffset); break;
      case 68:  glMatrix.vec3.add(this.camera_pos, this.camera_pos, rightOffset); break;
      case 80:  console.log(this.camera_pos, this.camera_front); break; // p
      case 90:  glMatrix.vec3.add(this.camera_pos, this.camera_pos, upOffset); break;
      case 88:  glMatrix.vec3.sub(this.camera_pos, this.camera_pos, upOffset); break;
      case 37:  break; //Left key
      case 38:  break; //Up key
      case 39:  break; //Right key
      case 40:  break; //Down key
      default:  updateMatrix = false; break; //Everything else
    }
  
    if(updateMatrix)
      this.recomputeViewMatrix();
  }

  update() {
    const ret_value   = this.need_update;
    this.need_update  = false;
    if (this.mouse[0] == this.last_mouse[0] && 
        this.mouse[1] == this.last_mouse[1])
      return ret_value;
      
    var delta = glMatrix.vec2.fromValues(1, 1);
    const mouse_delta = glMatrix.vec2.sub(glMatrix.vec2.create(), this.last_mouse, this.mouse);
    delta = glMatrix.vec2.mul(glMatrix.vec2.create(), delta, mouse_delta);
    delta[0] *= 0.004; // NEED to multiply with dt
    delta[1] *= 0.002;// NEED to multiply with dt
  
    this.last_mouse[0] = this.mouse[0];
    this.last_mouse[1] = this.mouse[1];
  
    //console.log("Start");

    this.angles[0] += delta[0]; // X axis 
    this.angles[1] += delta[1]; // Y axis 
    
    const halfPI = Math.PI * 0.495;
    this.angles[1] = Math.max(Math.min(this.angles[1], halfPI), -halfPI);          
    
    let camera_at = glMatrix.vec3.fromValues(0,0,-1);
    
    // Spherical Coordinates
    const theta = this.angles[1];
    const phi = -this.angles[0];
    camera_at[0] = Math.cos(theta) * Math.cos(phi);
	  camera_at[1] = Math.sin(theta);
	  camera_at[2] = Math.cos(theta) * Math.sin(phi);
    camera_at = glMatrix.vec3.normalize(camera_at, camera_at);
    // The Right vector is on the right of the view vector
	  const d = phi + 0.5 * Math.PI;
	  let right = glMatrix.vec3.fromValues(Math.cos(d), 0.0, Math.sin(d));
    right = glMatrix.vec3.normalize(right, right);
    // Compute the up vector
	  this.camera_up = glMatrix.vec3.cross(glMatrix.vec3.create(), right, camera_at);
	  this.camera_up = glMatrix.vec3.normalize(this.camera_up, this.camera_up);
    
    camera_at = glMatrix.vec3.add(glMatrix.vec3.create(), camera_at, this.camera_pos);
    this.view = glMatrix.mat4.lookAt( 
      glMatrix.mat4.create(),
      this.camera_pos, // pos
      camera_at, // at
      this.camera_up // up
    ); // view is [right, up, -forward, -pos]^T;
    
    this.camera_right = glMatrix.vec3.fromValues( this.view[0], this.view[4], this.view[8] );
    this.camera_up = glMatrix.vec3.fromValues( this.view[1], this.view[5], this.view[9] );
    this.camera_front = glMatrix.vec3.fromValues( -this.view[2], -this.view[6], -this.view[10] );
      
    //WebRaysViewer.last_mouse = WebRaysViewer.mouse;
    return true;
  }


  attachControls(canvas) {
    const call_during_capture = true;
    canvas.addEventListener("mousemove", (ev) => this.mouseMove(ev), call_during_capture);
    canvas.addEventListener("mouseup", (ev) => {this.mouseUp(ev);}, call_during_capture);
    canvas.addEventListener("mousedown", (ev) => {this.mouseDown(ev);}, call_during_capture);
    canvas.addEventListener("touchstart", ev => {this.touchStart(ev);}, false);
    canvas.addEventListener("touchend", (ev) => {this.touchEnd(ev);}, false);
    //canvas.addEventListener("touchcancel", handleCancel, false);
    canvas.addEventListener("touchmove", (ev) => {this.touchMove(ev);}, false);
    window.addEventListener('keydown', (ev) => {this.keyDown(ev);}, call_during_capture);
  }
}

class OrbitPerspectiveCamera {
  constructor(width, height, fov=60.0)
  {
    // Camera Properties
    this.width = width;
    this.height = height;
    this.FOV = fov;
    this.zNear = 0.1;
    this.zFar = 100.0;
    this.camera_pos = glMatrix.vec3.fromValues(0, 20, 50);
    this.camera_target = glMatrix.vec3.fromValues(0, 20, 0);
    this.camera_front = glMatrix.vec3.normalize(glMatrix.vec3.create(), glMatrix.vec3.subtract(glMatrix.vec3.create(), this.camera_target, this.camera_pos));
    this.camera_up = glMatrix.vec3.fromValues(0,1,0);
    
    this.sceneBBOX = null; // Else {minV: [], maxV: []}
    this.angles = [Math.atan2(this.camera_front[2], this.camera_front[0]), 0]; // Radians
    this.zoomLevel = 50.0;
    this.zoomBounds = [0.1, 1000];
    this.setSize(width, height);
    this.recomputeViewMatrix();
    
    // Mouse States
    this.mouse           = glMatrix.vec2.fromValues(0, 0);
    this.last_mouse      = glMatrix.vec2.fromValues(0, 0);
    this.arcball_on      = false; 
    this.need_update     = true;
  }

  setSize(width, height)
  {
    this.width = width;
    this.height = height;
    this.aspectRatio = width / height;

    this.recomputePerspectiveMatrix();   
  }

  setNearFar(near, far) {
    this.zNear = near;
    this.zFar = far;
    this.recomputePerspectiveMatrix();
  }

  setSceneBBOX(minV, maxV)
  {
    this.sceneBBOX = { minV: minV, maxV: maxV};
  }

  setPosition(x,y,z) {
    this.camera_pos = glMatrix.vec3.fromValues(x,y,z);
    this.camera_front = glMatrix.vec3.sub(glMatrix.vec3.create(), this.camera_target, this.camera_pos);
    this.zoomLevel = glMatrix.vec3.length(this.camera_front);
    this.camera_front = glMatrix.vec3.normalize(this.camera_front, this.camera_front);

    const half_pi = 0.5 * Math.PI;
    const d2 = Math.abs(this.camera_front[1]) > 0.99? 
      Math.sign(this.camera_front[1]) * half_pi : 
      Math.sqrt(this.camera_front[0] * this.camera_front[0] + this.camera_front[2] * this.camera_front[2]);
		this.angles = [
      Math.atan2(this.camera_front[2], this.camera_front[0]), 
      -Math.atan(this.camera_front[1] / d2)
    ]; // Radians

    // The Right vector is on the right of the view vector
    const theta = this.angles[1];
    const phi = -this.angles[0];
	  const d = phi - 0.5 * Math.PI;
	  let right = glMatrix.vec3.fromValues(Math.cos(d), 0.0, Math.sin(d));
    right = glMatrix.vec3.normalize(right, right);
    // Compute the up vector
	  this.camera_up = glMatrix.vec3.cross(glMatrix.vec3.create(), right, this.camera_front);
    this.camera_up = glMatrix.vec3.normalize(this.camera_up, this.camera_up);

    this.recomputeViewMatrix();  
  }

  setTarget(x,y,z) {
    this.camera_target = glMatrix.vec3.fromValues(x,y,z);
    this.camera_front = glMatrix.vec3.sub(glMatrix.vec3.create(), this.camera_target, this.camera_pos);
    this.camera_front = glMatrix.vec3.normalize(this.camera_front, this.camera_front);

    const half_pi = 0.5 * Math.PI;
    const d2 = Math.abs(this.camera_front[1]) > 0.99? 
      Math.sign(this.camera_front[1]) * half_pi : 
      Math.sqrt(this.camera_front[0] * this.camera_front[0] + this.camera_front[2] * this.camera_front[2]);
		this.angles = [
      Math.atan2(this.camera_front[2], this.camera_front[0]), 
      -Math.atan(this.camera_front[1] / d2)
    ]; // Radians

    // The Right vector is on the right of the view vector
    const theta = this.angles[1];
    const phi = -this.angles[0];
	  const d = phi - 0.5 * Math.PI;
	  let right = glMatrix.vec3.fromValues(Math.cos(d), 0.0, Math.sin(d));
    right = glMatrix.vec3.normalize(right, right);
    // Compute the up vector
	  this.camera_up = glMatrix.vec3.cross(glMatrix.vec3.create(), right, this.camera_front);
    this.camera_up = glMatrix.vec3.normalize(this.camera_up, this.camera_up);

    this.recomputeViewMatrix();
  }

  recomputePerspectiveMatrix() {
    const fieldOfView = this.FOV * Math.PI / 180;   // convert to radians
    
    this.projection = glMatrix.mat4.perspective(glMatrix.mat4.create(),
      fieldOfView,
      this.aspectRatio,
      this.zNear, this.zFar);
  }

  recomputeViewMatrix() {
    // View Matrix
    this.view = glMatrix.mat4.lookAt( 
      glMatrix.mat4.create(),
      this.camera_pos, // pos
      this.camera_target, // at
      this.camera_up // up
    ); // view is [right, up, forward, -pos]^T;

    // Decode Camera View Axis
    this.camera_right = glMatrix.vec3.fromValues( this.view[0], this.view[4], this.view[8] );
    this.camera_up    = glMatrix.vec3.fromValues( this.view[1], this.view[5], this.view[9] );
    this.camera_front = glMatrix.vec3.fromValues( -this.view[2], -this.view[6], -this.view[10] );
    this.need_update  = true;
    this.fitNearFarPlane();
  }

  fitNearFarPlane() {
    if(this.sceneBBOX === null)
      return;
    const scene_size = [this.sceneBBOX.maxV[0] - this.sceneBBOX.minV[0], this.sceneBBOX.maxV[1] - this.sceneBBOX.minV[1], this.sceneBBOX.maxV[2] - this.sceneBBOX.minV[2]];
    const scene_centerA = [
      0.5 * (this.sceneBBOX.maxV[0] + this.sceneBBOX.minV[0]), 
      0.5 * (this.sceneBBOX.maxV[1] + this.sceneBBOX.minV[1]), 
      0.5 * (this.sceneBBOX.maxV[2] + this.sceneBBOX.minV[2])
    ];
    const scene_center = glMatrix.vec3.fromValues(scene_centerA[0], scene_centerA[1], scene_centerA[2]);
    const scene_radius = 0.5 * Math.sqrt(scene_size[0] * scene_size[0] + scene_size[1] * scene_size[1] + scene_size[2] * scene_size[2]);
    const scene_max_side = Math.max(...scene_size);
    const units_scale = 0.1;

    let new_near = 0.13;
    let new_far = 133;

    const {minV, maxV} = this.sceneBBOX;

    let corner = glMatrix.vec3.create();
    let center_ecs = glMatrix.vec3.transformMat4(glMatrix.vec3.create(), this.camera_pos, this.view);

		let cur_far = -Number.MAX_VALUE;
		let cur_near = Number.MAX_VALUE;
    let changed = false;
		for (let i = 0; i < 8; ++i)
		{
      corner = glMatrix.vec3.fromValues(i & 1 ? maxV[0] : minV[0], i & 2 ? maxV[1] : minV[1], i & 4 ? maxV[2] : minV[2]);
      corner = glMatrix.vec3.transformMat4(glMatrix.vec3.create(), corner, this.view);
      			
			if (corner[2] >= 0.0)
			{
				cur_near = units_scale;
				continue;
			}

			let f = -corner[2];
			if (f < cur_near)
			{
				cur_near = f;
				changed = true;
			}
			if (f > cur_far)
			{
				cur_far = f;
				changed = true;
      }
    }
    //console.log("Near/Far", cur_near, cur_far, changed);

		let center_box_ecs = glMatrix.vec3.transformMat4(glMatrix.vec3.create(), scene_center, this.view);
		let dist_to_bbox_center = glMatrix.vec3.distance(center_ecs, center_box_ecs);
		let inside_bounding_sphere = dist_to_bbox_center <= scene_radius;

		if (inside_bounding_sphere)
		{
      //console.log("Inside Bounding Sphere");
			new_near = units_scale;
			new_far = new_near + cur_far;
		}
		else
		{
      //console.log("Outside Bounding Sphere");
			if (changed)
			{
				new_near = Math.max(units_scale, cur_near);
				new_far = new_near + cur_far;
			}
			else
			{
				// if we are outside the bounding sphere looking at nothing, just set a default near far
				new_near = units_scale;
				new_far = new_near + 1.0;
			}
    }
    
    new_near -= 0.1 + 0.1 * scene_max_side;//getSceneBox().getMaxSide();
	  new_near = Math.max(new_near, 0.1 /*unit scales*/);

    new_far += 0.1;
    //console.log("Near/Far :", new_near, new_far);
    this.setNearFar(new_near, new_far);
  }

  mouseMove(mouse)
  {
    if (this.arcball_on) {
      mouse.target.style.cursor = 'move';
      this.mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
    }
  }

  mouseUp(mouse) 
  {
    switch( mouse.button ) {
      case 0:
        if (this.arcball_on)
          mouse.target.style.cursor = 'pointer';
        this.arcball_on = false;
        break;
      case 1:
        break;
      case 2:
        break;  
      default:
        break;
    }
  }

  mouseDown(mouse)
  {
    switch( mouse.button ) {
      case 0:
        this.arcball_on = true;
        this.mouse = this.last_mouse = glMatrix.vec2.fromValues(mouse.offsetX, mouse.offsetY);
        break;
      case 1:
        break;
      case 2:
        break;
      default:
        break;
    }
  }

  touchStart(touch)
  {
    this.arcball_on = true;
    this.mouse = this.last_mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
  }

  touchEnd(touch) {
    this.arcball_on = false;
  }
  
  touchMove(touch) {
    if (this.arcball_on) {
      this.mouse = glMatrix.vec2.fromValues(touch.touches[0].clientX, touch.touches[0].clientY);
    }
  }

  wheelScroll(event) {
    //console.log(event.deltaX, event.deltaY, event.deltaMode);
    /*if(event.deltaMode === 0)
      this.zoomLevel *= Math.abs(event.deltaY) / this.height * 10.0 + (event.deltaY > 0.0? 0 : 1);
    else
      this.zoomLevel *= event.deltaY / this.height * 10.0;*/
    
    // ZoomIn/Out always 1.2 or 0.83 of the current zoom level
    let zoom_multiplier = (event.deltaY === 0)? 1.0 : event.deltaY > 0? 1.0 / 1.2 : 1.2;
    this.zoomLevel *= zoom_multiplier;
    this.zoomLevel = Math.min(this.zoomBounds[1], Math.max(this.zoomLevel, this.zoomBounds[0]));

    this.camera_pos[0] = -this.zoomLevel;
    this.camera_pos[1] = -this.zoomLevel;
    this.camera_pos[2] = -this.zoomLevel;
    this.camera_pos = glMatrix.vec3.multiply(glMatrix.vec3.create(), this.camera_pos, this.camera_front);
    this.camera_pos = glMatrix.vec3.add(glMatrix.vec3.create(), this.camera_pos, this.camera_target);
  
    this.recomputeViewMatrix();
    event.stopPropagation();
    event.preventDefault();
  }

  update() {
    const ret_value   = this.need_update;
    this.need_update  = false;
    if (this.mouse[0] == this.last_mouse[0] && 
        this.mouse[1] == this.last_mouse[1])
      return ret_value;
      
    var delta = glMatrix.vec2.fromValues(1, -1);
    const mouse_delta = glMatrix.vec2.sub(glMatrix.vec2.create(), this.last_mouse, this.mouse);
    delta = glMatrix.vec2.mul(glMatrix.vec2.create(), delta, mouse_delta);
    delta[0] *= 0.004; // NEED to multiply with dt
    delta[1] *= 0.002;// NEED to multiply with dt
  
    this.last_mouse[0] = this.mouse[0];
    this.last_mouse[1] = this.mouse[1];
  
    //console.log("Start");

    this.angles[0] += delta[0]; // X axis 
    this.angles[1] += delta[1]; // Y axis 
    
    const halfPI = Math.PI * 0.495;
    this.angles[1] = Math.max(Math.min(this.angles[1], halfPI), -halfPI);          
    
    let camera_pos = glMatrix.vec3.create();    
    // Spherical Coordinates
    const theta = this.angles[1];
    const phi = -this.angles[0];
    camera_pos[0] = Math.cos(theta) * Math.cos(phi);
	  camera_pos[1] = Math.sin(theta);
	  camera_pos[2] = Math.cos(theta) * Math.sin(phi);
    camera_pos = glMatrix.vec3.normalize(camera_pos, camera_pos);
    // The Right vector is on the right of the view vector
	  const d = phi + 0.5 * Math.PI;
	  let right = glMatrix.vec3.fromValues(Math.cos(d), 0.0, Math.sin(d));
    right = glMatrix.vec3.normalize(right, right);
    // Compute the up vector
	  this.camera_up = glMatrix.vec3.cross(glMatrix.vec3.create(), right, camera_pos);
    this.camera_up = glMatrix.vec3.normalize(this.camera_up, this.camera_up);
    
    camera_pos[0] *= this.zoomLevel;
    camera_pos[1] *= this.zoomLevel;
    camera_pos[2] *= this.zoomLevel;
    
    this.camera_pos = glMatrix.vec3.add(glMatrix.vec3.create(), camera_pos, this.camera_target);
    this.view = glMatrix.mat4.lookAt( 
      glMatrix.mat4.create(),
      this.camera_pos, // pos
      this.camera_target, // at
      this.camera_up // up
    ); // view is [right, up, -forward, -pos]^T;
    
    this.camera_right = glMatrix.vec3.fromValues( this.view[0], this.view[4], this.view[8] );
    this.camera_up = glMatrix.vec3.fromValues( this.view[1], this.view[5], this.view[9] );
    this.camera_front = glMatrix.vec3.fromValues( -this.view[2], -this.view[6], -this.view[10] );

    //WebRaysViewer.last_mouse = WebRaysViewer.mouse;
    return true;
  }

  attachControls(canvas) {
    const call_during_capture = true;
    canvas.addEventListener("mousemove", (ev) => this.mouseMove(ev), call_during_capture);
    canvas.addEventListener("mouseup", (ev) => {this.mouseUp(ev);}, call_during_capture);
    canvas.addEventListener("mousedown", (ev) => {this.mouseDown(ev);}, call_during_capture);
    canvas.addEventListener("touchstart", ev => {this.touchStart(ev);}, {passive: true});
    canvas.addEventListener("touchend", (ev) => {this.touchEnd(ev);}, {passive: true});
    //canvas.addEventListener("touchcancel", handleCancel, false);
    canvas.addEventListener("touchmove", (ev) => {this.touchMove(ev);}, {passive: true});
    canvas.addEventListener('wheel', (ev) => {this.wheelScroll(ev);}, call_during_capture); // {passive: true}
  }
}