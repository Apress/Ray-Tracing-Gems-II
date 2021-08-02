/* Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/ //--------------------------------------------------------------------
#pragma once

#include <nvmath/nvmath.h>
#include <nvh/nvprint.hpp>

#include <cmath>

using namespace nvmath;

//-----------------------------------------------------------------------------
// Camera
//-----------------------------------------------------------------------------
struct InertiaCamera
{
    vec3f    curEyePos, curFocusPos, curObjectPos;
    vec3f    eyePos, focusPos, objectPos;
    float   tau;
    float   epsilon;
    float   eyeD;
    float   focusD;
    float   objectD;
    mat4f m4_view;
    //------------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------------
    InertiaCamera(const vec3f eye=vec3f(0.0f,1.0f,-3.0f), const vec3f focus=vec3f(0,0,0), const vec3f object=vec3f(0,0,0))
    {
        epsilon = 0.001f;
        tau = 0.2f;
        curEyePos = eye;
        eyePos = eye;
        curFocusPos = focus;
        focusPos = focus;
        curObjectPos = object;
        objectPos = object;
        eyeD = 0.0f;
        focusD = 0.0f;
        objectD = 0.0f;
        m4_view.identity();
        mat4f Lookat = nvmath::look_at(curEyePos, curFocusPos, vec3f(0,1,0));
        m4_view *= Lookat;
    }
    //------------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------------
    void rotateH(float s, bool bPan=false)
    {
        vec3f p = eyePos;
        vec3f o = focusPos;
        vec3f po = p-o;
        float l = po.norm();
        vec3f dv = cross(po, vec3f(0,1,0) );
        dv *= s;
        p += dv;
        po = p-o;
        float l2 = po.norm();
        l = l2 - l;
        p -= (l/l2) * (po);
        eyePos = p;
        if(bPan)
            focusPos += dv;
    }
    //------------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------------
    void rotateV(float s, bool bPan=false)
    {
        vec3f p = eyePos;
        vec3f o = focusPos;
        vec3f po = p-o;
        float l = po.norm();
        vec3f dv = cross(po, vec3f(0,-1,0) );
        dv.normalize();
        vec3f dv2 = cross(po, dv );
        dv2 *= s;
        p += dv2;
        po = p-o;
        float l2 = po.norm();

        if(bPan)
            focusPos += dv2;

        // protect against gimbal lock
        if (std::fabs(dot(po/l2, vec3f(0,1,0))) > 0.99) return;

        l = l2 - l;
        p -= (l/l2) * (po);
        eyePos = p;
    }
    //------------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------------
    void move(float s, bool bPan)
    {
        vec3f p = eyePos;
        vec3f o = focusPos;
        vec3f po = p-o;
        po *= s;
        p -= po;
        if(bPan)
            focusPos -= po;
        eyePos = p;
    }
    //------------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------------
    bool update(float dt)
    {
        if(dt > (1.0f/60.0f))
            dt = (1.0f/60.0f);
        bool bContinue = false;
        static vec3f eyeVel = vec3f(0,0,0);
        static vec3f eyeAcc = vec3f(0,0,0);
        eyeD = nv_norm(curEyePos - eyePos);
        if(eyeD > epsilon)
        {
            bContinue = true;
            vec3f dV = curEyePos - eyePos;
            eyeAcc = (-2.0f/tau)*eyeVel - dV/(tau*tau);
            // integrate
            eyeVel += eyeAcc * vec3f(dt,dt,dt);
            curEyePos += eyeVel * vec3f(dt,dt,dt);
        } else {
            eyeVel = vec3f(0,0,0);
            eyeAcc = vec3f(0,0,0);
        }

        static vec3f focusVel = vec3f(0,0,0);
        static vec3f focusAcc = vec3f(0,0,0);
        focusD = nv_norm(curFocusPos - focusPos);
        if(focusD > epsilon)
        {
            bContinue = true;
            vec3f dV = curFocusPos - focusPos;
            focusAcc = (-2.0f/tau)*focusVel - dV/(tau*tau);
            // integrate
            focusVel += focusAcc * vec3f(dt,dt,dt);
            curFocusPos += focusVel * vec3f(dt,dt,dt);
        } else {
            focusVel = vec3f(0,0,0);
            focusAcc = vec3f(0,0,0);
        }

        static vec3f objectVel = vec3f(0,0,0);
        static vec3f objectAcc = vec3f(0,0,0);
        objectD = nv_norm(curObjectPos - objectPos);
        if(objectD > epsilon)
        {
            bContinue = true;
            vec3f dV = curObjectPos - objectPos;
            objectAcc = (-2.0f/tau)*objectVel - dV/(tau*tau);
            // integrate
            objectVel += objectAcc * vec3f(dt,dt,dt);
            curObjectPos += objectVel * vec3f(dt,dt,dt);
        } else {
            objectVel = vec3f(0,0,0);
            objectAcc = vec3f(0,0,0);
        }
        //
        // Camera View matrix
        //
        vec3f up(0,1,0);
        m4_view.identity();
        mat4f Lookat = nvmath::look_at(curEyePos, curFocusPos, up);
        m4_view *= Lookat;
        return bContinue;
    }
    //------------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------------
    void look_at(const vec3f& eye, const vec3f& center/*, const vec3f& up*/, bool reset=false)
    {
        eyePos = eye;
        focusPos = center;
        if(reset)
        {
            curEyePos = eye;
            curFocusPos = center;
            vec3f up(0,1,0);
            m4_view.identity();
            mat4f Lookat = nvmath::look_at(curEyePos, curFocusPos, up);
            m4_view *= Lookat;
        }
    }
    //------------------------------------------------------------------------------
    // 
    //------------------------------------------------------------------------------
    void print_look_at(bool cppLike=false)
    {
        if(cppLike)
        {
            LOGI("{vec3f(%.2f, %.2f, %.2f), vec3f(%.2f, %.2f, %.2f)},\n",
            eyePos.x, eyePos.y, eyePos.z, focusPos.x, focusPos.y, focusPos.z);
        } else {
            LOGI("%.2f %.2f %.2f %.2f %.2f %.2f 0.0\n",
            eyePos.x, eyePos.y, eyePos.z, focusPos.x, focusPos.y, focusPos.z);
        }
    }
};
