/* Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
 */

#pragma once

#include "Input.h"

using namespace DirectX;

namespace Input
{

//----------------------------------------------------------------------------------------------------------
// Private Functions
//----------------------------------------------------------------------------------------------------------

void Rotate(InputInfo &input, Camera &camera)
{
    XMFLOAT3 up = XMFLOAT3(0.f, 1.f, 0.f);
    XMMATRIX rotationPitch = XMMatrixRotationAxis(XMLoadFloat3(&camera.right), input.pitch * (XM_PI / 180.f));

    XMStoreFloat3(&camera.forward, XMVector3Normalize(XMVector3Transform(XMLoadFloat3(&camera.forward), rotationPitch)));
    XMStoreFloat3(&camera.up, XMVector3Normalize(XMVector3Transform(XMLoadFloat3(&camera.up), rotationPitch)));

    XMMATRIX rotationYaw = XMMatrixRotationAxis(XMLoadFloat3(&up), input.yaw * (XM_PI / 180.f));

    XMStoreFloat3(&camera.forward, XMVector3Normalize(XMVector3Transform(XMLoadFloat3(&camera.forward), rotationYaw)));
    XMStoreFloat3(&camera.right, XMVector3Normalize(-XMVector3Cross(XMLoadFloat3(&camera.forward), XMLoadFloat3(&up))));
    XMStoreFloat3(&camera.up, XMVector3Normalize(XMVector3Cross(XMLoadFloat3(&camera.forward), XMLoadFloat3(&camera.right))));
}

//----------------------------------------------------------------------------------------------------------
// Public Functions
//----------------------------------------------------------------------------------------------------------

static const float movementSpeed = 0.25f;
static const float rotationSpeed = 0.5f;

/**
* Handle keyboard inputs.
*/
bool KeyHandler(InputInfo &input, Camera& camera, float cameraSpeedAdjustment, float elapsedTime)
{
    Keyboard::State kb = input.keyboard.GetState();

    if (input.kbTracker.IsKeyReleased(Keyboard::Escape))
    {
        PostQuitMessage(0);
        return false;
    }

    float movement = movementSpeed * elapsedTime * cameraSpeedAdjustment;
    float speed = movement / 100.f;
    bool result = false;

    if (kb.IsKeyDown(Keyboard::LeftShift) || kb.IsKeyDown(Keyboard::RightShift))
    {
        speed *= 2.f;
    }

    if (kb.IsKeyDown(Keyboard::LeftControl) || kb.IsKeyDown(Keyboard::RightControl))
    {
        speed *= 0.1f;
    }

    if (kb.IsKeyDown(Keyboard::LeftAlt) || kb.IsKeyDown(Keyboard::RightAlt))
    {
        speed *= 0.01f;
    }

    if (kb.IsKeyDown(Keyboard::A))
    {
        camera.position = { camera.position.x - (camera.right.x * speed), camera.position.y - (camera.right.y * speed), camera.position.z - (camera.right.z * speed) };
        result = true;
    }

    if (kb.IsKeyDown(Keyboard::D))
    {
        camera.position = { camera.position.x + (camera.right.x * speed), camera.position.y + (camera.right.y * speed), camera.position.z + (camera.right.z * speed) };
        result = true;
    }

    if (kb.IsKeyDown(Keyboard::S))
    {
        camera.position = { camera.position.x - (camera.forward.x * speed), camera.position.y - (camera.forward.y * speed), camera.position.z - (camera.forward.z * speed) };
        result = true;
    }

    if (kb.IsKeyDown(Keyboard::W))
    {
        camera.position = { camera.position.x + (camera.forward.x * speed), camera.position.y + (camera.forward.y * speed), camera.position.z + (camera.forward.z * speed) };
        result = true;
    }

    if (kb.IsKeyDown(Keyboard::E))
    {
        camera.position.y += speed;
        result = true;
    }

    if (kb.IsKeyDown(Keyboard::Q))
    {
        camera.position.y -= speed;
        result = true;
    }

    if (input.kbTracker.IsKeyReleased(Keyboard::F1))
    {
        input.captureScreenshot = true;
        result = true;
    }

    if (input.kbTracker.IsKeyReleased(Keyboard::F2))
    {
        input.toggleGui = true;
        result = true;
    }

    if (input.kbTracker.IsKeyReleased(Keyboard::F5))
    {
        input.reloadShaders = true;
        result = true;
    }

    input.kbTracker.Update(kb);
    return result;
}

/**
* Handle mouse inputs.
*/
bool MouseHandler(InputInfo &input, Camera &camera, float elapsedTime)
{
    float movement = movementSpeed * elapsedTime;
    float rotation = rotationSpeed * elapsedTime;
    Mouse::State mouse = input.mouse.GetState();
    if (mouse.leftButton)
    {
        // Just pressed the left mouse button
        if (input.lastMouseXY.x == INT_MAX && input.lastMouseXY.y == INT_MAX)
        {
            input.lastMouseXY = { mouse.x, mouse.y };
            return false;
        }

        // Compute relative change in mouse position, and multiply by the degrees of change per pixel
        float degreesPerPixelX = (camera.fov / (float)input.width) * camera.aspect;
        float degreesPerPixelY = (camera.fov / (float)input.height);

        input.yaw += (float)(mouse.x - input.lastMouseXY.x) * degreesPerPixelX * rotation;
        input.pitch += (float)(mouse.y - input.lastMouseXY.y) * degreesPerPixelY * rotation;

        // Store current mouse position
        input.lastMouseXY = { mouse.x, mouse.y };

        // Compute and apply the rotation
        Rotate(input, camera);

        bool result = input.yaw > 0.01f || input.pitch > 0.01f;

        input.yaw = 0.f;
        input.pitch = 0.f;

        return true;
    }

    if (mouse.rightButton)
    {
        // Just pressed the right mouse button
        if (input.lastMouseXY.x == INT_MAX && input.lastMouseXY.y == INT_MAX)
        {
            input.lastMouseXY = { mouse.x, mouse.y };
            return false;
        }

        float speed = movement / 100.f;
        float speedX = (float)(mouse.x - input.lastMouseXY.x) * speed;
        float speedY = (float)(mouse.y - input.lastMouseXY.y) * -speed;

        // Store current mouse position
        input.lastMouseXY = { mouse.x, mouse.y };

        // Compute new camera position
        camera.position = { camera.position.x - (camera.right.x * speedX), camera.position.y - (camera.right.y * speedX), camera.position.z - (camera.right.z * speedX) };
        camera.position = { camera.position.x - (camera.up.x * speedY), camera.position.y - (camera.up.y * speedY), camera.position.z - (camera.up.z * speedY) };

        return true;
    }

    if (mouse.scrollWheelValue != input.scrollWheelValue)
    {
        if (input.scrollWheelValue == INT_MAX)
        {
            input.scrollWheelValue = mouse.scrollWheelValue;
            return false;
        }

        float speed = (input.scrollWheelValue - mouse.scrollWheelValue) * movement / 100.f;
        camera.position = { camera.position.x - (camera.forward.x * speed), camera.position.y - (camera.forward.y * speed), camera.position.z - (camera.forward.z * speed) };

        input.scrollWheelValue = mouse.scrollWheelValue;
        return true;
    }

    if (input.initialized)
    {
        if (abs(input.yaw) >= 360) input.yaw = 0.f;
        if (abs(input.pitch) >= 360) input.pitch = 0.f;

        Rotate(input, camera);

        input.initialized = false;
    }

    input.lastMouseXY = { INT_MAX, INT_MAX };
    return false;
}

}
