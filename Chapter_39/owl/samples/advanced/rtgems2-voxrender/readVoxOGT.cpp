
// Note: this code is based on demo_vox in the open game tools project.
// The original comment for demo_vox follows.

/*
    demo_vox - MIT license - Justin Paver, Oct 2019

    A demonstration program to show you how to use the MagicaVoxel scene reader,
    writer, and merger from the open game tools project: https://github.com/jpaver/opengametools.

    Please see the MIT license information at the end of this file, and please consider 
    sharing any improvements you make.
*/

#include "readVox.h"

#define OGT_VOX_IMPLEMENTATION
#include "ogt_vox.h"

#if defined(_MSC_VER)
    #include <io.h>
#endif

#include <cassert>
#include <stdio.h>

// a helper function to load a magica voxel scene given a filename.
const ogt_vox_scene* load_vox_scene(const char* filename, uint32_t scene_read_flags = 0)
{
    // open the file
#if defined(_MSC_VER) && _MSC_VER >= 1400
    FILE * fp;
    if (0 != fopen_s(&fp, filename, "rb"))
        fp = 0;
#else
    FILE * fp = fopen(filename, "rb");
#endif
    if (!fp)
        return NULL;

    // get the buffer size which matches the size of the file
    fseek(fp, 0, SEEK_END);
    uint32_t buffer_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // load the file into a memory buffer
    uint8_t * buffer = new uint8_t[buffer_size];
    size_t itemsread = fread(buffer, buffer_size, 1, fp);
    assert(itemsread == 1);
    if (itemsread < 1) {
        delete[] buffer;
        return NULL;
    }

    fclose(fp);

    // construct the scene from the buffer
    const ogt_vox_scene * scene = ogt_vox_read_scene_with_flags(buffer, buffer_size, scene_read_flags);

    // the buffer can be safely deleted once the scene is instantiated.
    delete[] buffer;

    return scene;
}

const ogt_vox_scene* load_vox_scene_with_groups(const char* filename)
{
    return load_vox_scene(filename, k_read_scene_flags_groups);
}

uint32_t count_solid_voxels_in_model(const ogt_vox_model* model)
{
    uint32_t solid_voxel_count = 0;
    uint32_t voxel_index = 0;
    for (uint32_t z = 0; z < model->size_z; z++) {
        for (uint32_t y = 0; y < model->size_y; y++) {
            for (uint32_t x = 0; x < model->size_x; x++, voxel_index++) {
                // if color index == 0, this voxel is empty, otherwise it is solid.
                uint32_t color_index = model->voxel_data[voxel_index];
                bool is_voxel_solid = (color_index != 0);
                // add to our accumulator
                solid_voxel_count += (is_voxel_solid ? 1 : 0);
            }
        }
    }
    return solid_voxel_count;
}

void logScene(const ogt_vox_scene *scene)
{
  if (!scene) return;

  printf("#layers: %u\n", scene->num_layers);
  for (uint32_t layer_index = 0; layer_index < scene->num_layers; layer_index++)
  {
    const ogt_vox_layer* layer = &scene->layers[layer_index];
    printf("layer[%u,name=%s] is %s\n",
        layer_index,
        layer->name ? layer->name : "",
        layer->hidden ? "hidden" : "shown");
  }
  printf("#groups: %u\n", scene->num_groups);
  for (uint32_t group_index = 0; group_index < scene->num_groups; group_index++)
  {
    const ogt_vox_group* group = &scene->groups[group_index];
    const ogt_vox_layer* group_layer = group->layer_index != UINT32_MAX ? &scene->layers[group->layer_index] : NULL;
    printf("group[%u] has parent group %u, is part of layer[%u,name=%s] and is %s\n", 
        group_index, 
        group->parent_group_index,
        group->layer_index,
        group_layer && group_layer->name ? group_layer->name : "",
        group->hidden ? "hidden" : "shown");
  }

  // iterate over all instances - and print basic information about the instance and the model that it references
  printf("# instances: %u\n", scene->num_instances);
  for (uint32_t instance_index = 0; instance_index < scene->num_instances; instance_index++)
  {
    const ogt_vox_instance* instance = &scene->instances[instance_index];
    const ogt_vox_model* model = scene->models[instance->model_index];

    const char* layer_name =
      instance->layer_index == UINT32_MAX ? "(no layer)":
      scene->layers[instance->layer_index].name ? scene->layers[instance->layer_index].name : 
      "";

    printf("instance[%u,name=%s] at position (%.0f,%.0f,%.0f) uses model %u and is in layer[%u, name='%s'], group %u, and is %s\n",
        instance_index,
        instance->name ? instance->name : "",
        instance->transform.m30, instance->transform.m31, instance->transform.m32, // translation components of the instance
        instance->model_index,
        instance->layer_index,
        layer_name,
        instance->group_index,
        instance->hidden ? "hidden" : "shown");
  }
  // iterate over all models and print basic information about the model.
  printf("# models: %u\n", scene->num_models);
  for (uint32_t model_index = 0; model_index < scene->num_models; model_index++)
  {
    const ogt_vox_model* model = scene->models[model_index];

    uint32_t solid_voxel_count = count_solid_voxels_in_model(model);
    uint32_t total_voxel_count = model->size_x * model->size_y * model->size_z;

    printf(" model[%u] has dimension %ux%ux%u, with %u solid voxels of the total %u voxels (hash=%u)!\n",
        model_index,
        model->size_x, model->size_y, model->size_z,
        solid_voxel_count,
        total_voxel_count,
        model->voxel_hash);
  }

}

const ogt_vox_scene* loadVoxScene(const uint8_t *voxBuffer, size_t voxBufferLen)
{
  const ogt_vox_scene *scene = ogt_vox_read_scene_with_flags(voxBuffer, (uint32_t)voxBufferLen, k_read_scene_flags_groups);
  return scene;
}

const ogt_vox_scene* loadVoxScene(const char *filename)
{
  const ogt_vox_scene *scene = load_vox_scene_with_groups(filename);

  if (scene) {
    logScene(scene);
  }
  return scene;
}

const ogt_vox_scene* loadVoxScenes(const std::vector<std::string> &filenames)
{
  if (filenames.empty()) return nullptr;

  std::vector<const ogt_vox_scene*> scenes;
  scenes.reserve(filenames.size());

  for (const std::string fn : filenames) {
    const ogt_vox_scene *scene = load_vox_scene_with_groups(fn.c_str());
    if (scene) {
      scenes.push_back(scene);
    }
  }
  if (scenes.empty()) {
    return nullptr;
  }
  if (scenes.size() == 1) {
    return scenes[0];
  }

  // Merge using palette from first scene, which works ok for the MMMM scenes
  ogt_vox_scene *merged = ogt_vox_merge_scenes(scenes.data(), (uint32_t)scenes.size(), &scenes[0]->palette.color[1], 255);

  for (const ogt_vox_scene *scene : scenes) {
    ogt_vox_destroy_scene(scene);
  }
  if (merged) {
    logScene(merged);
  }
  return merged;

}

// Original license from demo_vox:

/* -------------------------------------------------------------------------------------------------------------------------------------------------

    MIT License

    Copyright (c) 2019 Justin Paver

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

------------------------------------------------------------------------------------------------------------------------------------------------- */
