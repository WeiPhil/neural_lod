/* 
 * Copyright 2022 Intel Corporation.
 */

#include "vkr.h"

#include <stdio.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

PyObject *convertVkrMipLevelRecord(const VkrMipLevel *ml)
{
  if (!ml) {
    return NULL;
  }

  PyObject *o = Py_BuildValue(
    "{s:i,s:i,s:K,s:L}",
    "width", ml->width,
    "height", ml->height,
    "dataSize", ml->dataSize,
    "dataOffset", ml->dataOffset
  );

  return o;
}

PyObject *convertVkrTextureRecord(const VkrTexture *tex)
{
  PyObject *mipList = PyList_New(tex->numMipLevels);
  if (!mipList) {
    return NULL;
  }

  for (uint64_t i = 0; i < tex->numMipLevels; ++i) {
    PyObject *mip = convertVkrMipLevelRecord(tex->mipLevels + i);
    if (!mip || PyList_SetItem(mipList, i, mip) != 0) {
      Py_XDECREF(mip);
      Py_DECREF(mipList);
      return NULL;
    }
  }

  PyObject *o = Py_BuildValue(
    "{s:s,s:i,s:i,s:i,s:i,s:O,s:K,s:L}",
    "filename", tex->filename,
    "version", tex->version,
    "width", tex->width,
    "height", tex->height,
    "format", tex->format,
    "mipLevels", mipList,
    "dataSize", tex->dataSize,
    "dataOffset", tex->dataOffset
  );

  if (!o) {
    Py_DECREF(mipList);
    return NULL;
  }

  return o;
}

PyObject *convertVkrMaterialRecord(const VkrMaterial *mat)
{
  if (!mat) {
    return NULL;
  }

  uint32_t numFeatures = 0;
  while (numFeatures < VkrMaterialMaxFeatureTextures && mat->features[numFeatures].dataSize != 0)
    ++numFeatures;
  PyObject *featureList = PyList_New(numFeatures);
  if (!featureList) {
    return NULL;
  }
  for (uint64_t i = 0; i < numFeatures; ++i) {
    PyObject *feature = convertVkrTextureRecord(mat->features + i);
    if (!feature || PyList_SetItem(featureList, i, feature) != 0) {
      Py_XDECREF(feature);
      Py_DECREF(featureList);
      return NULL;
    }
  }

  PyObject *texBaseColor = convertVkrTextureRecord(&mat->texBaseColor);
  PyObject *texNormal = convertVkrTextureRecord(&mat->texNormal);
  PyObject *texAoMetalnessRoughness = convertVkrTextureRecord(&mat->texAoMetalnessRoughness);

  if (!texBaseColor || !texNormal || !texAoMetalnessRoughness) {
    Py_XDECREF(texBaseColor);
    Py_XDECREF(texNormal);
    Py_XDECREF(texAoMetalnessRoughness);
    Py_DECREF(featureList);
    return NULL;
  }

  PyObject *s = Py_BuildValue(
    "{s:s,s:(fff),s:O,s:(fff),s:O,s:f,s:f,s:f,s:O,s:f,s:f,s:f,s:f,s:f,s:L}",
    "name", mat->name,
    "baseColor", mat->baseColor[0], mat->baseColor[1], mat->baseColor[2],
    "texColor", texBaseColor,
    "normal", mat->normal[0], mat->normal[1], mat->normal[2],
    "texNormal", texNormal,
    "ao", mat->ao,
    "metalness", mat->metalness,
    "roughness", mat->roughness,
    "texAoMetalnessRoughness", texAoMetalnessRoughness,
    "emissionIntensity", mat->emissionIntensity,
    "specularTransmission", mat->specularTransmission,
    "iorEta", mat->iorEta,
    "iorK", mat->iorK,
    "translucency", mat->translucency,
    "features", featureList
  );
  
  if (!s) {
    Py_XDECREF(texBaseColor);
    Py_XDECREF(texNormal);
    Py_XDECREF(texAoMetalnessRoughness);
    Py_DECREF(featureList);
    return NULL;
  }

  return s;
}

PyObject *convertVkrMeshRecord(const VkrMesh *m)
{
  if (!m) {
    return NULL;
  }

  const npy_intp v_dim = 3;
  PyObject *vertexScale = PyArray_SimpleNew(1,  &v_dim, NPY_FLOAT);
  PyObject *vertexOffset = PyArray_SimpleNew(1, &v_dim, NPY_FLOAT);
  PyObject *scaleBoundsMin = PyArray_SimpleNew(1, &v_dim, NPY_FLOAT);
  PyObject *scaleBoundsMax = PyArray_SimpleNew(1, &v_dim, NPY_FLOAT);

  if (!vertexScale || !vertexOffset || !scaleBoundsMin || !scaleBoundsMax)
  {
    Py_XDECREF(vertexScale);
    Py_XDECREF(vertexOffset);
    Py_XDECREF(scaleBoundsMin);
    Py_XDECREF(scaleBoundsMax);
    return NULL;
  }

  float *d = (float *) PyArray_DATA((PyArrayObject *)vertexScale);
  memcpy(d, m->vertexScale, v_dim * sizeof(float));
  d = (float *) PyArray_DATA((PyArrayObject *)vertexOffset);
  memcpy(d, m->vertexOffset, v_dim * sizeof(float));
  d = (float *) PyArray_DATA((PyArrayObject *)scaleBoundsMin);
  memcpy(d, m->scaleBoundsMin, v_dim * sizeof(float));
  d = (float *) PyArray_DATA((PyArrayObject *)scaleBoundsMax);
  memcpy(d, m->scaleBoundsMax, v_dim * sizeof(float));

  PyObject *s = Py_BuildValue(
    "{s:s,s:O,s:O,s:O,s:O,s:i,s:K,s:K,s:L,s:L,s:L,s:i,s:L}",
    "name", m->name,
    "vertexScale", vertexScale,
    "vertexOffset", vertexOffset,
    "scaleBoundsMin", scaleBoundsMin,
    "scaleBoundsMax", scaleBoundsMax,
    "materialIdBufferBase", m->materialIdBufferBase,
    "numMaterialsInRange", m->numMaterialsInRange,
    "numTriangles", m->numTriangles,
    "vertexBufferOffset", m->vertexBufferOffset,
    "normalUvBufferOffset", m->normalUvBufferOffset,
    "materialIdBufferOffset", m->materialIdBufferOffset,
    "materialIdSize", m->materialIdSize,
    "indexBufferOffset", m->indexBufferOffset
  );

  if (!s) {
    Py_XDECREF(vertexScale);
    Py_XDECREF(vertexOffset);
    Py_XDECREF(scaleBoundsMin);
    Py_XDECREF(scaleBoundsMax);
    return NULL;
  }

  return s;
}

PyObject *convertVkrInstanceRecord(const VkrInstance *instance)
{
  if (!instance) {
    return NULL;
  }

  const npy_intp o_dim[] = { 3, 4 };
  PyObject *transform = PyArray_ZEROS(2, o_dim, NPY_FLOAT, 0);
  if (!transform) {
    printf("Cannot create transform with dimensions %d, %d\n", (int) o_dim[0], (int) o_dim[1]);
    Py_XDECREF(transform);
    return NULL;
  }

  float *vt = (float *) PyArray_DATA((PyArrayObject *)transform);
  for (int j = 0; j < 3; ++j)
  for (int i = 0; i < 4; ++i)
  {
    vt[j*4+i] = instance->transform[i][j];
  }

  PyObject *s = Py_BuildValue(
    "{s:s,s:i,s:O}",
    "name", instance->name,
    "meshId", instance->meshId,
    "transform", transform
  );

  if (!s) {
    Py_DECREF(transform);
    return NULL;
  }

  return s;
}

PyObject *convertVkrSceneRecord(const VkrScene *v)
{
  if (!v) {
    return NULL;
  }

  PyObject *materialList = PyList_New(v->numMaterials);
  if (!materialList) {
    return NULL;
  }

  for (uint64_t i = 0; i < v->numMaterials; ++i) {
    PyObject *m = convertVkrMaterialRecord(v->materials + i);
    if (!m || PyList_SetItem(materialList, i, m) != 0) {
      Py_XDECREF(m);
      Py_DECREF(materialList);
      return NULL;
    }
  }

  PyObject *meshList = PyList_New(v->numMeshes);
  if (!meshList)
  {
    Py_DECREF(materialList);
    return NULL;
  }

  for (uint64_t i = 0; i < v->numMeshes; ++i) {
    PyObject *m = convertVkrMeshRecord(v->meshes + i);
    if (!m || PyList_SetItem(meshList, i, m) != 0) {
      Py_XDECREF(m);
      Py_DECREF(meshList);
      Py_DECREF(materialList);
      return NULL;
    }
  }

  PyObject *instanceList = PyList_New(v->numInstances);
  if (!instanceList)
  {
    Py_DECREF(meshList);
    Py_DECREF(materialList);
    return NULL;
  }

  for (uint64_t i = 0; i < v->numInstances; ++i) {
    PyObject *inst = convertVkrInstanceRecord(v->instances + i);
    if (!inst || PyList_SetItem(instanceList, i, inst) != 0) {
      Py_XDECREF(inst);
      Py_DECREF(instanceList);
      Py_DECREF(meshList);
      Py_DECREF(materialList);
      return NULL;
    }
  }

  PyObject *s = Py_BuildValue(
    "{s:i,s:s,s:O,s:K,s:O,s:O}",
    "version", v->version,
    "textureDir", v->textureDir,
    "materials", materialList,
    "numTriangles", v->numTriangles,
    "meshes", meshList,
    "instances", instanceList
  );

  if (!s) {
    Py_DECREF(instanceList);
    Py_DECREF(meshList);
    Py_DECREF(materialList);
  }
  return s;
}

/* FUNCTION WRAPPERS */

void errorHandler(VkrResult result, const char *msg)
{
  PyErr_SetString(PyExc_RuntimeError, msg);
}

static PyObject *pyvkr_open_scene(PyObject *self, PyObject *args)
{
  const char *filename;
  if (!PyArg_ParseTuple(args, "s", &filename))
  {
    return NULL;
  }

  VkrScene scene;
  if (vkr_open_scene(filename, &scene, errorHandler) != VKR_SUCCESS)
  {
    return NULL;
  }

  PyObject *pyScene = convertVkrSceneRecord(&scene);
  vkr_close_scene(&scene);
  return pyScene;
}

static PyObject *pyvkr_open_texture(PyObject *self, PyObject *args)
{
  const char *filename;
  if (!PyArg_ParseTuple(args, "s", &filename))
  {
    return NULL;
  }

  VkrTexture texture;
  if (vkr_open_texture(filename, &texture, errorHandler) != VKR_SUCCESS)
  {
    return NULL;
  }

  PyObject *pyScene = convertVkrTextureRecord(&texture);
  vkr_close_texture(&texture);
  return pyScene;
}

static PyObject *pyvkr_dequantize_vertices(PyObject *self, PyObject *args)
{
  PyObject *quantizedArg;
  float vertexScale[] = {1.f, 1.f, 1.f};
  float vertexOffset[] = {0.f, 0.f, 0.f};

  if (!PyArg_ParseTuple(args, "O!(fff)(fff)", &PyArray_Type, &quantizedArg,
        vertexScale+0, vertexScale+1, vertexScale+2,
        vertexOffset+0, vertexOffset+1, vertexOffset+2))
  {
    return NULL;
  }

  PyObject *quantized = PyArray_FROM_OTF(quantizedArg, NPY_UINT64,
      NPY_ARRAY_IN_ARRAY);

  if (!quantized)
  {
    Py_XDECREF(quantized);
    return NULL;
  }

  const int i_ndim = PyArray_NDIM((PyArrayObject *)quantized);
  const npy_intp *i_dims = PyArray_DIMS((PyArrayObject *)quantized);

  const int o_ndim = i_ndim+1;
  npy_intp *o_dims = (npy_intp*) malloc(o_ndim * sizeof(npy_intp));
  if (!o_dims) {
    Py_XDECREF(quantized);
    return NULL;
  }

  for (int i = 0; i+1 < o_ndim; ++i)
  {
    o_dims[i] = i_dims[i];
  }
  o_dims[o_ndim-1] = 3;

  PyObject *dequantized = PyArray_SimpleNew(o_ndim, o_dims, NPY_FLOAT);
  free(o_dims);

  if (!dequantized) {
    Py_XDECREF(quantized);
    return NULL;
  }

  const uint64_t *vq = (uint64_t *) PyArray_DATA((PyArrayObject *)quantized);
  float *v = (float *) PyArray_DATA((PyArrayObject *)dequantized);

  const uint64_t numVertices =
    (uint64_t) PyArray_SIZE((PyArrayObject *)quantized);
  vkr_dequantize_vertices(vq, numVertices, vertexScale, vertexOffset, v);

  Py_XDECREF(quantized);
  return dequantized;
}

static PyObject *pyvkr_dequantize_normal_uv(PyObject *self, PyObject *args)
{
  PyObject *quantizedArg;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &quantizedArg))
  {
    return NULL;
  }

  PyObject *quantized = PyArray_FROM_OTF(quantizedArg, NPY_UINT64,
      NPY_ARRAY_IN_ARRAY);

  if (!quantized)
  {
    Py_XDECREF(quantized);
    return NULL;
  }

  const int i_ndim = PyArray_NDIM((PyArrayObject *)quantized);
  const npy_intp *i_dims = PyArray_DIMS((PyArrayObject *)quantized);

  const int o_ndim = i_ndim+1;
  npy_intp *o_dims = (npy_intp*) malloc(o_ndim * sizeof(npy_intp));
  if (!o_dims) {
    Py_XDECREF(quantized);
    return NULL;
  }

  for (int i = 0; i+1 < o_ndim; ++i)
  {
    o_dims[i] = i_dims[i];
  }

  o_dims[o_ndim-1] = 3;
  PyObject *normals = PyArray_SimpleNew(o_ndim, o_dims, NPY_FLOAT);
  o_dims[o_ndim-1] = 2;
  PyObject *uv = PyArray_SimpleNew(o_ndim, o_dims, NPY_FLOAT);
  PyObject *tuple = Py_BuildValue("(OO)", normals, uv);

  Py_XDECREF(uv);
  Py_XDECREF(normals);
  free(o_dims);

  if (!normals || !uv || !tuple) {
    Py_XDECREF(tuple);
    Py_XDECREF(quantized);
    return NULL;
  }

  const uint64_t *vq = (uint64_t *) PyArray_DATA((PyArrayObject *)quantized);
  float *fn = (float *) PyArray_DATA((PyArrayObject *)normals);
  float *fuv = (float *) PyArray_DATA((PyArrayObject *)uv);

  const uint64_t numNormals =
    (uint64_t) PyArray_SIZE((PyArrayObject *)quantized);
  vkr_dequantize_normal_uv(vq, numNormals, fn, fuv);

  Py_XDECREF(quantized);
  return tuple;
}

static PyObject *pyvkr_convert_texture(PyObject *self, PyObject *args)
{
  const char *inputFile = NULL;
  const char *outputFile = NULL;
  int format = 0, opaqueFormat = -1;
  if (!PyArg_ParseTuple(args, "ssi|i", &inputFile, &outputFile, &format, &opaqueFormat))
  {
    return NULL;
  }
  if (opaqueFormat == -1)
  {
    opaqueFormat = format;
  }

  if (vkr_convert_texture(inputFile, outputFile, format, opaqueFormat, errorHandler)
      != VKR_SUCCESS)
  {
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}


/* MODULE DEFINITION */

static PyMethodDef VkrMethods[] = {
  {
    "open_scene",
    (PyCFunction)(void(*)(void))pyvkr_open_scene,
    METH_VARARGS,
    "Open a .vkr scene"
  },
  {
    "open_texture",
    (PyCFunction)(void(*)(void))pyvkr_open_texture,
    METH_VARARGS,
    "Open a .vkt texture file."
  },
  {
    "dequantize_vertices",
    (PyCFunction)(void(*)(void))pyvkr_dequantize_vertices,
    METH_VARARGS,
    "Dequantize compressed vertex data from a .vks file."
  },
  {
    "dequantize_normal_uv",
    (PyCFunction)(void(*)(void))pyvkr_dequantize_normal_uv,
    METH_VARARGS,
    "Dequantize compressed normal/uv data from a .vks file."
  },
  {
    "convert_texture",
    (PyCFunction)(void(*)(void))pyvkr_convert_texture,
    METH_VARARGS,
    "Convert a texture to .vkt format."
  },
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vkrmodule = {
  PyModuleDef_HEAD_INIT,
  .m_name = "pyvkr",
  .m_doc  = "Python wrapper for libvkr.",
  .m_size = -1,
  VkrMethods
};

PyMODINIT_FUNC PyInit_pyvkr(void)
{
  PyObject *module = PyModule_Create(&vkrmodule);
  if (!module) {
    return NULL;
  }

  import_array(); // NUMPY.

  return module;
}


