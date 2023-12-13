import bpy
from mathutils import Matrix, Vector
from math import radians

import numpy as np
import copy
import struct

from collections import defaultdict

def apply_edge_split(bobject, smoothing_angle):
    esm = None
    for m in bobject.modifiers:
        if m.type == 'EDGE_SPLIT':
            esm = m
            break
    if esm is None:
        esm = bobject.modifiers.new('SharpEdgeSplit', 'EDGE_SPLIT')
    if esm is None:
        if bobject.type not in ['GPENCIL', 'ARMATURE', 'EMPTY', 'LIGHT', 'LIGHT_PROBE', 'CAMERA', 'SPEAKER']:
            print("Error applying edge splut modifier to ", bobject.name)
        return bobject
    esm.split_angle = smoothing_angle
    esm.use_edge_angle = True
    esm.use_edge_sharp = True
    return bobject

def try_quad_combine(context_scene, bobject):
    context_scene.objects.active = bobject
    #todo: ensure selected
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.object.mode_set(mode='OBJECT')

def matmul(m, v):
    return np.matmul(m, v[:,:,np.newaxis])[:,:,0]
def matmulh(m, v):
    v = np.hstack( (v, np.ones((v.shape[0], 1))) )
    v = matmul(m, v)
    return v[:,:3]

# https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
def inplace_part_1_by_2(x):
  x &= 0x000003ff                   # x = ---- ---- ---- ---- ---- --98 7654 3210
  #x = (x ^ (x << 16)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
  x ^= (x << 16)
  x &= 0xff0000ff
  #x = (x ^ (x <<  8)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x ^= (x << 8)
  x &= 0x0300f00f
  #x = (x ^ (x <<  4)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x ^= (x << 4)
  x &= 0x030c30c3
  #x = (x ^ (x <<  2)) & 0x09249249 # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x ^= (x << 2)
  x &= 0x09249249

def morton_codes_destructive(intvecs):
    inplace_part_1_by_2(intvecs)
    out_view = intvecs[:,0] << 1
    out_view += intvecs[:,1]
    out_view <<= 1
    out_view += intvecs[:,2]
    return out_view

class MeshData:
    def __init__(self, mesh=None, global_material_index_map=None):
        if mesh is not None:
            self.name = mesh.name
            mesh.calc_loop_triangles()
            triangle_count = len(mesh.loop_triangles)
        else:
            self.name = '<empty mesh>'
            triangle_count = 0

        # expanding to unindexed format
        vertex_count = 3 * triangle_count

        self.vertices = np.zeros((vertex_count,3), dtype=np.float32)
        self.normals = np.zeros((vertex_count,3), dtype=np.float32)
        self.uvs = np.zeros((vertex_count,2), dtype=np.float32)
        self.material_indices = np.zeros((triangle_count,), dtype=np.int32)
        self.indices = np.zeros((triangle_count*3,), dtype=np.uint32)

        # empty mesh is fully constructed at this point
        if vertex_count == 0:
            return

        mesh.loop_triangles.foreach_get('vertices', self.indices)

        shared_vertices = np.zeros(len(mesh.vertices)*3, dtype=np.float32)
        mesh.vertices.foreach_get('co', shared_vertices)
        self.vertices[:] = shared_vertices.reshape((len(mesh.vertices), 3))[self.indices]
        shared_vertices = None

        shared_normals = np.zeros(len(mesh.vertices)*3, dtype=np.float32)
        mesh.vertices.foreach_get('normal', shared_normals)
        self.normals[:] = shared_normals.reshape((len(mesh.vertices), 3))[self.indices]
        shared_normals = None

        if len(mesh.uv_layers) > 0:
            active_uv_layer = mesh.uv_layers.active.data

            uv_per_loop = np.zeros((len(active_uv_layer)*2,), dtype=np.float32)
            active_uv_layer.foreach_get('uv', uv_per_loop)

            loop_idcs = np.zeros((len(mesh.loop_triangles)*3,), dtype=np.int32)
            mesh.loop_triangles.foreach_get('loops', loop_idcs)
            self.uvs[:] = uv_per_loop.reshape((len(active_uv_layer),2))[loop_idcs]

            uv_per_loop = None
            loop_idcs = None

        if len(mesh.materials) > 0:
            local_material_indices = np.zeros(len(mesh.loop_triangles), dtype=np.int32)
            mesh.loop_triangles.foreach_get('material_index', local_material_indices)
            self.material_indices[:] = np.array([global_material_index_map[mat] for mat in mesh.materials])[local_material_indices]
            local_material_indices = None

    def clone(self):
        return copy.deepcopy(self)

    def transform(self, tx):
        tx = np.array(tx) # todo: this might be wrong
        txN = np.linalg.inv(tx[:3,:3]).T
        self.vertices[:] = matmulh(tx, self.vertices)
        self.normals[:] = matmul(txN, self.normals)

    def transformed(self, tx):
        clone = self.clone()
        clone.transform(tx)
        return clone

    def append(self, other, preliminary=False):
        append_list = getattr(self, 'append_list', [])
        append_list.append(other)
        self.append_list = append_list
        if not preliminary:
            self.finalize_append()
    def finalize_append(self):
        append_list = getattr(self, 'append_list', [])
        if len(append_list) == 0:
            return
        def concat(a):
            return np.concatenate((a(self, 0), *(a(x, 1+i) for i, x in enumerate(append_list))), axis=0)
        self.vertices = concat(lambda x, _: x.vertices)
        self.normals = concat(lambda x, _: x.normals)
        self.uvs = concat(lambda x, _: x.uvs)
        self.material_indices = concat(lambda x, _: x.material_indices)

        index_offsets = np.cumsum(np.array([np.max(self.indices)+1, *(np.max(x.indices)+1 for x in append_list) ], dtype=np.int64))
        self.indices = concat(lambda x, i: x.indices + index_offsets[i-1] if i > 0 else x.indices)
        del self.append_list

    def select(self, selected_vertices):
        self.vertices = self.vertices.reshape(-1, 3, 3)[selected_vertices].reshape(-1, 3)
        self.normals = self.normals.reshape(-1, 3, 3)[selected_vertices].reshape(-1, 3)
        self.uvs = self.uvs.reshape(-1, 3, 2)[selected_vertices].reshape(-1, 2)
        self.material_indices = self.material_indices[selected_vertices]
        assert False # currently not supporting index remapping here

    def split(self, separate):
        remainder = copy.copy(self)
        self.select(~separate)
        remainder.select(separate)
        return remainder

    def compute_bounds(self, recompute=True):
        if not recompute and getattr(self, "min", None) is not None:
            return
        self.min = np.amin(self.vertices, axis=0)
        self.max = np.amax(self.vertices, axis=0)
        self.quantization_base = self.min
        self.quantization_extent = self.max - self.min

    def compute_material_range(self, recompute=True):
        if not recompute and getattr(self, "material_base", None) is not None:
            return
        self.material_base = np.amin(self.material_indices)
        self.material_range_count = np.amax(self.material_indices) + 1 - self.material_base

    def cache_optimize(self, sort_by_material=False):
        self.compute_bounds(recompute=False)
        means = np.mean(self.vertices.reshape(-1, 3, 3), axis=1)
        means -= self.min
        means *= 2**10 / (self.max - self.min)
        np.clip(means, a_min=0, a_max=2**10-1, out=means)
        means = means.astype(np.uint32)
        codes = morton_codes_destructive(means)
        means = None

        gather_idcs = np.argsort(codes)
        codes = None

        if sort_by_material:
            gather_idcs2 = np.argsort(self.material_indices[gather_idcs], kind='stable')
            gather_idcs[:] = gather_idcs[gather_idcs2]
            gather_idcs2 = None

        self.vertices[:] = self.vertices.reshape(-1, 3, 3)[gather_idcs].reshape(-1, 3)
        self.normals[:] = self.normals.reshape(-1, 3, 3)[gather_idcs].reshape(-1, 3)
        self.uvs[:] = self.uvs.reshape(-1, 3, 2)[gather_idcs].reshape(-1, 2)
        self.material_indices[:] = self.material_indices[gather_idcs]
        self.indices[:] = self.indices.reshape(-1, 3)[gather_idcs].reshape(-1)

    def recompute_shared_vertices(self, recompute=True):
        if not recompute and getattr(self, "shared_vertices", None) is not None:
            return
        self.default_segments()
        base_triangle = 0
        self.shared_vertices = np.zeros_like(self.indices)
        for segment_tri_count in self.segment_triangle_counts:
            segment_indices = self.indices[3*base_triangle:3*(base_triangle+segment_tri_count)]
            segment_indices = segment_indices - np.min(segment_indices)

            shared_indices = np.zeros(np.max(segment_indices)+1, dtype=self.shared_vertices.dtype)
            shared_indices[segment_indices] = np.arange(3*base_triangle, 3*(base_triangle+segment_tri_count), dtype=shared_indices.dtype)

            self.shared_vertices[3*base_triangle:3*(base_triangle+segment_tri_count)] = shared_indices[segment_indices]

            base_triangle += segment_tri_count

    def segment_by_material(self):
        total_triangle_count = self.material_indices.shape[0]
        if total_triangle_count < 2:
            return
        count_from_beginning = np.arange(1, total_triangle_count, dtype=np.int64)[self.material_indices[1:] != self.material_indices[:-1]]
        if count_from_beginning.shape[0] > 0:
            self.segment_triangle_counts = [ count_from_beginning[0] ]
            for n in (count_from_beginning[1:] - count_from_beginning[:-1]):
                self.segment_triangle_counts.append(n)
            self.segment_triangle_counts.append(total_triangle_count - count_from_beginning[-1])
        else:
            self.segment_triangle_counts = [ total_triangle_count ]
        self.segment_triangle_counts = np.array(self.segment_triangle_counts, dtype=np.int64)
        self.segment_material_ids = self.material_indices[np.cumsum(self.segment_triangle_counts)-1]

        self.material_base = self.segment_material_ids[0]
        self.material_range_count = self.segment_material_ids[-1] - self.material_base + 1

    def default_segments(self):
        if hasattr(self, 'segment_triangle_counts'):
            return
        self.compute_material_range(recompute=False)
        total_triangle_count = self.material_indices.shape[0]
        self.segment_triangle_counts = np.array([total_triangle_count], dtype=np.int64)
        self.segment_material_ids = self.material_base

class SceneData:
    RootLocation = Matrix.Identity(4).freeze()

    class Instance:
        def __init__(self):
            self.locations = []
            self.meshes = defaultdict(list)

    def collect_meshes(self, deg, parent_instance, transform, objects, object_preprocessors=[]):
        for bobject in objects:
            for pp in object_preprocessors:
                bobject = pp(deg, bobject)
            if deg is not None:
                bobject = bobject.evaluated_get(deg)

            if getattr(bobject, 'instance_type', '') == 'COLLECTION':
                instanced_collection = bobject.instance_collection
                instance_transform = bobject.matrix_world
                instance = self.instances[instanced_collection]
                if len(instance.locations) == 0:
                    self.collect_meshes(deg, instance, transform @ instance_transform,
                        instanced_collection.objects, object_preprocessors)
                instance.locations.append(transform @ instance_transform)
            
            # skip mesh collection for now, deg needs to be re-evaluated in a second sweep of objects
            if deg is not None and len(object_preprocessors) > 0:
                continue

            mesh_or_other = bobject.data
            if mesh_or_other is None:
                continue
            if bobject.type == 'MESH':
                mesh = mesh_or_other
            else:
                try:
                    # todo: we should somehow non-redunantly handle collection conversions etc.
                    mesh = bpy.data.meshes.new_from_object(bobject, preserve_all_data_layers=True, depsgraph=deg)
                    print("Converted object ", bobject.name, " to mesh")
                except RuntimeError:
                    if bobject.type not in ['GPENCIL', 'ARMATURE', 'LIGHT', 'LIGHT_PROBE', 'CAMERA', 'SPEAKER']:
                        print("Error converting object ", bobject.name, " to mesh")
                    continue
            
            # Interpret top-level object w/ parent as instanced
            if bobject.parent is not None and bobject.parent_type == 'OBJECT' and parent_instance is self.instances[SceneData.RootLocation]:
                instance = self.instances[bobject.parent]
                if len(instance.locations) == 0:
                    instance.locations.append(transform @ bobject.parent.matrix_world)
                instance.meshes[mesh].append(bobject.matrix_local)
            else:
                parent_instance.meshes[mesh].append(bobject.matrix_world)
    
    def collect_materials(self):
        for mesh in self.meshes:
            for mat in mesh.materials:
                if mat not in self.materials:
                    next_material_id = len(self.materials)
                    self.materials[mat] = next_material_id

    def __init__(self, context, scene, objects, object_preprocessors):
        while True:
            self.instances = defaultdict(SceneData.Instance)
            self.mesh_data = dict()
            self.materials = dict()

            deg = context.evaluated_depsgraph_get()

            self.collect_meshes(deg, self.instances[SceneData.RootLocation], SceneData.RootLocation, objects, object_preprocessors)

            if len(object_preprocessors) > 0:
                object_preprocessors = []
            else:
                break

        self.instances[SceneData.RootLocation].locations = [ SceneData.RootLocation ]
        self.meshes = set(m for i in self.instances.values() for m in i.meshes.keys())
        self.collect_materials()
        #print(self.meshes)

    def triangulate(self, context):
        print('Triangulate')
        for mesh in self.meshes:
            mesh_data = MeshData(mesh, self.materials)
            self.mesh_data[mesh] = mesh_data
        for instance in self.instances.values():
            instance.mesh_data = defaultdict(list, ((self.mesh_data[mesh], txs) for mesh, txs in instance.meshes.items()))

    def flatten(self, context, keep_instances=False):
        print('Flatten Instances')
        root_instance = self.instances[SceneData.RootLocation]
        # instances will no longer correspond to Blender collections
        root_instance.meshes = None
        # unroll any eligible instances (or all if no instances kept)
        for instance_key, instance in list(self.instances.items()):
            if instance is root_instance:
                continue
            if keep_instances and len(instance.locations) >= 1:
                continue
            del self.instances[instance_key]
            for mesh, ltxs in instance.mesh_data.items():
                root_instance.mesh_data[mesh] += [ itx @ ltx for itx in instance.locations for ltx in ltxs ]
        # count mesh data users
        for instance in self.instances.values():
            for mesh_data, txs in instance.mesh_data.items():
                mesh_data.user_count = getattr(mesh_data, 'user_count', 0) + len(txs)
        print('Flatten', len(self.instances), 'Instance Meshes')
        # mesh un-flattened data mapping will no longer be relevant
        self.mesh_data = None
        # pre-transform and combine meshes
        for instance in self.instances.values():
            print('Flatten', len(instance.mesh_data.keys()), 'Meshes')
            joint_mesh = None
            for mesh, ltxs in instance.mesh_data.items():
                print('Flatten', len(ltxs), 'Locations')
                local_mesh = mesh
                for ltx in ltxs:
                    if local_mesh.user_count > 1:
                        mesh = local_mesh.clone()
                        local_mesh.user_count -= 1
                    else:
                        mesh = local_mesh
                    
                    mesh.transform(ltx)
                    if joint_mesh is None:
                        joint_mesh = mesh
                    else:
                        joint_mesh.append(mesh, preliminary=True)
            instance.meshes = None
            if joint_mesh is not None:
                joint_mesh.finalize_append()
                instance.mesh_data = { joint_mesh: [ SceneData.RootLocation ] }

    def enumerate_meshes(self):
        print('Enumerate')
        if getattr(self, 'mesh_instances', None) is None:
            self.mesh_instances = defaultdict(list)
            for instance in self.instances.values():
                for mesh, ltxs in instance.mesh_data.items():
                    self.mesh_instances[mesh] += [ itx @ ltx for itx in instance.locations for ltx in ltxs ]
        return self.mesh_instances.items()

    def enforce_material_range(self, max_index=0xff):
        for mesh, instances in list(self.enumerate_meshes()):
            mesh.compute_material_range(recompute=False)
            if mesh.material_range_count <= max_index+1:
                continue
            print("Splitting mesh", mesh.name, "to enforce a maximal material index of", max_index, "( current range is ", mesh.material_range_count, ")")
            current_mesh = mesh
            while current_mesh.material_range_count > max_index+1:
                offenders = (mesh.material_indices - mesh.material_base) > max_index
                remainder_mesh = current_mesh.split(offenders)
                current_mesh.compute_material_range() # make sure range is up-to-date
                remainder_mesh.compute_material_range() # make sure range is up-to-date
                self.mesh_instances[remainder_mesh] = instances
                current_mesh = remainder_mesh

    def optimize_materials(self):
        counted_meshes = [m for m, _ in self.enumerate_meshes()]
        unique_materials_for_meshes = []
        material_counts = np.zeros((len(counted_meshes)), dtype=np.int32)
        for mesh_idx, mesh in enumerate(counted_meshes):
            unique_materials = np.unique(mesh.material_indices)
            unique_materials_for_meshes.append(unique_materials)
            material_counts[mesh_idx] = unique_materials.size

        material_reorder_table = np.full((len(self.materials)), -1, dtype=np.int32)
        new_material_idx = 0
        for mesh_idx in np.argsort(material_counts):
            mesh = counted_meshes[mesh_idx]
            unique_materials = unique_materials_for_meshes[mesh_idx]
            print('Counted ', material_counts[mesh_idx], " for ", mesh.name)
            for old_material_idx in unique_materials:
                if material_reorder_table[old_material_idx] == -1:
                    material_reorder_table[old_material_idx] = new_material_idx
                    new_material_idx += 1

        for material, old_material_idx in self.materials.items():
            self.materials[material] = material_reorder_table[old_material_idx]

        for mesh in counted_meshes:
            mesh.material_indices = material_reorder_table[mesh.material_indices]
            mesh.compute_material_range() # make sure range is up-to-date

    def optimize(self, sort_by_material=False):
        for mesh, _ in self.enumerate_meshes():
            mesh.cache_optimize(sort_by_material=sort_by_material)

    def segment_by_material(self):
        for mesh, _ in self.enumerate_meshes():
            mesh.segment_by_material()

def quantize_vertices(vertices, quantization_base, quantization_extent):
    quantization_scaling = 0x200000 / quantization_extent
    vertices = vertices * quantization_scaling
    vertices -= quantization_base * quantization_scaling
    np.clip(vertices, a_min=0, a_max=0x1FFFFF, out=vertices)
    vertices = vertices.astype(np.uint32)
    qunatized_vertices = (vertices[:,2] << np.array([21], dtype=np.uint64))
    qunatized_vertices += vertices[:,1]
    qunatized_vertices <<= 21
    qunatized_vertices += vertices[:,0]
    return qunatized_vertices

def dequantization_scaling_offsets(quantization_base, quantization_extent):
    quantization_scaling = quantization_extent / 0x200000
    return (quantization_scaling, quantization_base + 0.5 * quantization_scaling)

# represent 0, -1 and 1 precisely by integers
def quantize_normals(normals):
    nl1 = np.abs(normals[:,0])
    nl1 += np.abs(normals[:,1])
    nl1 += np.abs(normals[:,2])
    pn = normals[:,:2] / nl1[:,np.newaxis]
    nl1 = None

    pnN = np.abs(pn[:,::-1])
    pnN -= 1.0
    pnN[pn >= 0.0] *= -1.0
    np.copyto(pn, pnN, where=normals[:,2:3] <= 0.0)
    pnN = None

    pn *= 0x8000
    pn = pn.astype(np.int32)
    np.clip(pn, a_min=-0x7FFF, a_max=0x7FFF, out=pn)
    pn += 0x8000
    pn = pn.astype(np.uint32, copy=False)

    qn = pn[:,1] << 16
    qn += pn[:,0]
    return qn

# tile cleanly by snapping boundaries to integers (wastes 0.5 step on each side)
def quantize_uvs(uv):
    # Avoid broken UV conversion by shifting each face into the positive domain.
    # We can do this because the domain is repeated.
    uv3 = uv.reshape((-1,3,2))
    uv3Org = np.floor(uv3.min(axis=1))
    uv3 -= uv3Org[:, np.newaxis, :]
    uv = uv3.reshape((-1,2))

    # Actual quantization (note that inputs should be < 8).
    uv = uv * 0xFFFF / 8.0
    uv += 0.5
    uv = uv.astype(np.int32)
    uv &= 0xFFFF
    uv = uv.astype(np.uint32, copy=False)

    quv = uv[:,1] << 16
    quv += uv[:,0]
    return quv

def write_string_to_file(f, string):
    ecs = string.encode('utf-8')
    f.write(struct.pack('q', len(ecs)))
    f.write(ecs)
    f.write(struct.pack('b', 0))

def write_scene_data(context, filepath, scene_data, allow_old_format=False, quad_share_indices=False):
    f = open(filepath, 'wb') # , encoding='utf-8'

    total_triangle_count = 0
    total_instance_count = 0
    for mesh, instances in scene_data.enumerate_meshes():
        total_triangle_count += mesh.vertices.shape[0] // 3
        total_instance_count += len(instances)
    total_material_count = len(scene_data.materials)

    file_flags = 0

    format_version = 3
    if allow_old_format and total_instance_count == 1 and file_flags == 0:
        format_version = 1

    f.write(struct.pack('II', 0x00abcabc, format_version))
    if format_version >= 3:
        f.write(struct.pack('Q', file_flags))
        header_stride_fpos = f.tell()
        f.write(struct.pack('q', 0))
        data_offset_fpos = f.tell()
        f.write(struct.pack('q', 0))
    if format_version >= 2:
        f.write(struct.pack('qq', len(scene_data.mesh_instances), total_instance_count))
    f.write(struct.pack('qq', total_material_count, total_triangle_count))
    if format_version >= 3:
        f.write(struct.pack('q', len(scene_data.mesh_instances))) # currently, instance group count == mesh count

    if format_version >= 3:
        fpos = f.tell()
        f.seek(header_stride_fpos)
        f.write(struct.pack('q', fpos))
        f.seek(fpos)

    instance_transforms = []
    mesh_index = 0
    MESH_FLAGS_INDICES = 0x1
    for mesh, instances in scene_data.enumerate_meshes():
        mesh.compute_bounds(recompute=False)

        dqs, dqb = dequantization_scaling_offsets(mesh.quantization_base, mesh.quantization_extent)
        f.write(struct.pack('fff', *dqs.flat))
        f.write(struct.pack('fff', *dqb.flat))

        if format_version < 2:
            mesh_index += 1
            continue
        assert format_version >= 3 # v2 is deprecated

        mesh_flags = 0
        if quad_share_indices:
            mesh_flags |= MESH_FLAGS_INDICES
        f.write(struct.pack('Q', mesh_flags))
        mesh_header_stride_fpos = f.tell()
        f.write(struct.pack('q', 0))
        mesh.data_offset_fpos = f.tell()
        f.write(struct.pack('q', 0))

        mesh.default_segments()
        f.write(struct.pack('qq', mesh.segment_triangle_counts.shape[0], mesh.vertices.shape[0] // 3))

        material_base = mesh.material_base
        material_count = mesh.material_range_count
        print(mesh.vertices.shape[0] // 3, material_count, material_base)
        f.write(struct.pack('iI', material_base, material_count))

        f.write(struct.pack('qqqqq', 0, 0, 0, 0, 0)) # reserved

        stc = mesh.segment_triangle_counts.astype(np.int64, copy=False)
        stc.tofile(f)
        stc = None
        stm = mesh.segment_material_ids.astype(np.int32, copy=False)
        stm.tofile(f)
        stm = None

        write_string_to_file(f, mesh.name)

        fpos = f.tell()
        f.seek(mesh_header_stride_fpos)
        f.write(struct.pack('q', fpos))
        f.seek(fpos)

        instance_transforms.append(instances if len(instances) > 0 else [ Matrix.Identity(4) ])
        mesh_index += 1
    
    written_mesh_count = mesh_index

    if format_version >= 2:
        for mesh_index in range(written_mesh_count):
            assert format_version >= 3 # v2 is deprecated

            instances = instance_transforms[mesh_index]
            instance_flags = 0
            f.write(struct.pack('Ii', instance_flags, mesh_index))

            instance_header_stride_fpos = f.tell()
            f.write(struct.pack('q', 0))
            instance_data_offset_fpos = f.tell()
            f.write(struct.pack('q', 0))

            f.write(struct.pack('q', len(instances)))

            write_string_to_file(f, 'N/A') # todo

            fpos = f.tell()
            f.seek(instance_data_offset_fpos)
            f.write(struct.pack('q', fpos))
            f.seek(fpos)

            for tx in instances:
                tx = np.array(tx)[:3].T
                f.write(struct.pack(4 * 'fff', *tx.flat))

            fpos = f.tell()
            f.seek(instance_header_stride_fpos)
            f.write(struct.pack('q', fpos))
            f.seek(fpos)

    if format_version >= 3:
        fpos = f.tell()
        f.seek(data_offset_fpos)
        f.write(struct.pack('q', fpos))
        f.seek(fpos)

    material_table = [''] * len(scene_data.materials)
    for mat, idx in scene_data.materials.items():
        material_table[idx] = mat.name
    for matname in material_table:
        write_string_to_file(f, matname)

    mesh_index = 0
    for mesh, instances in scene_data.enumerate_meshes():
        if mesh.vertices.size == 0:
            print("Empty mesh ", mesh.name)
            continue

        fpos = f.tell()
        f.seek(mesh.data_offset_fpos)
        f.write(struct.pack('q', fpos))
        f.seek(fpos)

        qpos = quantize_vertices(mesh.vertices, mesh.quantization_base, mesh.quantization_extent)
        #print(qpos.dtype)
        qpos.tofile(f)
        #print(qpos.shape, qpos.size * 8 / 1024 / 1024)
        qpos = None

        qnrm = quantize_normals(mesh.normals)
        #print(qnrm.dtype)
        #qnrm.tofile(f)
        #print(qnrm.shape, qnrm.size * 4 / 1024 / 1024)
        #qnrm = None

        quvs = quantize_uvs(mesh.uvs)
        #print(quvs.dtype)
        #quvs.tofile(f)
        np.dstack((qnrm, quvs)).tofile(f)
        #print(quvs.shape, quvs.size * 4 / 1024 / 1024)
        quvs = None

        mat_ids = mesh.material_indices - mesh.material_base
        mat_ids = mat_ids.astype(np.uint8, copy=False)
        #print(mat_ids.dtype)
        mat_ids.tofile(f)
        #print(mat_ids.shape, mat_ids.size / 1024 / 1024)
        mat_ids = None

        if quad_share_indices:
            mesh.recompute_shared_vertices(recompute=False)
            indices = mesh.shared_vertices.astype(np.uint32, copy=False)
            indices.tofile(f)
            indices = None

        #print(f.tell())

        #print(instances, mesh)

        mesh_index += 1

    return {'FINISHED'}


# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy.types import Operator


class ExportVulkanRendererScene(Operator, ExportHelper):
    """Export Vulkan Renderer scene format"""
    bl_idname = "export.vulkan_renderer_scene"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export Vulkan Renderer scene"

    # ExportHelper mixin class uses this
    filename_ext = ".vks"

    filter_glob: StringProperty(
        default="*.vks",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    selection_only: BoolProperty(
        name="Selection Only",
        description="Only export selected objects (otherwise, exports all visible objects)",
        default=False,
    )
    split_sharp_edges: BoolProperty(
        name="Split Sharp Edges",
        description="Adds a split edge modifier, if required (changes the scene permanently!)",
        default=True,
    )
    sharp_edge_angle: FloatProperty(
        name="Sharp Edge Angle",
        description="Maximum angle for edges to be shaded smoothly (degrees)",
        default=35.0,
    )
    flatten_static_meshes: BoolProperty(
        name="Flatten Static Meshes",
        description="Combines all static meshes",
        default=True,
    )
    keep_instances: BoolProperty(
        name="Keep Instances",
        description="Keep instanced objects as separate instanced meshes",
        default=True,
    )
    segment_by_material: BoolProperty(
        name="Segment by Material",
        description="Write segmented meshes that group geometry by material",
        default=True,
    )
    export_quad_share_indices: BoolProperty(
        name="Quad-share Indices",
        description="Write out an index buffer suitable for quad vertex sharing detection",
        default=True,
    )
    allow_old_format: BoolProperty(
        name="Allow 1.0 Format",
        description="Allow writing the old format for simple uninstanced, unsegmented meshes",
        default=False,
    )

    """
    type: EnumProperty(
        name="Example Enum",
        description="Choose between two items",
        items=(
            ('OPT_A', "First Option", "Description one"),
            ('OPT_B', "Second Option", "Description two"),
        ),
        default='OPT_A',
    )
    """

    def execute(self, context):
        object_preprocessors = []
        if self.split_sharp_edges:
            object_preprocessors.append(lambda c, o: apply_edge_split(o, radians(self.sharp_edge_angle)))

        allow_old_format = self.allow_old_format

        scene_data = SceneData(context, context.scene,
            context.selected_objects if self.selection_only else context.visible_objects, object_preprocessors)
        
        scene_data.triangulate(context)

        if self.flatten_static_meshes:
            scene_data.flatten(context, keep_instances=self.keep_instances)

        scene_data.enumerate_meshes()
        scene_data.optimize(sort_by_material=self.segment_by_material)
        if self.segment_by_material:
            allow_old_format = False
            scene_data.segment_by_material()
        else:
            scene_data.optimize_materials()
            scene_data.enforce_material_range()
        
        return write_scene_data(context, self.filepath, scene_data,
            allow_old_format=allow_old_format,
            quad_share_indices=self.export_quad_share_indices)


# Only needed if you want to add into a dynamic menu
def menu_func_export(self, context):
    self.layout.operator(ExportVulkanRendererScene.bl_idname, text="Vulkan Renderer Export")

