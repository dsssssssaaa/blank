import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from mathutils import Vector

bl_info = {
    "name": "Hair Card Clipping Fixer",
    "author": "Jules",
    "version": (1, 1),
    "blender": (2, 90, 0),
    "location": "View3D > Object",
    "description": "Stops hair cards from clipping with each other. Requires Blender 2.90+.",
    "category": "Object",
}

def get_hair_card_instances(context):
    """
    Get all hair card instances from the active object's particle systems.
    Returns a list of tuples, where each tuple is (instance, particle).
    """
    depsgraph = context.evaluated_depsgraph_get()
    active_obj = context.active_object
    hair_card_data = []

    if not active_obj or not active_obj.particle_systems:
        return hair_card_data

    for ps in active_obj.particle_systems:
        if ps.settings.type == 'HAIR' and ps.settings.render_type in ('OBJECT', 'COLLECTION'):
            # This check is important to avoid trying to access particles on a system that doesn't have them
            if not hasattr(ps, 'particles'):
                continue

            if ps.settings.render_type == 'OBJECT' and ps.settings.instance_object:
                instance_obj = ps.settings.instance_object
                for obj_instance in depsgraph.object_instances:
                    if obj_instance.is_instance and obj_instance.parent == active_obj and obj_instance.instance_object.original == instance_obj and hasattr(obj_instance, 'particle_system') and obj_instance.particle_system == ps:
                        if obj_instance.particle_id < len(ps.particles):
                            particle = ps.particles[obj_instance.particle_id]
                            hair_card_data.append((obj_instance, particle))
            elif ps.settings.render_type == 'COLLECTION' and ps.settings.instance_collection:
                for obj in ps.settings.instance_collection.objects:
                     for obj_instance in depsgraph.object_instances:
                        if obj_instance.is_instance and obj_instance.parent == active_obj and obj_instance.instance_object.original == obj and hasattr(obj_instance, 'particle_system') and obj_instance.particle_system == ps:
                            if obj_instance.particle_id < len(ps.particles):
                                particle = ps.particles[obj_instance.particle_id]
                                hair_card_data.append((obj_instance, particle))
    return hair_card_data


class OBJECT_OT_fix_hair_card_clipping(bpy.types.Operator):
    """Fixes hair card clipping"""
    bl_idname = "object.fix_hair_card_clipping"
    bl_label = "Fix Hair Card Clipping"
    bl_options = {'REGISTER', 'UNDO'}

    separation_distance: bpy.props.FloatProperty(
        name="Separation Distance",
        description="The distance to move intersecting hair cards",
        default=0.01,
        min=0.0,
    )

    def execute(self, context):
        hair_card_data = get_hair_card_instances(context)

        if not hair_card_data:
            self.report({'INFO'}, "No hair card instances found on the active object.")
            return {'CANCELLED'}

        bvhtrees_and_data = []
        for instance, particle in hair_card_data:
            # It's possible for an instance to exist without a mesh (e.g. if the object is hidden)
            if instance.object.type != 'MESH':
                continue

            # Important: Get the evaluated mesh from the dependency graph
            mesh = instance.object.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(mesh)

            # We need to transform the bmesh to world space to build a correct BVH tree
            bmesh.ops.transform(bm, verts=bm.verts, matrix=instance.matrix_world)

            bvhtree = BVHTree.FromBMesh(bm)
            bvhtrees_and_data.append((bvhtree, instance, particle))
            bm.free()
            bpy.data.meshes.remove(mesh)

        particle_translations = {} # particle -> vector translation

        # The following nested loop is O(n^2) in the number of hair cards.
        # For very large numbers of cards (e.g., >10,000), this could be slow.
        # A possible optimization would be to use a single spatial data structure (like a global BVH tree or a grid)
        # for all instances, but that would add significant complexity.
        for i in range(len(bvhtrees_and_data)):
            for j in range(i + 1, len(bvhtrees_and_data)):
                bvhtree1, instance1, particle1 = bvhtrees_and_data[i]
                bvhtree2, instance2, particle2 = bvhtrees_and_data[j]

                if bvhtree1.overlap(bvhtree2):
                    loc1 = instance1.matrix_world.translation
                    loc2 = instance2.matrix_world.translation

                    # Avoid division by zero if locations are the same
                    if (loc1 - loc2).length_squared == 0:
                        continue

                    direction = (loc2 - loc1).normalized()

                    if particle1 not in particle_translations:
                        particle_translations[particle1] = Vector((0.0, 0.0, 0.0))
                    if particle2 not in particle_translations:
                        particle_translations[particle2] = Vector((0.0, 0.0, 0.0))

                    # The translation is applied to particle.location which is in local space.
                    # The direction vector is in world space. We need to transform it to the parent's local space.

                    world_to_local = instance1.parent.matrix_world.inverted()

                    # For directions, we use the upper 3x3 of the matrix
                    local_direction = world_to_local.to_3x3() @ direction

                    particle_translations[particle1] -= local_direction * self.separation_distance
                    particle_translations[particle2] += local_direction * self.separation_distance


        # This part is tricky. We need to make sure we are in object mode to modify particle locations
        # and that the particle system is editable.
        active_obj = context.active_object
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        for particle, translation in particle_translations.items():
            particle.location += translation

        # We need to update the dependency graph after changing particle locations
        context.view_layer.update()

        self.report({'INFO'}, f"Checked {len(hair_card_data)} hair cards and fixed intersections.")
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(OBJECT_OT_fix_hair_card_clipping.bl_idname)

def register():
    bpy.utils.register_class(OBJECT_OT_fix_hair_card_clipping)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_fix_hair_card_clipping)
    bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == "__main__":
    register()
