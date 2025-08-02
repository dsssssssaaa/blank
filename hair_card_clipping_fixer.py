import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from mathutils import Vector

bl_info = {
    "name": "Hair Card Clipping Fixer",
    "author": "Jules",
    "version": (2, 0),
    "blender": (2, 90, 0),
    "location": "View3D > Sidebar > Hair Tools",
    "description": "Stops hair card curves from clipping with each other.",
    "category": "Object",
}

class OBJECT_OT_fix_hair_card_clipping(bpy.types.Operator):
    """Fixes hair card clipping for selected curve objects"""
    bl_idname = "object.fix_hair_card_clipping"
    bl_label = "Fix Clipping"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Only allow running the operator if there are at least two curve objects selected
        selected_curves = [obj for obj in context.selected_objects if obj.type == 'CURVE']
        return len(selected_curves) >= 2

    def execute(self, context):
        hair_card_objects = [obj for obj in context.selected_objects if obj.type == 'CURVE']

        if len(hair_card_objects) < 2:
            self.report({'INFO'}, "Please select at least two curve objects to fix clipping.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Found {len(hair_card_objects)} selected curve objects to process.")

        depsgraph = context.evaluated_depsgraph_get()
        bvhtrees_and_objects = []

        for obj in hair_card_objects:
            # Get the evaluated mesh from the dependency graph. This is important for curves.
            mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)

            bm = bmesh.new()
            bm.from_mesh(mesh)

            # We need to transform the bmesh to world space to build a correct BVH tree
            bm.transform(matrix=obj.matrix_world)

            bvhtree = BVHTree.FromBMesh(bm)
            bvhtrees_and_objects.append((bvhtree, obj))
            bm.free()
            # Do not remove the mesh here. The mesh from to_mesh() is temporary
            # and Blender will handle its memory automatically.
            # Trying to remove it manually causes a crash.

        object_translations = {} # obj -> vector translation

        for i in range(len(bvhtrees_and_objects)):
            for j in range(i + 1, len(bvhtrees_and_objects)):
                bvhtree1, obj1 = bvhtrees_and_objects[i]
                bvhtree2, obj2 = bvhtrees_and_objects[j]

                if bvhtree1.overlap(bvhtree2):
                    loc1 = obj1.matrix_world.translation
                    loc2 = obj2.matrix_world.translation

                    # Avoid division by zero if locations are the same
                    if (loc1 - loc2).length_squared == 0:
                        continue

                    direction = (loc2 - loc1).normalized()

                    if obj1 not in object_translations:
                        object_translations[obj1] = Vector((0.0, 0.0, 0.0))
                    if obj2 not in object_translations:
                        object_translations[obj2] = Vector((0.0, 0.0, 0.0))

                    separation_distance = context.scene.hair_card_separation_distance
                    object_translations[obj1] -= direction * separation_distance
                    object_translations[obj2] += direction * separation_distance

        for obj, translation in object_translations.items():
            obj.location += translation

        self.report({'INFO'}, f"Checked {len(hair_card_objects)} objects and fixed {len(object_translations)} intersections.")
        return {'FINISHED'}


class HAIR_PT_tools_panel(bpy.types.Panel):
    """Creates a Panel in the 3D view sidebar"""
    bl_label = "Hair Card Tools"
    bl_idname = "HAIR_PT_tools_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hair Tools'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Clipping Fixer:")
        row = layout.row()
        row.prop(scene, "hair_card_separation_distance")

        row = layout.row()
        row.operator(OBJECT_OT_fix_hair_card_clipping.bl_idname)


classes = (
    OBJECT_OT_fix_hair_card_clipping,
    HAIR_PT_tools_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.hair_card_separation_distance = bpy.props.FloatProperty(
        name="Separation",
        description="The distance to move intersecting hair cards",
        default=0.01,
        min=0.0,
    )

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.hair_card_separation_distance


if __name__ == "__main__":
    register()
