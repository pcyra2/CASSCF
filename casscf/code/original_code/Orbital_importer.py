import bpy
import os
import numpy

Distance = 5.7
JmolLi_Colour = [0.801, 0.500, 1.000, 1.000]
JmolA_colour = [0.316, 0.129, 0.629, 1.000]
JmolB_colour = [0.629, 0.129, 0.316, 1.000]

Li_Mat = bpy.data.materials.get("Li")
Alpha_Mat = bpy.data.materials.get("alpha")
Beta_Mat = bpy.data.materials.get("Beta")
Hyd_Mat = bpy.data.materials.get("H")
collection = bpy.data.collections.new(str(Distance))
bpy.context.scene.collection.children.link(collection)

path = "F:\Phasecraft\Core_Data\Orbitals"

for i in range(7):
    for j in range(2):
        SubColl = bpy.data.collections.new(f"{Distance}-{i}_{j}")
        collection.children.link(SubColl)
        orb = f"{path}\{Distance}\Orbital_{i}-{j}.wrl"        
        print(orb)
        bpy.ops.import_scene.x3d(filepath=orb, axis_forward = 'Z', axis_up='Y')
        for k in bpy.context.selected_objects:
            try:
                material = k.active_material
                inputs = material.node_tree.nodes["Principled BSDF"].inputs
                color = inputs["Base Color"].default_value
                if numpy.round(color[0],3) == JmolLi_Colour[0]:
                    k.name = f"{Distance}_{i}-{j}_Li"
                    k.active_material = Li_Mat
                elif numpy.round(color[0],3) == JmolA_colour[0]:
                    k.name = f"{Distance}_{i}-{j}_a"
                    k.active_material = Alpha_Mat
                elif numpy.round(color[0],3) == JmolB_colour[0]:
                    k.name = f"{Distance}_{i}-{j}_b"
                    k.active_material = Beta_Mat
            except AttributeError:
                k.name = f"{Distance}_{i}-{j}_H"
                k.active_material = Hyd_Mat
            for existing_col in k.users_collection:
                existing_col.objects.unlink(k)
            SubColl.objects.link(k)
        bpy.context.scene.render.filepath = f"{path}\{Distance}\Orbital_{i}-{j}.png"
        bpy.ops.render.render(use_viewport=True, write_still=True)
#        SubColl.hide_select = True
        for k in bpy.context.selected_objects:
            k.hide_render = True
            k.hide_set(True)
            k.select_set(False)
#        bpy.data.collections[f"{Distance}-{i}_{j}"].hide_viewport = True
#        SubColl.exclude = True

