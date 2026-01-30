import os
import io

def OrbVis():
    jmol_script = """set frank off
set antialiasimages false
background [x010101]
"""
        for orb in mo_list:
            if os.path.isfile(f"./Orbitals/Orbital_{orb}-0.wrl") == False:
                orb_path = f"./Orbitals/Orbital_{orb}-0.cube"
                jmol_script += f"""load {orb_path}
isosurface a cutoff +0.05 {orb_path} color TRANSLUCENT red
isosurface b cutoff -0.05 {orb_path} color TRANSLUCENT blue
write ./Orbitals/Orbital_{orb}-0.wrl
write PNGT ./Orbitals/Orbital_{orb}-0.png
delete all

"""
    io.textDump(jmol_script, "cube_to_wrl.txt")

    return text