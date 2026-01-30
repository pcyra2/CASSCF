import numpy
import pickle

path = "F:\Phasecraft\Core_Data\Orbitals"

Resolution = 50
Orbitals = [0,1,2,3,4,5,6]
Spin = [0,1]
script = ""
with open("/mnt/f/Phasecraft/Core_Data/Points.pickle", "rb") as f:
    Points = pickle.load(f)
for i in range(Resolution):
    for j in Orbitals:
        for k in Spin: # Add connect delete if H-H bond forms as blender doesnt like... 
            Orb_Path = f"{path}\{Points[i]}\Orbital_{Orbitals[j]}-{Spin[k]}.cube"
            wrl_Path = f"{path}\{Points[i]}\Orbital_{Orbitals[j]}-{Spin[k]}.wrl"
            if Points[i] >=5.5:
                text = f"""load {Orb_Path}
rotate y 90
rotate x 90
connect delete
isosurface a cutoff +0.04 {Orb_Path}
isosurface b cutoff -0.04 {Orb_Path}
write {wrl_Path}
delete all
"""
            else:
                text = f"""load {Orb_Path}
rotate y 90
rotate x 90
isosurface a cutoff +0.04 {Orb_Path}
isosurface b cutoff -0.04 {Orb_Path}
write {wrl_Path}
delete all
"""
            script = script + text

with open(f"/mnt/f/Phasecraft/Core_Data/Orbitals/orb_script-windows.txt", "w") as f:
    print(script, file = f)
