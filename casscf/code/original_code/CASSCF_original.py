#!/usr/bin/env python

import Code.io as io
import Code.Hamiltonian as Hamiltonian
import Code.pyscf_tools as pyscfTools
from Code.helpers.storage_conversion import dense_to_sparse

from Code.helpers.extract_integrals import construct_spin_one_body, construct_spin_two_body
import os
import glob
from pprint import pprint
import numpy
from pyscf import ao2mo
import pyscf
import sys

system_info = dict(filename=glob.glob("*.xyz")[0],
                   charge=-2,
                   spin=0,
                   basis="sto3g",
                   nFrozen=61, # Number of frozen spatial orbitals
                   CASSCF = True)

def main():
    molpath = system_info["filename"]
    for arg in sys.argv:
        if "-c" in arg:
            system_info["charge"] = int(arg.split("=")[1])
        if "-s" in arg:
            system_info["spin"] = int(arg.split("=")[1])
        if "-nf" in arg:
            system_info["nFrozen"] = int(arg.split("=")[1])
    molecule = pyscfTools.genMol(molpath, system_info["charge"], system_info["spin"], system_info["basis"], symmetry=False)
    nelec = molecule.nelec
    UHF = pyscfTools.RHF(molecule)
    if os.path.isfile("MP2_natural_orbitals.json"):
        MP2 = io.jsonRead("MP2_natural_orbitals.json")
        natocc = numpy.asarray(MP2["natocc"])
        natorb = numpy.asarray(MP2["natorb"])
    else:
        mp2, natocc , natorb = pyscfTools.make_natural_orbitals(UHF, True, False)
        MP2 =dict(natocc =natocc.tolist(), natorb = natorb.tolist(), e_tot = mp2.e_tot)
        io.jsonDump(MP2, "MP2_natural_orbitals.json")
        pprint(MP2)
        pyscf.tools.molden.dump_scf(MP2, "MP2_Nat_Orbs.molden")
    if os.path.isfile("MP2_natural_orbitals.molden") == False:
        pyscf.tools.molden.from_mo(molecule, "MP2_natural_orbitals.molden", natorb, occ=natocc)
    # if os.path.isdir("./Orbitals") == False:
    pyscfTools.genCube(natorb, molecule, "./", False)
    for i in range(len(natocc)):
        print(f"Orb: {i}, Occ: {natocc[i]}")
    if len(sys.argv) < 2 or "-" in sys.argv[-1]:
        for active_space in [2,4,6,8,10,12,14,16,18,20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]:
            active_occupied = int(active_space/2)
            mo_list = []
            for orb in range(active_occupied):
                mo_list.append(nelec[0] - orb)
                mo_list.append(nelec[0] + orb+1 )
            mo_list.sort()
            # mo_list = [169,170]
            print(f"MO List: {mo_list}")
            print(f"Active space: {active_space}")
            # print(f"Occupied orbitals: {nelec[0]}")
            if os.path.isfile(f"{active_space}_CASCI.json") is False:
                CASCI, CIorb, CIocc = pyscfTools.CASCI(UHF, nActiveElectrons=active_space, nActiveOrbitals=active_space, natocc=natocc, natorb=natorb)
                try:
                    print("CASCI Energy: ", CASCI.e_tot)
                except:
                    pass
                result = Hamiltonian.CAS_to_Hamiltonian(CASCI, mo_list, CIorb, active_space)
                io.jsonDump(result, f"{active_space}_CASCI.json")
            if os.path.isfile(f"{active_space}_CASSCF.json") is False and system_info["CASSCF"]==True:
                CASSCF, CASorb, natocc = pyscfTools.CASSCF(UHF, nActiveElectrons=active_space, nActiveOrbitals=active_space, natocc=natocc, natorb=natorb, NFrozen=system_info["nFrozen"])
                print("CASSCF Energy: ", CASSCF.e_tot)
                result = Hamiltonian.CAS_to_Hamiltonian(CASSCF, mo_list, CASorb, active_space)
                io.jsonDump(result, f"{active_space}_CASSCF.json")
        
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
    else:
        active_elec = int(sys.argv[1])
        active_orbs = int(sys.argv[2])
        assert len(sys.argv) == active_orbs + 3
        mo_list = [int(sys.argv[i+3])+1 for i in range(active_orbs)] # The +1 is becasue this list is 1 indexed
        print(mo_list)
        active_space = f"{active_orbs}_{active_elec}"
        if os.path.isfile(f"{active_space}_CASCI.json") is False:
            CASCI, CIorb, CIocc = pyscfTools.CASCI(UHF, nActiveElectrons=active_elec, nActiveOrbitals=active_orbs, natocc=natocc, natorb=natorb, cas_list=mo_list)
            try:
                print("CASCI Energy: ", CASCI.e_tot)
            except:
                pass
            result = Hamiltonian.CAS_to_Hamiltonian(CASCI, mo_list, CIorb, active_orbs=active_orbs, active_elecs=active_elec)
            io.jsonDump(result, f"{active_space}_CASCI.json")
        if os.path.isfile(f"{active_space}_CASSCF.json") is False and system_info["CASSCF"]==True:
            CASSCF, CASorb, natocc = pyscfTools.CASSCF(UHF, nActiveElectrons=active_elec, nActiveOrbitals=active_orbs, natocc=natocc, natorb=natorb, NFrozen=system_info["nFrozen"], cas_list = mo_list)
            print("CASSCF Energy: ", CASSCF.e_tot)
            result = Hamiltonian.CAS_to_Hamiltonian(CASSCF, mo_list, CASorb, active_orbs=active_orbs, active_elecs=active_elec)
            io.jsonDump(result, f"{active_space}_CASSCF.json")

        # orbitals = CASSCF.sort_mo(mo_list, natorb)
        # one_body_cas, constant_energy = CASSCF.get_h1cas(mo_coeff=orbitals)
        # two_body_cas = CASSCF.get_h2eff(mo_coeff=orbitals)
        
        # h1e = construct_spin_one_body([one_body_cas]*2,active_space)
        # h2e = construct_spin_two_body([ao2mo.restore(1,two_body_cas,active_space)]*4,active_space)
        # h2e = numpy.swapaxes(h2e,1,2)
        # # h1e = construct_spin_one_body(one_body_cas, active_space)
        # # h2e = construct_spin_two_body(ao2mo.restore(1,two_body_cas,active_space), active_space)

        # result = {
        #     "constant_energy": constant_energy,
        #     "one_body": dense_to_sparse(h1e),
        #     "two_body": dense_to_sparse(h2e),
        #     "n_electrons": active_space,
        #     "n_modes": active_space*2,
        #     "molecule_name": 'Inhibitor',
        #     "molecule_basis": 'sto3g',
        #     "molecule_uhf": False,
            
        # }
        
        # ED,_ = Hamiltonian.diagonalise_integrals(result)
        # result["ED"] = ED
        # print(ED,CASSCF.e_tot )
        # io.jsonDump(result, f"{active_space}_phasecraft.json")
        # # assert numpy.isclose(ED, CASSCF.e_tot)

if __name__ == "__main__":
    main()
