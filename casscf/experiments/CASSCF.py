#!/usr/bin/env python

import casscf.code.io as io
import casscf.code.Hamiltonian as Hamiltonian
import casscf.code.pyscf_tools as pyscfTools
from asf.wrapper import sized_space_from_scf
import os
import glob
from pprint import pprint
import numpy
import pyscf
import sys
import time


system_info = dict(filename=glob.glob("*.xyz")[0],
                   charge=0,
                   spin=0,
                   basis="sto3g",
                   nFrozen=None, # Number of frozen spatial orbitals
                   max_memory=15000,
                   max_CASCI = 10,
                   max_CASSCF = 8,
                   max_active_space = 40,
                   ccsd = False,
                   noRDM = False,
                   nevpt2 = False,
                   nelec_active = None,
                   ASF = False,
                   ASF_Size = 0)

def main():
    if os.path.isfile("./summary.json"):
        tmp = io.jsonRead("./summary.json")
        system_info["nFrozen"] = tmp["General"]["nFrozen"]
        system_info["charge"] = tmp["General"]["charge"]
        system_info["spin"] = tmp["General"]["spin"]

    if os.path.isdir("outputs") == False:
        os.mkdir("outputs")
    if os.path.isdir("hamiltonians") == False:
        os.mkdir("hamiltonians")
    molpath = system_info["filename"]
    orbs = []
    nfrozen_override = False
    for i, arg in enumerate(sys.argv):
        if "=" in arg:
            if "-c=" in arg:
                system_info["charge"] = int(arg.split("=")[1])
            elif "-s=" in arg:
                system_info["spin"] = int(arg.split("=")[1])
            elif "-nf=" in arg:
                system_info["nFrozen"] = int(arg.split("=")[1])
                nfrozen_override = True
                nfrozen = int(arg.split("=")[1])
            elif "-CI=" in arg or "-CASCI" in arg:
                system_info["max_CASCI"] = int(arg.split("=")[1])
            elif "-SCF=" in arg or "-CASSCF" in arg:
                system_info["max_CASSCF"] = int(arg.split("=")[1])
            elif "-m=" in arg:
                system_info["max_memory"] = int(arg.split("=")[1])
            elif "-size=" in arg:
                system_info["max_active_space"] = int(arg.split("=")[1])

            elif "-nelec=" in arg:
                system_info["nelec_active"] = int(arg.split("=")[1])
            elif "-asf=" in arg:
                system_info["ASF"] = True
                system_info["ASF_Size"] = int(arg.split("=")[1])
            else:
                raise Exception("Unknown argument")
        elif "-CCSD" in arg:
            system_info["ccsd"] = True
        elif "-noRDM" in arg:
            system_info["noRDM"] = True
        elif "-nevpt" in arg:
            system_info["nevpt2"] = True
        
        else:
            try:
                orb = int(i)
                if orb > 1:
                    orbs.append(int(arg))
            except:
                pass
    if len(orbs) > 0:
        assert system_info["nelec_active"] is not None
        print("Using orbitals from command line: ", orbs)
    else:
        

        print("No orbitals specified, using the systematic approach.")
    
    molecule = pyscfTools.genMol(molpath, system_info["charge"], 
                                 system_info["spin"], system_info["basis"], 
                                 symmetry=False, mem=system_info["max_memory"])
    nelec = molecule.nelec
    molecule.output = "outputs/MF.out"
    molecule.build()
    hf_s = time.perf_counter()
    mf = pyscfTools.RHF(molecule)
    hf_e = time.perf_counter()
    if os.path.isfile("MP2_natural_orbitals.json"):
        MP2 = io.jsonRead("MP2_natural_orbitals.json")
        natocc = numpy.asarray(MP2["natocc"])
        natorb = numpy.asarray(MP2["natorb"])
        system_info["nFrozen"] = MP2["nfrozen"]
    else:
        mf.mol.output = "outputs/MP2.out"
        mf.mol.build()
        mp2_s = time.perf_counter()
        mp2, natocc , natorb = pyscfTools.make_natural_orbitals(mf, True, False)
        mp2_e = time.perf_counter()
        MP2 =dict(natocc =natocc.tolist(), natorb = natorb.tolist(), 
                  e_tot = mp2.e_tot, nfrozen=mp2.frozen, e_corr = mp2.e_corr, 
                  e_hf = mp2.e_hf, norm_t2 = numpy.linalg.norm(mp2.t2), 
                  time=mp2_e - mp2_s)
        system_info["nFrozen"] = mp2.frozen
        io.jsonDump(MP2, "MP2_natural_orbitals.json")
        
    if nfrozen_override:
        system_info["nFrozen"] = nfrozen
    if os.path.isfile("MP2_natural_orbitals.molden") == False:
        pyscf.tools.molden.from_mo(molecule, "MP2_natural_orbitals.molden", natorb, occ=natocc)
    
    if os.path.isfile("./summary.json"):
        if tmp["General"]["nFrozen"] != system_info["nFrozen"]:
            print("ERROR: Frozen orbital number in summary.json does not match the current setting. please delete summary.json to continue.")
            quit()
        else:
            summary = tmp
    else:
        summary = dict()#
        summary["HF"] = dict(Energy = mf.e_tot,
                             Time = hf_e - hf_s)
        summary["MP2"] = dict(Energy = MP2["e_tot"],
                              Time = MP2["time"])
        summary["CASCI"] = dict()
        summary["CASSCF"] = dict()
        summary["General"] = dict(HOMO = nelec[0],
                                  LUMO = nelec[0]+1,
                                  charge = system_info["charge"],
                                  spin = system_info["spin"],
                                  nFrozen = system_info["nFrozen"],
                                  e_nuc = molecule.energy_nuc())
        io.jsonDump(summary, "./summary.json")

    if system_info["ccsd"] == True and "CCSD" not in summary:
        mf.mol.output = "outputs/CCSD.out"
        mf.mol.build()
        cc_s = time.perf_counter()
        CCSD, et = pyscfTools.CCSD(mf, True, True)
        
        if system_info["noRDM"] == False:
            rdm = dict(RDM1 = CCSD.make_rdm1(),
                   RDM2 = CCSD.make_rdm2())
            io.h5Write(rdm, "outputs/CCSD_RDMS.hdf5")

        cc_e = time.perf_counter()
        print("CCSD Energy: ", CCSD.e_tot)
        summary["CCSD"] = dict(
            e_tot = CCSD.e_tot,
            e_corr = CCSD.e_corr,
            tripples_correction = et,
            norm_t1 = numpy.linalg.norm(CCSD.t1),
            norm_t2 = numpy.linalg.norm(CCSD.t2),
            e_hf = CCSD.e_hf,
            time = cc_e - cc_s)
        io.jsonDump(summary, "./summary.json")

        
    if len(orbs) == 0 and system_info["max_active_space"] != 0 and system_info["ASF"] == False: ## Brute force CAS method
        print("No orbitals specified, running systematic CASSCF/CASCI calculations.")
        for active_space in range(2, system_info["max_active_space"]+2, 2):
            print(f"Running active space: {active_space}")
            
            active_occupied = int(active_space/2)
            mo_list = []
            for orb in range(active_occupied):
                mo_list.append(nelec[0] - orb)
                mo_list.append(nelec[0] + orb+1 )
            mo_list.sort()
            print(f"MO List: {mo_list}")

            if os.path.isfile(f"hamiltonians/{active_space}_CASCI.json") is False:
                mf.mol.output = f"outputs/CASCI_{active_space}.out"
                mf.mol.build()
                ci_s = time.perf_counter()
                CASCI, CIorb, CIocc, nevpt = pyscfTools.CASCI(mf, nActiveElectrons=active_space, 
                                                       nActiveOrbitals=active_space, natocc=natocc, 
                                                       natorb=natorb, cas_list=mo_list, 
                                                       max_run=system_info["max_CASCI"], nevpt2 = system_info["nevpt2"])
                ci_e = time.perf_counter()
                try:
                    print("CASCI Energy: ", CASCI.e_tot)
                except:
                    if active_space <= system_info["max_CASCI"]:
                        print("CASCI did not converge.")

                result = Hamiltonian.CAS_to_Hamiltonian(CASCI, mo_list, CIorb, active_space)
                io.jsonDump(result, f"hamiltonians/{active_space}_CASCI.json")
                
                summary["CASCI"][active_space] = dict(e_tot = CASCI.e_tot, 
                                                      e_cas = CASCI.e_cas, 
                                                      time= ci_e - ci_s)
                if nevpt is not None:
                    summary["CASCI"][active_space]["nevpt2"] = nevpt.e_corr
                io.jsonDump(summary, "./summary.json")

            if os.path.isfile(f"hamiltonians/{active_space}_CASSCF.json") is False and active_space <= system_info["max_CASSCF"]:
                mf.mol.output = f"outputs/CASSCF_{active_space}.out"
                mf.mol.build()
                cs_s = time.perf_counter()
                CASSCF, CASorb, natocc, nevpt = pyscfTools.CASSCF(mf, nActiveElectrons=active_space, 
                                                           nActiveOrbitals=active_space, natocc=natocc, 
                                                           natorb=natorb, NFrozen=system_info["nFrozen"], 
                                                           max_run = system_info["max_CASSCF"],
                                                           nevpt2 = system_info["nevpt2"])
                cs_e = time.perf_counter()
                try:
                    print("CASSCF Energy: ", CASSCF.e_tot)
                except:
                    if active_space <= system_info["max_CASSCF"]:
                        print("CASSCF did not converge.")
                result = Hamiltonian.CAS_to_Hamiltonian(CASSCF, mo_list, CASorb, active_space)
                io.jsonDump(result, f"hamiltonians/{active_space}_CASSCF.json")

                summary["CASSCF"][active_space] = dict(e_tot = CASSCF.e_tot, 
                                                       e_cas = CASSCF.e_cas, 
                                                       time = cs_e - cs_s)
                if nevpt is not None:
                    summary["CASSCF"][active_space]["nevpt2"] = nevpt
                io.jsonDump(summary, "./summary.json")

    elif len(orbs) == 0 and system_info["max_active_space"] != 0 and system_info["ASF"] == True and system_info["ASF_Size"] > 1: ### ASF Method
        print(f"INFO: Running ASF approach to select active space. Generating active space of size {system_info['ASF_Size']} using ASF.")
        ActSpace = sized_space_from_scf(mf, system_info["ASF_Size"])
        active_space = f"{system_info['ASF_Size']}_ASF"
        if os.path.isfile(f"hamiltonians/{active_space}_CASCI.json") is False:
                mf.mol.output = f"outputs/CASCI_{active_space}.out"
                mf.mol.build()
                ci_s = time.perf_counter()
                CASCI, CIorb, CIocc, nevpt = pyscfTools.CASCI(mf, nActiveElectrons=ActSpace.nel, 
                                                       nActiveOrbitals= ActSpace.norb, natocc=[], 
                                                       natorb=ActSpace.mo_coeff, cas_list=ActSpace.mo_list, 
                                                       max_run=system_info["max_CASCI"], nevpt2 = system_info["nevpt2"])
                ci_e = time.perf_counter()
                try:
                    print("CASCI Energy: ", CASCI.e_tot)
                except:
                    if active_space <= system_info["max_CASCI"]:
                        print("CASCI did not converge.")

                result = Hamiltonian.CAS_to_Hamiltonian(CASCI, ActSpace.mo_list, CIorb, ActSpace.norb, ActSpace.nel)
                result["coeff"] = ActSpace.mo_list
                io.jsonDump(result, f"hamiltonians/{active_space}_CASCI.json")

                summary["CASCI"][active_space] = dict(e_tot = CASCI.e_tot, 
                                                      e_cas = CASCI.e_cas, 
                                                      time= ci_e - ci_s,
                                                      orbitals = ActSpace.mo_list,
                                                      coeff = ActSpace.mo_list)
                if nevpt is not None:
                    summary["CASCI"][active_space]["nevpt2"] = nevpt.e_corr
                io.jsonDump(summary, "./summary.json")



    elif len(orbs) > 1: ### Manual selection method
        active_space = f"{system_info['nelec_active']}-{len(orbs)}"
        mo_list = orbs
        if os.path.isfile(f"hamiltonians/{active_space}_CASCI.json") is False:
                mf.mol.output = f"outputs/CASCI_{active_space}.out"
                mf.mol.build()
                ci_s = time.perf_counter()
                CASCI, CIorb, CIocc, nevpt = pyscfTools.CASCI(mf, nActiveElectrons=system_info["nelec_active"], 
                                                       nActiveOrbitals=len(orbs), natocc=natocc, 
                                                       natorb=natorb, cas_list=orbs, 
                                                       max_run=system_info["max_CASCI"], nevpt2 = system_info["nevpt2"])
                ci_e = time.perf_counter()
                try:
                    print("CASCI Energy: ", CASCI.e_tot)
                except:
                    if active_space <= system_info["max_CASCI"]:
                        print("CASCI did not converge.")

                result = Hamiltonian.CAS_to_Hamiltonian(CASCI, mo_list, CIorb, len(mo_list), system_info["nelec_active"])
                io.jsonDump(result, f"hamiltonians/{active_space}_CASCI.json")
                
                summary["CASCI"][active_space] = dict(e_tot = CASCI.e_tot, 
                                                      e_cas = CASCI.e_cas, 
                                                      time= ci_e - ci_s,
                                                      orbitals = mo_list)
                if nevpt is not None:
                    summary["CASCI"][active_space]["nevpt2"] = nevpt.e_corr
                io.jsonDump(summary, "./summary.json")

        if os.path.isfile(f"hamiltonians/{active_space}_CASSCF.json") is False and len(mo_list) <= system_info["max_CASSCF"]:
            mf.mol.output = f"outputs/CASSCF_{active_space}.out"
            mf.mol.build()
            cs_s = time.perf_counter()
            CASSCF, CASorb, natocc, nevpt = pyscfTools.CASSCF(mf, nActiveElectrons=system_info["nelec_active"], 
                                                        nActiveOrbitals=len(orbs), natocc=natocc, 
                                                        natorb=natorb, NFrozen=system_info["nFrozen"], 
                                                        max_run = system_info["max_CASSCF"],
                                                        nevpt2 = system_info["nevpt2"])
            cs_e = time.perf_counter()
            try:
                print("CASSCF Energy: ", CASSCF.e_tot)
            except:
                if active_space <= system_info["max_CASSCF"]:
                    print("CASSCF did not converge.")
            result = Hamiltonian.CAS_to_Hamiltonian(CASSCF, mo_list, CASorb, len(mo_list), system_info["nelec_active"])
            io.jsonDump(result, f"hamiltonians/{active_space}_CASSCF.json")

            summary["CASSCF"][active_space] = dict(e_tot = CASSCF.e_tot, 
                                                    e_cas = CASSCF.e_cas, 
                                                    time = cs_e - cs_s,
                                                    orbitals = mo_list)
            if nevpt is not None:
                summary["CASSCF"][active_space]["nevpt2"] = nevpt
            io.jsonDump(summary, "./summary.json")


if __name__ == "__main__":
    main()
