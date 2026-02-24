from asf.wrapper import sized_space_from_scf, find_from_scf,  create_asf_switched_cisolver
from asf.preselection import MP2PairinfoPreselection, MP2NatorbPreselection
from asf import asfbase, ASFCI, ASFDMRG
from asf.scf import stable_scf, DEFAULT_MAX_RESTARTS, DEFAULT_SCF_SETTINGS
from asf.filters import FilterFunction
from asf.asfbase import DEFAULT_ENTROPY_THRESHOLD
import pyscf
from asf.pairinfo import MOListInfo
from pprint import pprint

import casscf.code.pyscf_tools as pyscfTools


def get_active(mf:pyscf.scf.RHF, size=None, maxM=250, verbose=False, nroots=1, switch_dmrg=10, dmrg_kwargs={}, fci_kwargs={}, max_size=12):
    """Get active space from ASF wrapper."""
    # if size is not None:
    #     try:
    #         ActSpace = sized_space_from_scf(mf, size, verbose=verbose, dmrg_kwargs={"maxM": maxM})
    #     except:
    #         ActSpace = find_from_scf(mf, max_norb=size, verbose=verbose, dmrg_kwargs={"maxM": maxM})
    # else:
    #     ActSpace = find_from_scf(mf, max_norb=12, verbose=verbose, dmrg_kwargs={"maxM": maxM})

    scf_settings = DEFAULT_SCF_SETTINGS.copy()
    if size is 0:
        size = None


    target_norb = target_nel = None
    if isinstance(size, int):
        target_norb = size
    elif isinstance(size, tuple):
        target_nel, target_norb = size
    elif size is None:
        target_norb = None
        target_nel = None
        
    try:
        print("Trying MP2 pair info preselection for ASF...")
        mf = stable_scf(
            mf.mol,
            with_uhf=True,
            max_restarts=5,
            scf_kwargs=scf_settings,
            basic_print=verbose,
        )
        pre = MP2PairinfoPreselection(mf)
        space_mp2 = pre.select()
    except:
        print("MP2 pair info preselection failed. Trying MP2 natural orbital preselection for ASF...")
        pre = MP2NatorbPreselection(mf)
        space_mp2 = pre.select()
    


    # return ActSpace
    spin, root = (mf.mol.spin, 0)
    print(f"Spin is {spin}")
    
    
    
    

    
    space_filters: list[FilterFunction] = []
    # if target_norb is not None:
    def truncate_max_norb(space: MOListInfo) -> bool:
        return len(space.mo_list) <= max_size

    space_filters.append(truncate_max_norb)


    print(f"-> Selected initial orbital window of {space_mp2.nel:d} electrons in ")
    print(f"{space_mp2.norb:d} MP2 natural orbitals.")

    calc = create_asf_switched_cisolver(
        mol=mf.mol,
        initial_space=space_mp2,
        spin=spin,
        nroots=nroots,
        switch_dmrg=switch_dmrg,
        dmrg_kwargs=dmrg_kwargs,
        fci_kwargs=fci_kwargs,
    )
    calc.calculate()

    pprint(vars(calc))
    space = None
    if target_norb is  not None:
        try:
            space = calc.find_one_sized(root=root, norb=target_norb, nel=target_nel)
        except:
            print(f"Failed to find a space of size {size} for root {root}. Trying to find a space with at most {max_size} orbitals.")
    if space is None:

        space = calc.find_one_entropy(
                root=root,
                entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
                filters=space_filters,
            )

    asfbase.print_mo_table(
        calc.one_orbital_density(root=root),
        mo_list=list(calc.mo_list),
        selections={"a": space.mo_list},
        )
    
    print(f"-> Selected an active space of {space.nel:d} electrons in {space.norb:d} orbitals.",)
    print(space.mo_list)

    cas, orb, occ, nev = pyscfTools.CASCI(mf, space.nel, space.norb, space.mo_coeff, space.mo_list)

    print(cas.e_tot)
    return space