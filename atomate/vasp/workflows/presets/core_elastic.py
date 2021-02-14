# coding: utf-8
import numpy as np
from atomate.vasp.config import ADD_WF_METADATA, DB_FILE, VASP_CMD
from atomate.vasp.powerups import add_common_powerups, add_wf_metadata
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.workflows.base.elastic_energy_method import get_wf_elastic_constant
from pymatgen import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet


# TODO combine this with the function of the same name in core.py
#  we create a new file currently to add the worflow (mjwen)
def wf_elastic_constant(
    structure: Structure, c=None, order: int = 2, sym_reduce: bool = False
):
    """
    Returns a workflow to calculate elastic tensor using the energy based method.

    Firework 1: Write vasp input set for structural relaxation, run vasp,
    pass run location, database insertion.

    Firework 2 (number of total deformations): Static runs on the deformed structures
    to compute energies.

    Firework 3: Analyze energy / strain data and fit the elastic tensor.

    Note, steps 2 and 3 are obtained from get_wf_elastic_constant().

    Args:
        structure: pymatgen material structure
        c: atomate configuration
        order: order of the elastic tensor to compute
        sym_reduce: whether to reduce computations according to material symmetry
    """

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)

    uis_optimize = {"ENCUT": 700, "EDIFF": 1e-6, "LAECHG": False, "LREAL": False}

    if order > 2:
        uis_optimize.update(
            {
                "EDIFF": 1e-10,
                "EDIFFG": -0.001,
                "ADDGRID": True,
                "ISYM": 0,
            }
        )
        # This ensures a consistent k-point mesh across all calculations
        # We also turn off symmetry to prevent VASP from changing the
        # mesh internally
        kpts_settings = Kpoints.automatic_density(structure, 40000, force_gamma=True)
        stencils = np.linspace(-0.075, 0.075, 7)
    else:
        kpts_settings = {"grid_density": 7000}
        stencils = None

    uis_static = uis_optimize.copy()
    # TODO ISIF could be set to 0, since stress is not needed (mjwen)
    uis_static.update({"ISIF": 2, "IBRION": 2, "NSW": 99, "ISTART": 1})

    # vasp input set for structure optimization
    vis_relax = MPRelaxSet(
        structure,
        force_gamma=True,
        user_incar_settings=uis_optimize,
        user_kpoints_settings=kpts_settings,
    )

    # optimization only workflow
    wf = get_wf(
        structure,
        "optimize_only.yaml",
        vis=vis_relax,
        params=[
            {
                "vasp_cmd": vasp_cmd,
                "db_file": db_file,
                "name": "elastic structure optimization",
            }
        ],
    )

    # vasp input set for static calculation
    vis_static = MPStaticSet(
        structure,
        force_gamma=True,
        lepsilon=False,
        user_incar_settings=uis_static,
        user_kpoints_settings=kpts_settings,
    )

    # deformations wflow for elasticity calculation
    wf_elastic = get_wf_elastic_constant(
        structure,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        order=order,
        stencils=stencils,
        copy_vasp_outputs=True,
        vasp_input_set=vis_static,
        sym_reduce=sym_reduce,
    )
    wf.append_wf(wf_elastic, wf.leaf_fw_ids)

    wf = add_common_powerups(wf, c)
    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf
