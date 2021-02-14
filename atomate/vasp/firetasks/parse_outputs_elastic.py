# coding: utf-8


import json

from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.drones import VaspDrone
from fireworks import FiretaskBase, FWAction, explicit_serialize
from pymatgen import Structure
from pymatgen.analysis.elasticity.strain import Deformation, Strain
from pymatgen.analysis.elasticity.stress import Stress

logger = get_logger(__name__)


@explicit_serialize
class ElasticTensorToDb(FiretaskBase):
    """
    Analyzes the stress/strain data of an elastic workflow to produce
    an elastic tensor and various other quantities.

    Required params:
        structure (Structure): structure to use for symmetrization,
            input structure.  If an optimization was used, will
            look for relaxed structure in calc locs

    Optional params:
        db_file (str): path to file containing the database credentials.
            Supports env_chk. Default: write data to JSON file.
        order (int): order of fit to perform
        fw_spec_field (str): if set, will update the task doc with the contents
            of this key in the fw_spec.
        fitting_method (str): if set, will use one of the specified
            fitting methods from pymatgen.  Supported methods are
            "independent", "pseudoinverse", and "finite_difference."
            Note that order 3 and higher required finite difference
            fitting, and will override.
    """

    required_params = ["structure"]
    optional_params = ["db_file", "order", "fw_spec_field", "fitting_method"]

    def run_task(self, fw_spec):
        ref_struct = self["structure"]
        d = {"analysis": {}, "initial_structure": self["structure"].as_dict()}

        # Get optimized structure
        calc_locs_opt = [
            cl for cl in fw_spec.get("calc_locs", []) if "optimiz" in cl["name"]
        ]
        if calc_locs_opt:
            optimize_loc = calc_locs_opt[-1]["path"]
            logger.info(
                "Parsing initial optimization directory: {}".format(optimize_loc)
            )
            drone = VaspDrone()
            optimize_doc = drone.assimilate(optimize_loc)
            opt_struct = Structure.from_dict(
                optimize_doc["calcs_reversed"][0]["output"]["structure"]
            )
            d.update({"optimized_structure": opt_struct.as_dict()})
            ref_struct = opt_struct
            eq_stress = -0.1 * Stress(
                optimize_doc["calcs_reversed"][0]["output"]["ionic_steps"][-1]["stress"]
            )
        else:
            eq_stress = None

        if self.get("fw_spec_field"):
            d.update(
                {self.get("fw_spec_field"): fw_spec.get(self.get("fw_spec_field"))}
            )

        # Get the stresses, strains, deformations from deformation tasks
        defo_dicts = fw_spec["deformation_tasks"].values()
        energies, strains, deformations = [], [], []
        for defo_dict in defo_dicts:
            energies.append(Stress(defo_dict["energy"]))
            strains.append(Strain(defo_dict["strain"]))
            deformations.append(Deformation(defo_dict["deformation_matrix"]))
            # Add derived energy and strains if symmops is present
            for symmop in defo_dict.get("symmops", []):
                energies.append(Stress(defo_dict["energy"]).transform(symmop))
                strains.append(Strain(defo_dict["strain"]).transform(symmop))
                deformations.append(
                    Deformation(defo_dict["deformation_matrix"]).transform(symmop)
                )

        logger.info("Analyzing energy/strain data")

        results = {
            "energies": energies,
            "strains": strains,
            "deformations": deformations,
            "elastic_tensor": None,
        }

        # TODO based on energies and strains, compute elastic tensor and update it in
        #  results  we may want to use:
        #  from pymatgen.analysis.elasticity.elastic import ElasticTensor, ElasticTensorExpansion
        #  from pymatgen.analysis.elasticity.strain import Deformation, Strain
        #  from pymatgen.analysis.elasticity.stress import Stress

        print("Elastic tensor results", results)

        return FWAction()
