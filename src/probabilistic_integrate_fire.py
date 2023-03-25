# ------------------------------------------------------------------------------ #
# @Author:        J. Zierenberg
# @Email:         johannes.zierenberg@ds.mpg.de
# ------------------------------------------------------------------------------ #

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import logging
import humanize
import zarr
import os
import pyfftw
from tqdm import tqdm
from functools import partialmethod
from scipy.sparse.linalg import eigs

# we do to 1 / (1 + e^massive_number) = 0.0, numpy complains about this.
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
)
log = logging.getLogger("pif")
log.setLevel("INFO")


# the @dataclass decorator automatically creates __init__
# with the same arguments as the class attributes.
# we later save them to our zarr files.
@dataclass(kw_only=True)
class network:
    """
    Creates an instance of a network of N probabilistic integrate-and-fire neurons neurons. Neurons can be excitatory and inhibitory.

    # Parameters
    N: int
        Number of neurons in the network.
    k: int
        Number of incoming connections per neuron.
    fI: int
        fraction of inhibitory neurons
    g: float
        ratio of excitatory to inhibitory activation
    J: float
        reference coupling strength 
    rep: int
        Simulation id, used for seed.
    h: float
        rate of external activation per neuron
    duration_equil: float
        Calibration period before recording starts.
    duration_record: float
        Simulation duration in seconds, default 20 min.
    dt: float
        Simulation timestep in seconds, default 1 ms.
    meta_note: str
        A string to add, as metadata.
    """

    N: int
    k: int
    fI: float = 0.8

    J: float = 1
    g: float = 4

    rep: int = 0
    h: float = 0
 
    duration_equil: float = 0
    duration_record: float = 100
    dt: float = 1.0 / 1000

    def __post_init__(self):
        """
        Our custom init, called automatically after __init__
        Here we build the model
        """
        self.num_timesteps_equil = int(self.duration_equil / self.dt)
        self.num_timesteps_record = int(self.duration_record / self.dt)
        self.num_timesteps = self.num_timesteps_equil + self.num_timesteps_record

        # variables for current state
        self.current_timestep = 0
        self.state = np.ones(shape=self.N, dtype=bool)

        np.random.seed(42 + self.rep)

        # ER topology
        # Generate adjacency matrix, random sparse with density k_in / num_nodes
        self.adjacency_matrix = scipy.sparse.random(
            self.N,
            self.N,
            density=self.k / (self.N - 1),
            # lil format is fast for initial creation
            format="lil",
            data_rvs=None, 
        )
        # remove self-connections
        self.adjacency_matrix.setdiag(0)
        log.debug(
            "measured k without self-coupling:"
            f" {np.sum(self.adjacency_matrix > 0)/self.N}"
        )

        # normalize each column so that mean is J/k
        mean = 0.5 # when sampling from [0,1) randomly
        self.adjacency_matrix = self.adjacency_matrix.multiply(self.J/self.k/mean)
        log.debug(
            "mean input at full activity measured from adjacency matrix:"
            f" {self.adjacency_matrix.sum(axis=0).mean()}"
        )
        
        # csr is fast for matrix multiplication
        self.adjacency_matrix = self.adjacency_matrix.tocsr()

        # choose inhibitory neurons (since random can be simple first ones) and multiply weights by -g;
        # indexing not possible in lil format
        self.NI = int(self.N * self.fI)
        self.adjacency_matrix[:,0:self.NI] *= -self.g

        log.debug(
            "mean input at full activity measured from adjacency matrix:"
            f" {self.adjacency_matrix.sum(axis=0).mean()}"
        )

        self.lambda=eigs(self.adjacency_matrix, 1, which='LR')[0].real
        log.debug(
            "largest eigenvalue:"
            f"{}"
        )


    def run(self, callback=lambda: None):
        """
        Run the network for the number of timesteps specified in the constructor.
        (Input generation depends on num_timesteps, thus we specify them already there.)

        # Parameters
        callback: function
            called after each timestep (without any arguments)
        """

        self.current_timestep = 0

        # safe some time series for the equilibriation phase
        self._equil_ts_activity = np.zeros(shape=self.num_timesteps_equil)

        # safe time series for recordings
        self.ts_activity = np.zeros(shape=self.num_timesteps_record)

        log.debug("Equilibrating")
        for step in tqdm(range(self.num_timesteps_equil), desc="Equilibrating"):
            self.step()

            act = np.sum(self.state) / self.N
            self._equil_ts_activity[step] = act

            callback()

        log.debug("Recording")
        for step in tqdm(range(self.num_timesteps_record), desc="Recording"):
            self.step()

            act = np.sum(self.state) / self.N
            self.ts_activity[step] = act

            callback()

        log.info(f"simulation run done!")

    def step(self):
        """
        One timestep of the network. Uses dot products to compute the state update.
        Check the equation in the SM for details.
        """
        past_state = self.state

        p_ext = 1.0 - np.exp(-self.h*self.dt)

        # the clipping is just for safety, should not be needed since we normalized
        # columns to m during init. also implements rectified linar
        p_rec = np.clip(self.adjacency_matrix.dot(past_state), 0, 1)

        # total activation probability accounting for coalescence,
        # DOI: 10.1103/PhysRevE.101.022301
        # 1 - p_no_ext * p_no_rec
        p_act = 1.0 - (1.0 - p_ext) * (1.0 - p_rec)

        self.state = np.random.uniform(size=self.N) < p_act

        self.current_timestep += 1


# ------------------------------------------------------------------------------ #
# module level helper
# ------------------------------------------------------------------------------ #
# on the cluster we dont want to see the progress bar
# https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
def disable_tqdm():
    """Disable tqdm progress bar."""
    global tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# disable by default
disable_tqdm()


def enable_tqdm():
    """Enable tqdm progress bar."""
    global tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)


def save_object_attributes(obj, dset):
    """
    trying something new here, hoping its more future proof than saving
    metadata manually.

    saves all attributes of an object as attributes of the zarr group.
    lists and arrays asigned a string representation of their type and length.
    attributes starting with "_" are skipped.

    # Parameters
    obj: object
        object to save attributes from
    dset: zarr group or the like.
        needs to support the dset.attrs["foo"] = "bar" syntax
    """

    for key, val in obj.__dict__.items():
        if key.startswith("_"):
            continue
        rep = None
        # do not save lists, arrays etc as attributes
        try:
            if len(val.shape) == 0:
                # np scalars might have a shape but are just numbers
                raise ValueError
            rep = f"{str(type(val))[8:-2:]} with shape {val.shape}"
        except:
            if isinstance(val, (dict, list, set, tuple)):
                rep = f"{str(type(val))[8:-2:]} with length {len(val)}"
            elif val is None:
                rep = f"None"
            else:
                rep = val

        if rep is not None:
            dset.attrs[key] = rep
        else:
            log.debug(f"Failed to save {key}")


def get_extended_metadata(prefix="meta_"):
    """
    Get a bunch of standard metadata. Usually good to add as model attributes.
    Git details need the git package: pip install gitpython

    # Parameters
    prefix : str
        Prefix to add to the keys of the metadata dictionary. default: "meta_"

    # Returns
    metadata : dict

    # Example
    ```python
    model = BasicHierarchicalModel()
    # add all dict keys as attributes
    model.__dict__.update(get_metadata())
    ```
    """
    p = prefix
    md = dict()
    md[f"{p}hostname"] = os.uname()[1]
    md[f"{p}username"] = os.environ.get("USER", None)
    md[f"{p}slurm_job_id"] = os.environ.get("SLURM_JOB_ID", None)
    md[f"{p}slurm_job_array_id"] = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    md[f"{p}slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    md[f"{p}slurm_submit_dir"] = os.environ.get("SLURM_SUBMIT_DIR", None)

    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        md[f"{p}git_url"] = repo.remotes.origin.url
        md[f"{p}git_commit"] = repo.head.object.hexsha
        md[f"{p}git_branch"] = repo.active_branch.name
        md[f"{p}git_commit_message"] = repo.head.object.message
    except:
        log.debug("Could not import gitpython. Not saving git detail.")
        md[f"{p}git_url"] = None
        md[f"{p}git_commit"] = None
        md[f"{p}git_branch"] = None
        md[f"{p}git_commit_message"] = None

    return md


# ------------------------------------------------------------------------------ #
# thin wrapper to be able to run simulations from command line
# ------------------------------------------------------------------------------ #

# TODO
def main():
    """
    This simply calls the constructor and passes comman line arguments as keywords.

    # Example
    ```bash
    python3 ./src/branching_network.py \
        -N=100 -k=10 \
        -input_type="OU" \
        -meta_note="lorem ipsum"
    ```
    """
    import argparse

    parser = argparse.ArgumentParser()
    _, args = parser.parse_known_args()

    log.info(f"Branching Network from command line with args:")
    for arg in args:
        log.info(arg)

    # let useful metadata appear in the log file
    try:
        meta = get_extended_metadata(prefix="")
        log.info(f"Extended metadata for this thread:")
        for k, v in meta.items():
            log.info(f"{k}: {v}")
    except:
        pass

    # in the command line we do not want progress bars (usually run on cluster)
    disable_tqdm()

    # remove leading dashes
    args = [arg.lstrip("--") for arg in args]
    args = [arg.lstrip("-") for arg in args]

    # create a dict of keyword arguments that we can pass to the constructor
    kwargs = {k: v for k, v in [arg.split("=") for arg in args]}

    # try to cast to float, or to boolean if = "True" or "False"
    for k, v in kwargs.items():
        if v == "True" or v == "False":
            kwargs[k] = eval(v)
        elif v == "None":
            kwargs[k] = None
        else:
            # try integers first, then floats
            try:
                kwargs[k] = int(v)
            except ValueError:
                try:
                    kwargs[k] = float(v)
                except ValueError:
                    pass

    log.info(f"passing kwargs to constructor: {kwargs}")

    # we are not super careful with casting here. passing debug=False will not work.
    # use debug=0, and double check for other issues
    system = pif(**kwargs)
    system.run()


if __name__ == "__main__":
    main()
