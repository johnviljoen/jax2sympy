import cvxopt
import cvxopt.cholmod
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import lineax as lx
from jax.experimental.sparse import BCOO
import equinox as eqx
import numpy as np


def add_output_info(problem_fn):
    """Decorator to add information to the problem output."""
    def wrapper(width, height, target_vf, *args, **kwargs):
        out = problem_fn(width, height, target_vf, *args, **kwargs)
        normals = out["normals"]
        forces = out["forces"]
        fixdofs, freedofs = _get_fixed_and_free_dofs(
            normals, forces, width, height)
        out["fixdofs"] = fixdofs
        out["freedofs"] = freedofs

        # Add x and y lists for the problem
        out["_k_x_list"], out["_k_y_list"] = get_k_x_and_y_lists(
            width, height)
        out["_n_valid_inds"] = int(get_num_valid_inds(out["_k_x_list"],
                                                      out["_k_y_list"],
                                                      freedofs))
        return out
    return wrapper


def get_k_x_and_y_lists(Nx, Ny):
    """Get the x and y lists for the stiffness matrix.

    Assumes counter-clockwise ordering of the nodes in a Quad4 element.
    starting from the bottom left node.
    """
    # get position of the nodes of each element in the stiffness matrix
    ely, elx = jnp.meshgrid(jnp.arange(Ny), jnp.arange(Nx))  # x, y coords
    ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)
    n4 = (Ny + 1) * (elx + 0) + (ely + 0)
    n3 = (Ny + 1) * (elx + 1) + (ely + 0)
    n2 = (Ny + 1) * (elx + 1) + (ely + 1)
    n1 = (Ny + 1) * (elx + 0) + (ely + 1)
    edof = jnp.array(
        [
            2 * n1,
            2 * n1 + 1,
            2 * n2,
            2 * n2 + 1,
            2 * n3,
            2 * n3 + 1,
            2 * n4,
            2 * n4 + 1,
        ]
    )
    edof = edof.T[0]
    # flat list pointer of each node in an element
    x_list = jnp.repeat(edof, 8)
    y_list = jnp.tile(edof, 8).flatten()
    return x_list, y_list


def _get_fixed_and_free_dofs(normals, forces, width, height):
    normals = np.ravel(normals)
    forces = np.ravel(forces)
    fixdofs = np.flatnonzero(normals.ravel())
    alldofs = np.arange(2 * (width + 1) * (height + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))
    return fixdofs, freedofs

# == Problem Definitions ==


@add_output_info
def mbb_beam(width, height, target_vf):
    normals = np.zeros((width + 1, height + 1, 2))
    normals[-1, -1, 1] = 1
    normals[0, :, 0] = 1

    forces = np.zeros((width + 1, height + 1, 2))
    forces[0, 0, 1] = -1

    return {
        "normals": normals,
        "forces": forces,
        "target_vf": target_vf,
        "id": 0,
        "reference_compliance": 1.0,
        "strain_energy": None,
    }


def get_elem_stiffness_matrix(young, poisson, Lx, Ly, Nx, Ny):
    """Obtain the stiffness matrix for a single element.

    Assumes a 2D plane stress problem with Quad4 elements.
    """
    H = Ly / Ny
    W = Lx / Nx
    # Element stiffness matrix (scaled)
    E, nu = young, poisson
    k = jnp.array(
        [
            12 * H**2 - 6 * W**2 * nu + 6 * W**2,
            4.5 * H * W * (1 + nu),
            -12 * H**2 - 3 * W**2 * nu + 3 * W**2,
            36 * H * W * (-0.125 + 0.375 * nu),
            -6 * H**2 + 3 * W**2 * nu - 3 * W**2,
            -4.5 * H * W * (1 + nu),
            6 * H**2 + 6 * W**2 * nu - 6 * W**2,
            36 * H * W * (0.125 - 0.375 * nu),
        ]
    )
    return (
        E
        / (36 * W * H * (1 - nu**2))
        * jnp.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )


def young_modulus(x, E_0, E_min, p=3.0):
    return E_min + x**p * (E_0 - E_min)


def compliance(Ny, Nx, u, ke):
    """Computes un-scaled compliance of the structure.

    c(x) = \Sum_{e=1}^{Nx*Ny} u^T @ ke @ u
         = \Sum_{e=1}^{Nx*Ny} E(xe) u^T @ ke0 @ u

    This function calculates u^T @ ke0 @ u for each element.
    """
    # TODO: Cant this just be F @ u?
    # index map
    # Ny, Nx = x_phys.shape
    ely, elx = jnp.meshgrid(jnp.arange(Ny), jnp.arange(Nx))  # x, y coords

    # nodes
    n4 = (Ny + 1) * (elx + 0) + (ely + 0)
    n3 = (Ny + 1) * (elx + 1) + (ely + 0)
    n2 = (Ny + 1) * (elx + 1) + (ely + 1)
    n1 = (Ny + 1) * (elx + 0) + (ely + 1)

    all_ixs = jnp.array(
        [
            2 * n1,
            2 * n1 + 1,
            2 * n2,
            2 * n2 + 1,
            2 * n3,
            2 * n3 + 1,
            2 * n4,
            2 * n4 + 1,
        ]
    )

    # select from u matrix
    u_selected = u[all_ixs]

    # compute U.T @ ke @ U in a vectorized way
    ke_u = jnp.einsum("ij,jkl->ikl", ke, u_selected)
    ce = jnp.einsum("ijk,ijk->jk", u_selected, ke_u)
    return ce.T  # This is elementwise compliances [same shape as x_phys]


# Solver using Lineax wrapper


class SparseMatrixLinearOperator(lx.AbstractLinearOperator, strict=True):
    bcoo_matrix: BCOO

    def __init__(self, bcoo_matrix):
        self.bcoo_matrix = bcoo_matrix

    def mv(self, x):
        return self.bcoo_matrix @ x

    def as_matrix(self):
        return self.bcoo_matrix.todense()

    def transpose(self):
        return self  # SparseMatrixLinearOperator(self.bcoo_matrix.transpose())

    def in_structure(self):
        _, in_size = self.bcoo_matrix.shape
        return jax.ShapeDtypeStruct(shape=(in_size,),
                                    dtype=self.bcoo_matrix.dtype)

    def out_structure(self):
        out_size, _ = self.bcoo_matrix.shape
        return jax.ShapeDtypeStruct(shape=(out_size,),
                                    dtype=self.bcoo_matrix.dtype)


@lx.is_symmetric.register(SparseMatrixLinearOperator)
def _(operator):
    return True


@lx.linearise.register(SparseMatrixLinearOperator)
def _(operator):
    return operator


class CVXOPTSolver(lx.AbstractLinearSolver):
    symmetrsize_before_solve: bool = False

    def __init__(self, symmetrsize_before_solve=False):
        self.symmetrsize_before_solve = symmetrsize_before_solve

    def init(self, operator, options):
        return operator  # return the solver state

    def compute(self, solver_state, rhs, options):
        bcoo_matrix = solver_state.bcoo_matrix
        data = bcoo_matrix.data
        indices = bcoo_matrix.indices

        def host_solve(data, indices, rhs):
            data = np.array(data).astype(rhs.dtype)
            indices = np.array(indices.T)
            rhs = np.array(rhs)
            K = cvxopt.spmatrix(data, indices[0, :], indices[1, :])
            if self.symmetrsize_before_solve:
                K = (K + K.T) / 2.0
            B = cvxopt.matrix(rhs)
            cvxopt.cholmod.linsolve(K, B)
            return np.array(B).astype(rhs.dtype).reshape(rhs.shape)

        # call the solver
        result_shape_dtypes = jax.ShapeDtypeStruct(
            jnp.broadcast_shapes(rhs.shape), rhs.dtype)
        sol = jax.pure_callback(
            host_solve,
            result_shape_dtypes,
            data,
            indices,
            rhs,
            vmap_method="sequential")
        return sol, lx.RESULTS.successful, {}

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False

    def conj(self, solver_state, options):
        return solver_state.transpose(), options

    def transpose(self, state, options):
        return state.transpose(), options


def get_k_entries(stiffness, ke):
    Ny, Nx = stiffness.shape
    kd = stiffness.T.reshape(Nx * Ny, 1, 1)
    value_list = (kd * jnp.tile(ke, kd.shape)).flatten()
    return value_list


def inverse_permutation_jax(indices):
    inverse_perm = jnp.zeros(len(indices), dtype=int)
    inverse_perm = inverse_perm.at[indices].set(
        jnp.arange(len(indices), dtype=int))
    return inverse_perm


def safe_boolean_indexing(arr, mask, n_valid_items):
    """Safely index an array with a boolean mask.

    https://github.com/jax-ml/jax/issues/2765
    """
    # Get 1 for valid entries, 0 otherwise
    # TODO: Change to float
    scores = mask.astype(jnp.float32)
    # `top_k` will bring valid entries to front
    _, topk_indices = jax.lax.top_k(scores, k=arr.shape[0])
    sorted_array = arr[topk_indices]
    return sorted_array[:n_valid_items]


def filter_dofs(freedofs, fixdofs, k_ylist, k_xlist, n_valid_inds):
    """Filter the dof indices to only include the free dofs.

    The last argument should be static for JIT compilation.
    """
    index_map = inverse_permutation_jax(jnp.concatenate([freedofs, fixdofs]))
    keep = jnp.isin(k_xlist, freedofs) & jnp.isin(k_ylist, freedofs)
    i_temp = index_map[k_xlist]  # k_xlist[keep]
    j_temp = index_map[k_ylist]
    i = safe_boolean_indexing(i_temp, keep, n_valid_inds)
    j = safe_boolean_indexing(j_temp, keep, n_valid_inds)
    return index_map, keep, jnp.stack([i, j])


def get_num_valid_inds(x_list, y_list, freedofs):
    keep = jnp.isin(x_list, freedofs) & jnp.isin(y_list, freedofs)
    n_valid_inds = jnp.sum(keep)
    return n_valid_inds


def setup_solver_with_lineax(cfg: DictConfig, solver=None,
                             symmetrsize_before_solve=True):
    """Returns a function that computes the compliance.

    penal_for_stress_calc is the penalty for the stress calculation.
     This should be = 3.0 in the limit case (qp relaxtation starts at q<p and
     ends at q=p). This is the default value.
     See Duysinx and Bendsoe:
     TOPOLOGY OPTIMIZATION OF CONTINUUM STRUCTURES
            WITH LOCAL STRESS CONSTRAINTS
    - No singularity problem if q < p
    """
    # Get details from config
    young_mod, young_min, poisson, Nx, Ny, Lx, Ly = (
        cfg.young, cfg.young_min, cfg.poisson, cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
    # prereqs for compliance calculation
    ke = get_elem_stiffness_matrix(young_mod, poisson, Lx, Ly, Nx, Ny)
    if solver is None:
        solver = CVXOPTSolver(symmetrsize_before_solve)

    @eqx.filter_jit
    def total_compliance(x_filtered, problem, penalty=3.0):
        x_filtered = x_filtered.reshape(Ny, Nx, order="F")
        penalty = jnp.asarray(penalty)
        # clip x to avoid numerical issues
        x_filtered = jnp.clip(x_filtered, 1e-3, 1.0)
        # get details from problem
        n_valid_inds = problem["_n_valid_inds"]
        freedofs = problem["freedofs"]
        fixdofs = problem["fixdofs"]
        x_list = problem["_k_x_list"]
        y_list = problem["_k_y_list"]
        forces = jnp.ravel(problem["forces"])
        # get the stiffness matrix
        stiffness = young_modulus(x_filtered, young_mod, young_min, p=penalty)
        k_entries = get_k_entries(stiffness, ke)
        index_map, keep, indices = filter_dofs(
            freedofs, fixdofs, y_list, x_list,
            n_valid_inds)
        filtered_k_entries = safe_boolean_indexing(k_entries, keep,
                                                   n_valid_items=n_valid_inds)
        # k_entries[keep]
        filtered_force = forces[freedofs]
        bcoo_matrix = BCOO(
            (filtered_k_entries, indices.T), shape=(
                filtered_force.size, filtered_force.size))
        bcoo_matrix = (bcoo_matrix + bcoo_matrix.T)/2.0
        operator = SparseMatrixLinearOperator(bcoo_matrix)
        u_filtered = lx.linear_solve(
            operator, filtered_force, solver=solver, options=None).value
        u_values = jnp.concatenate(
            [u_filtered.ravel(), jnp.zeros(len(fixdofs))])
        u_values = u_values[index_map]
        # Calculate compliance and stress
        ce_unscaled = compliance(Ny, Nx, u_values, ke)
        ce_scaled = stiffness * ce_unscaled
        c = jnp.sum(ce_scaled)
        return c

    return total_compliance


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    # Test the compliance function
    cfg = DictConfig({
        "young": 1.0,
        "young_min": 1e-9,
        "poisson": 0.3,
        "Nx": 10,
        "Ny": 10,
        "Lx": 10.0,
        "Ly": 10.0
    })
    Nx, Ny, vf = cfg.Nx, cfg.Ny, 1.0
    problem = mbb_beam(Nx, Ny, vf)
    x = np.ones((Ny, Nx)) * vf
    # Now with new solver
    # problem = mtp.dataset.to.cantilever_beam(Nx, Ny, vf, distribute=True)
    # x = np.ones((Ny, Nx)) * vf

    new_solver = setup_solver_with_lineax(cfg)
    val2 = new_solver(x, problem, 3.0)
    print("Compliance:", val2)

    def fn(x): return new_solver(x, problem, 3.0)
    # Test gradient
    grads = jax.jacrev(fn)(x)
    print("Gradient:", jnp.linalg.norm(grads))

    # Test with sparsity
    import jax2sympy as j2s
    from jax2sympy.sparsify import get_sparsity_pattern, sparse_jacobian
    spar_pattern = get_sparsity_pattern(
        fn, x, type='jacobian')
    sparse_grads = sparse_jacobian(
        fn, spar_pattern, (x.shape, ))

    pass
    # from jax.test_util import check_grads
    # check_grads(lambda x: new_solver(x, problem, 3.0)[0], (x,), order=1)
    # check_grads(lambda x: new_solver(x, problem, 3.0)[1].sum(), (x,), order=1)