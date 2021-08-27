from tensornetwork import FiniteMPS
from dmrg import FiniteDMRG, BaseDMRG
from tensornetwork.backends import backend_factory
from mpo import FiniteXXZ
import pytest
import numpy as np


@pytest.fixture(
    name="backend_dtype_values",
    params=[('numpy', np.float64), ('numpy', np.complex128),
            ('jax', np.float64), ('jax', np.complex128),
            ('pytorch', np.float64)])
def backend_dtype(request):
  return request.param


def get_XXZ_Hamiltonian(N, Jx, Jy, Jz):
  Sx = {}
  Sy = {}
  Sz = {}
  sx = np.array([[0, 0.5], [0.5, 0]])
  sy = np.array([[0, 0.5], [-0.5, 0]])
  sz = np.diag([-0.5, 0.5])
  for n in range(N):
    Sx[n] = np.kron(np.kron(np.eye(2**n), sx), np.eye(2**(N - 1 - n)))
    Sy[n] = np.kron(np.kron(np.eye(2**n), sy), np.eye(2**(N - 1 - n)))
    Sz[n] = np.kron(np.kron(np.eye(2**n), sz), np.eye(2**(N - 1 - n)))
  H = np.zeros((2**N, 2**N))
  for n in range(N - 1):
    H += Jx * Sx[n] @ Sx[n + 1] - Jy * Sy[n] @ Sy[n + 1] + Jz * Sz[n] @ Sz[n +
                                                                           1]
  return H



#@pytest.mark.parametrize("N", [4, 6, 7])
def test_finite_DMRG_init(backend_dtype_values, N):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]
  H = get_XXZ_Hamiltonian(N, 1, 1, 1)
  eta, _ = np.linalg.eigh(H)

  mpo = FiniteXXZ(
      Jz=np.ones(N - 1),
      Jxy=np.ones(N - 1),
      Bz=np.zeros(N),
      dtype=dtype,
      backend=backend)
  D = 4
  mps = FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype, backend=backend)
  dmrg = FiniteDMRG(mps, mpo)
  t1 = time.time()
  energy = dmrg.run_one_site(num_sweeps=4, num_krylov_vecs=4)
  t2 = time.time()
  print("\n")
  print("time for one site: ", str(t2-t1))


test_finite_DMRG_init(('pytorch', np.float64), 10)
