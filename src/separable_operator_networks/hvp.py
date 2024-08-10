"""This module enables convenient computation of Hessian-vector products
(HVPs) using JAX. It supports four different modes of HVP computation:
- forward-over-forward (fwdfwd)
- reverse-over-reverse (revrev)
- forward-over-reverse (fwdrev)
- reverse-over-forward (revfwd)

This module was adapted from the implementation in the SPINN repo:
    https://github.com/stnamjef/SPINN
"""

import jax


# forward over forward
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.jvp(f, (primals,), tangents)[1]  # noqa: E731
    primals_out, tangents_out = jax.jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# reverse over reverse
def hvp_revrev(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.vjp(f, primals)[1](tangents)  # noqa: E731
    primals_out, vjp_fn = jax.vjp(g, primals)
    tangents_out = vjp_fn((tangents,))[0]
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# forward over reverse
def hvp_fwdrev(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.vjp(f, primals)[1](tangents[0])[0]  # noqa: E731
    primals_out, tangents_out = jax.jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# reverse over forward
def hvp_revfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.jvp(f, primals, tangents)[1]  # noqa: E731
    primals_out, vjp_fn = jax.vjp(g, primals)
    tangents_out = vjp_fn(tangents[0])[0][0]
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out
