"""Compatibility aliases loaded automatically by worker-side Python."""

try:
    import pulp
except ImportError:
    pulp = None

if (
    pulp is not None
    and not hasattr(pulp, "list_solvers")
    and hasattr(pulp, "listSolvers")
):
    pulp.list_solvers = pulp.listSolvers
