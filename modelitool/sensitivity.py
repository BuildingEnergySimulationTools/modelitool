
def modelitool_to_salib_problem(modelitool_problem):
    return {
        'num_vars': len(modelitool_problem),
        'names': list(modelitool_problem.keys()),
        'bounds': list(modelitool_problem.values())
    }