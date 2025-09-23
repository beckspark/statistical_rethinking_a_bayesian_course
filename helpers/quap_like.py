import ast

import numpy as np
import pandas as pd
import pymc as pm

def quap_like(formula_list: list, data: pd.DataFrame):
    """
    A helper function to mimic the behavior of R's rethinking::quap.
    It builds and runs a PyMC model based on a simplified formula syntax,
    then returns the Maximum a Posteriori (MAP) estimate.

    Args:
        formula_list (list): A list of strings representing the model formulas.
            The first string is the likelihood, subsequent strings are priors.
            Example: ['M ~ dnorm(mu, sigma)', 'mu <- a + bAM * A', ...]
        data (pd.DataFrame): The pandas DataFrame containing the observed data.

    Returns:
        dict: A dictionary of the MAP estimates for each parameter.
    """

    # Check for valid inputs
    if not isinstance(formula_list, list) or not all(
        isinstance(f, str) for f in formula_list
    ):
        raise TypeError("formula_list must be a list of strings.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # Initialize a dictionary to store parsed priors and likelihood
    priors = {}
    likelihood_formula = None
    linear_models = {}

    # Helper function to parse formula strings
    def parse_formula(formula):
        formula = formula.replace(" ", "")

        # Parse linear model relationships (e.g., 'mu <- a + bAM * A')
        if "<-" in formula:
            parts = formula.split("<-")
            if len(parts) != 2:
                raise ValueError(f"Invalid linear model formula: {formula}")
            variable_name = parts[0]
            expression = parts[1]
            linear_models[variable_name] = expression
            return None  # Not a prior or likelihood

        # Parse dnorm(), dexp(), etc.
        if "~" in formula:
            parts = formula.split("~")
            if len(parts) != 2:
                raise ValueError(f"Invalid formula: {formula}")

            variable_name = parts[0]
            distribution_str = parts[1]

            # Use a safe way to evaluate the distribution string
            try:
                # ast.parse safely creates an abstract syntax tree without executing code
                parsed_expr = ast.parse(distribution_str).body[0].value
                dist_name = parsed_expr.func.id
                dist_args = [
                    arg.id if isinstance(arg, ast.Name) else arg.value
                    for arg in parsed_expr.args
                ]

                # Check for observed variables for likelihood
                if variable_name in data.columns:
                    return {
                        "type": "likelihood",
                        "variable": variable_name,
                        "dist": dist_name,
                        "params": dist_args,
                    }
                else:
                    return {
                        "type": "prior",
                        "variable": variable_name,
                        "dist": dist_name,
                        "params": dist_args,
                    }
            except (SyntaxError, AttributeError):
                raise ValueError(f"Could not parse distribution: {distribution_str}")

        raise ValueError(f"Formula does not contain '~' or '<-': {formula}")

    # Parse all formulas
    for formula in formula_list:
        parsed = parse_formula(formula)
        if parsed:
            if parsed["type"] == "likelihood":
                if likelihood_formula:
                    raise ValueError("Only one likelihood formula is allowed.")
                likelihood_formula = parsed
            else:
                priors[parsed["variable"]] = parsed

    if not likelihood_formula:
        raise ValueError("A likelihood formula (e.g., 'y ~ dnorm(...)') is required.")

    # Build the PyMC model
    with pm.Model() as model:
        # Define priors
        for var_name, prior_info in priors.items():
            dist_name = prior_info["dist"]
            dist_params = prior_info["params"]

            # Map simplified names to PyMC distributions
            if dist_name == "dnorm":
                pm.Normal(var_name, mu=dist_params[0], sigma=dist_params[1])
            elif dist_name == "dexp":
                pm.Exponential(var_name, lam=dist_params[0])
            else:
                raise NotImplementedError(f"Distribution '{dist_name}' not supported.")

        # Evaluate and define linear models
        for var_name, expression in linear_models.items():
            # A simple eval is okay here as it's within a controlled environment,
            # but in a real-world app, you'd want a safer expression parser.
            expression_eval = eval(
                expression,
                {},
                {**model.named_vars, **{col: data[col] for col in data.columns}},
            )
            pm.Deterministic(var_name, expression_eval)

        # Define the likelihood
        likelihood_var = likelihood_formula["variable"]
        likelihood_dist = likelihood_formula["dist"]
        likelihood_params = likelihood_formula["params"]

        if likelihood_dist == "dnorm":
            mu_val = model.named_vars[likelihood_params[0]]
            sigma_val = model.named_vars[likelihood_params[1]]
            pm.Normal(
                likelihood_var,
                mu=mu_val,
                sigma=sigma_val,
                observed=data[likelihood_var],
            )
        else:
            raise NotImplementedError(
                f"Likelihood distribution '{likelihood_dist}' not supported."
            )

    # Find the MAP estimate
    map_estimate = pm.find_MAP(model=model)
    return {k: v.item() for k, v in map_estimate.items()}  # Return as a dictionary


# --- Example Usage ---

if __name__ == "__main__":
    # 1. Create the same model specification as in the R code
    model_spec = [
        "M ~ dnorm(mu, sigma)",
        "mu <- a + bAM * A",
        "a ~ dnorm(0, 0.2)",
        "bAM ~ dnorm(0, 0.5)",
        "sigma ~ dexp(1)",
    ]

    # 2. Simulate the data for this example
    d = pd.DataFrame(
        {"A": np.random.normal(0, 1, 100), "M": np.random.normal(0, 1, 100)}
    )
    d["M"] = 1.2 + 0.5 * d["A"] + np.random.normal(0, 1, 100)

    # 3. Call the new quap_like function
    try:
        results = quap_like(model_spec, d)
        print("MAP estimates from quap_like function:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")
