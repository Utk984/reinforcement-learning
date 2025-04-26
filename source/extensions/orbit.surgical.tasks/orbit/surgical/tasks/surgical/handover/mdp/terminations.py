def modify_termination_param(env, term_name: str, param_name: str, value):
    term_cfg = env.termination_cfg[term_name]
    term_cfg.params[param_name] = value

