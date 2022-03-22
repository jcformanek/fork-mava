from mava.specs import MAEnvironmentSpec
import numpy as np

def get_schema(environment_spec: MAEnvironmentSpec):
    agent_specs = environment_spec.get_agent_specs()

    schema = {}
    for agent in environment_spec.get_agent_ids():
        spec = agent_specs[agent]

        schema[agent + "observations"] = spec.observations.observation
        schema[agent + "legal_actions"] = spec.observations.legal_actions
        schema[agent + "actions"] = spec.actions
        schema[agent + "rewards"] = spec.rewards
        schema[agent + "discounts"] = spec.discounts
    
    ## Extras
    # Zero-padding mask
    schema["zero_padding_mask"] = np.array(1, dtype=np.float32)

    # Global env state
    extras_spec = environment_spec.get_extra_specs()
    if "s_t" in extras_spec:
        schema["env_state"] = extras_spec["s_t"]

    return schema
