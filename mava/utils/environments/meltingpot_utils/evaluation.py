from mava.utils.environments.meltingpot_utils.registry import (
    get_evaluator_creator,
    get_focal_networks_setter,
)


class ScenarioEvaluation:
    def __init__(self, system_name, scenario_system_creator, trained_networks) -> None:
        self.system = scenario_system_creator()
        self.evaluator = get_evaluator_creator(system_name)(self.system)
        self._trained_networks = trained_networks
        get_focal_networks_setter(system_name)(self.evaluator, trained_networks)

    def run(self):
        self.evaluator.run()
