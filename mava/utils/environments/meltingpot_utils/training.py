import launchpad as lp

from mava.utils import lp_utils


class SubstrateTraining:
    def __init__(self, substrate_system_creator) -> None:
        self._system = substrate_system_creator()
        self._program = self._system.build()
        self._local_resources = None

    def setup_local_resources(self):
        # Ensure only trainer runs on gpu, while other processes run on cpu.
        self._local_resources = lp_utils.to_device(
            program_nodes=self._program.groups.keys(), nodes_on_gpu=["trainer"]
        )

    def run(self):
        self.setup_local_resources()

        import pdb

        pdb.set_trace()
        # Launch.
        lp.launch(
            self._program,
            lp.LaunchType.LOCAL_MULTI_PROCESSING,
            terminal="current_terminal",
            local_resources=self._local_resources,
        )
