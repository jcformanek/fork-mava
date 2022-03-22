import launchpad as lp
from mava.wrappers import DetailedPerAgentStatistics
from mava.project.components.offline import MAOfflineEnvironmentDataset
import wandb

class OfflineSystem:

    def evaluator(
        self,
        variable_source,
    ):
        """System evaluator.

        Args:
            variable_source: variable server for updating
                network variables.
            logger: logger object.

        Returns:
            environment-executor evaluation loop instance for evaluating the
                performance of a system.
        """
        # Executor with no exploration and no adder
        executor = self._build_executor(variable_source, evaluator=True) # hook

        # Extra executor setup
        executor = self._extra_executor_setup(executor, evaluator=True) # hook

        # Make the environment
        environment = self._environment_factory()  # type: ignore

        # Create logger and counter
        logger = self._logger_factory("evaluator")


        # Create the loop to connect environment and executor
        executor_environment_loop = self._eval_loop_fn(
            environment,
            executor,
            logger=logger,
        )

        # Environment loop statistics
        executor_environment_loop = DetailedPerAgentStatistics(executor_environment_loop)

        return executor_environment_loop

    def trainer(
        self,
    ):
        """System trainer.

        Args:

        Returns:
            system trainer.
        """
        environment = self._environment_factory()  # type: ignore

        # Create logger
        logger = self._logger_factory("trainer")

        # Build offline dataset
        dataset = MAOfflineEnvironmentDataset(
            environment=environment,
            logdir=self._offline_env__log_dir,
            batch_size=self._batch_size,
            shuffle_buffer_size=self._shuffle_buffer_size
        )

        # Make the trainer
        trainer = self._build_trainer(dataset, logger) # hook

        # Possibly do extra trainer setup
        trainer = self._extra_trainer_setup(trainer) # hook

        return trainer

    def build(self, name: str = "offline idqn"):
        """Build the distributed system as a graph program.

        Args:
            name: system name.

        Returns:
            graph program for distributed system training.
        """
        program = lp.Program(name=name)

        with program.group("trainer"):
            # Add trainer
            trainer = program.add_node(
                lp.CourierNode(self.trainer)
            )

        with program.group("evaluator"):
            # Add evaluator
            program.add_node(lp.CourierNode(self.evaluator, variable_source=trainer))

        return program

    def run_single_proc_system(self, evaluator_period=5, max_trainer_steps=1_000_000):

        trainer = self.trainer()

        evaluator = self.evaluator(trainer)

        trainer_steps = 0
        while trainer_steps < max_trainer_steps:

            trainer_stats = trainer.step() # logging done in trainer

            if trainer_steps % evaluator_period == 0:
                evaluator_stats = evaluator.run_episode()
                evaluator._logger.write(evaluator_stats) # logging

                if self._wandb:
                    # Add all logs
                    all_logs = {}
                    # Add evaluator logs
                    for key, stat in evaluator_stats.items():
                        all_logs["evaluator_"+key] = stat
                    # Add trainer logs
                    all_logs.update(trainer_stats)

                    # Wandb logging
                    wandb.log(
                        all_logs
                    )

            trainer_steps += 1

        return