
class TrainerStatisticsWrapper:
    def __init__(
        self,
        trainer: mava.Trainer,
    ) -> None:

    def step(self) -> None:
        # Run the learning step.
        fetches = self._step()
        if self._require_loggers:
            self._create_loggers(list(fetches.keys()))
            self._require_loggers = False

        # compute statistics
        self._compute_statistics(fetches)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp: float = timestamp

        # Update our counts and record it.
        self._variable_client.add_async(
            ["trainer_steps", "trainer_walltime"],
            {"trainer_steps": 1, "trainer_walltime": elapsed_time},
        )

        # Set and get the latest variables
        self._variable_client.set_and_get_async()

        fetches.update(self._counts)

        if self._logger:
            self._logger.write(fetches)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        return self._trainer.get_variables(names)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying trainer."""
        return getattr(self._trainer, name)