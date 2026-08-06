"""Weights & Biases experiment logger for PPO training."""

import wandb


class Logger:
    """Logs scalar metrics to Weights & Biases.

    Logging is optional: pass project=None or omit wandb_token (e.g. no token
    file present) to run console-only.
    """

    def __init__(self, project: str | None, wandb_token: str | None = None,
                 hparams: dict | None = None, tags: list[str] | None = None,
                 run_name: str | None = None):
        if project is not None and wandb_token is not None:
            wandb.login(key=wandb_token)
            self.wb_run = wandb.init(
                project=project.split("/")[-1] if "/" in project else project,
                name=run_name,
                config=hparams or {},
                tags=tags,
            )
        else:
            self.wb_run = None

    def log(self, step: int, metrics: dict):
        """Log scalar metrics at the given training iteration."""
        if self.wb_run is not None:
            self.wb_run.log({k: float(v) for k, v in metrics.items()}, step=step)

    def log_eval(self, step: int, wins: int, losses: int, draws: int, total: int):
        """Log evaluation win/draw rates."""
        wr = wins / max(total, 1)
        dr = draws / max(total, 1)
        self.log(step, {"eval/win_rate": wr, "eval/draw_rate": dr})

    def finish(self):
        """Stop the W&B run."""
        if self.wb_run is not None:
            self.wb_run.finish()
