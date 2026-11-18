"""AdamW的Lookahead慢权重同步，与全部模型共同使用。"""
import torch


class Lookahead:
    def __init__(self, optimizer, k=10, alpha=0.5):
        if k < 1 or not 0 < alpha <= 1:
            raise ValueError("Lookahead要求k>=1且0<alpha<=1")
        self.optimizer, self.k, self.alpha = optimizer, k, alpha
        self.parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        self.slow_weights = [parameter.detach().clone() for parameter in self.parameters]
        self.steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.k == 0:
            for parameter, slow in zip(self.parameters, self.slow_weights):
                slow.add_(parameter - slow, alpha=self.alpha)
                parameter.copy_(slow)

    def state_dict(self):
        return {"base_optimizer": self.optimizer.state_dict(), "slow_weights": self.slow_weights,
                "steps": self.steps, "k": self.k, "alpha": self.alpha}
