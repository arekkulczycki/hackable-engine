from abc import ABC

from torch import nn

from hackable_engine.training.models import BaseModule
from torch.distributions import Categorical


class ActorLogitMixin(BaseModule, ABC):

    def get_action(self, obs):
        logits = self(obs)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = nn.functional.log_softmax(logits, dim=1)
        return action, log_prob, action_probs, logits
