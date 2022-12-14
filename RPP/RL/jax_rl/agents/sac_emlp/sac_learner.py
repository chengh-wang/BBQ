"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import functools

from jax_rl.agents.actor_critic_temp import ActorCriticTemp
from jax_rl.agents.sac_emlp import actor, critic, temperature
from jax_rl.datasets import Batch
from jax_rl.networks import critic_net, policies
from jax_rl.networks.common import InfoDict, Model
from emlp.reps import Rep
from emlp.groups import Group

@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _update_jit(sac: ActorCriticTemp, batch: Batch, discount: float,
                tau: float, target_entropy: float,
                update_target: bool,
                 actor_wd: float = 1e-5, critic_wd: float = 1e-5
               ) -> Tuple[ActorCriticTemp, InfoDict]:

    sac, critic_info = critic.update(sac, batch, discount, soft_critic=True)
    # sac, critic_info = critic.update(sac, batch, discount, True,critic_basic_wd,critic_equiv_wd)
    if update_target:
        sac = critic.target_update(sac, tau)

    sac, actor_info = actor.update(sac, batch, actor_wd)
    sac, alpha_info = temperature.update(sac, actor_info['entropy'],
                                         target_entropy)

    return sac, {**critic_info, **actor_info, **alpha_info}


class SACEMLPLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 actor_wd: float = 0.,
                 
                 critic_lr: float = 3e-4,
                 critic_wd: float = 0.,

                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0,
                 symmetry_group: Optional[Group] = None,
                 state_rep: Optional[Rep] = None,
                 action_rep: Optional[Rep] = None,
                 action_std_rep: Optional[Rep] = None,
                 state_transform=lambda x:x,
                 inv_state_transform=lambda x:x,
                 action_transform=lambda x:x,
                 inv_action_transform=lambda x:x,
                 action_space="continuous",
                 rpp_value=False):

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        
        self.actor_wd = actor_wd
        self.critic_wd = critic_wd
        
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if symmetry_group is not None: # Use RPP-EMLP policy
            if action_space == "discrete":
                actor_def = policies.EMLPSoftmaxPolicy(state_rep,action_rep,symmetry_group,hidden_dims,
                    state_transform=state_transform,inv_action_transform=inv_action_transform)
            else:
                actor_def = policies.EMLPNormalTanhPolicy(state_rep,action_rep,action_std_rep,symmetry_group,hidden_dims,
                    state_transform=state_transform,inv_action_transform=inv_action_transform)
        else:
            if action_space == "discrete":
                actor_def = policies.SoftmaxPolicy(hidden_dims, action_dim)
            else:
                actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))
        if rpp_value:
            critic_def = critic_net.RPPDoubleCritic(state_rep,action_rep,symmetry_group,hidden_dims,
                        state_transform=state_transform,inv_action_transform=inv_action_transform)
        elif action_space=='discrete':
            critic_def = critic_net.DoubleDiscreteCritic(hidden_dims)
        else:
            critic_def = critic_net.DoubleCritic(hidden_dims)
            
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.sac = ActorCriticTemp(actor=actor,
                                   critic=critic,
                                   target_critic=target_critic,
                                   temp=temp,
                                   rng=rng)
        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.sac.rng,
                                               self.sac.actor.apply_fn,
                                               self.sac.actor.params,
                                               observations, temperature)
        self.sac = self.sac.replace(rng=rng)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        self.sac, info = _update_jit(
            self.sac, batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0,
            self.actor_wd, self.critic_wd)
        return info
