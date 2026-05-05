from collections import deque

import config
import doy
import paths
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import matplotlib.pyplot as plt
from torch.distributions import Categorical

import env_utils
from ppo import create_buffer


class EvalWorldModel:
    def __init__(self, state_dict_policy, state_dict_idm, buflen=3, imagine_len=3, device='cuda'):
        cfg  = config.get(base_cfg=state_dict_policy["cfg"], reload_keys=["stage2", "stage3"])
        cfg.stage_exp_name += doy.random_proquint(1)
        config.print_cfg(cfg)
        self.policy = self.load_decoded_policy(cfg, state_dict_policy)
        self.idm, self.fdm = self.load_idm_fdm(cfg, state_dict_idm)

        self.buflen = buflen
        self.imagine_len = imagine_len
        self.buf_obs = deque(maxlen=buflen)
        self.buf_imagine = deque(maxlen=imagine_len)
        self.prev_fdm_next_state = None

        self.device = device

        self.policy.eval()
        self.idm.eval()
        self.fdm.eval()

    def reset_buffers(self):
        self.buf_obs = deque(maxlen=self.buflen)
        self.buf_imagine = deque(maxlen=self.imagine_len)

    def load_decoded_policy(self, cfg, state_dict):
    

        policy = utils.create_policy(
            cfg.model,
            action_dim=cfg.model.la_dim,
        )

        policy.decoder = utils.create_decoder(
            in_dim=cfg.model.la_dim,
            out_dim=cfg.model.ta_dim,
            hidden_sizes=(192, 128, 64),
        )

        ## ---- hacky stuff to monkey patch policy arch ----
        policy.policy_head_sl = policy.policy_head
        policy.policy_head_rl = nn.Linear(
            policy.policy_head_sl.in_features, cfg.model.ta_dim
        ).to(config.DEVICE)
        # init rl path to 0 output
        with torch.no_grad():
            policy.policy_head_rl.weight[:] = 0
            policy.policy_head_rl.bias[:] = 0

        policy.fc_rl = policy.fc
        policy.fc_sl = nn.Sequential(
            nn.Linear(policy.fc.in_features, policy.fc.out_features), nn.ReLU()
        ).to(config.DEVICE)
        policy.fc_sl[0].load_state_dict(policy.fc.state_dict())


        policy.load_state_dict(state_dict['policy'])

        return policy
    
    def load_idm_fdm(self, cfg, state_dict):
        state_dicts = torch.load(paths.get_models_path(config.get().exp_name), weights_only=False)
        cfg = config.get(base_cfg=state_dicts["cfg"], reload_keys=["stage2", "stage3"])


        idm, fdm = utils.create_dynamics_models(cfg.model, state_dicts=state_dict)

        return idm, fdm
    
    def action_selection_hook(self, next_obs: torch.Tensor, global_step: int = None, la_noise=None):#, action=None):
        with torch.no_grad():
            # sample action
            hidden_base = self.policy.conv_stack(next_obs)

            hidden_rl = F.relu(self.policy.fc_rl(hidden_base))
            hidden_sl = F.relu(self.policy.fc_sl(hidden_base))

            logits_rl = self.policy.policy_head_rl(hidden_rl)

            latent_action = self.policy.policy_head_sl(hidden_sl)

            if la_noise is not None:
                latent_action += la_noise

            logits_sl = self.policy.decoder(latent_action)
            logits = (logits_rl + logits_sl) / 2
            probs = Categorical(logits=logits)

            # action_given = action is not None

            # if not action_given:
            # action = probs.sample()
            action = torch.tanh(logits)

            self.buf_obs.append(next_obs.unsqueeze(1))
            if len(self.buf_obs) == 3:
                cat_obs = torch.cat(list(self.buf_obs), dim=1)
                fdm_out = self.fdm(cat_obs[:, :-1], latent_action)
                self.prev_fdm_next_state = fdm_out
            else:
                fdm_out = torch.zeros(size=(64, 3, 64, 64))



        return action, 0, torch.zeros(64), self.policy.value_head(hidden_rl), (latent_action, fdm_out)#(idm_out_la, fdm_out)
    
    def get_fdm_pred(self, latent_action):
        cat_obs = torch.cat(list(self.buf_obs), dim=1)

        fdm_out = self.fdm(cat_obs[:, :-1], latent_action)
        self.prev_fdm_next_state = fdm_out
        return fdm_out

    def imagine(self, n=None, buf_obs=None, gif_buffer=None):

        with torch.no_grad():
            if n is None: n = self.imagine_len

            frames = []
            if buf_obs is None:
                buf_temp = self.buf_obs.copy()
            else:
                if type(buf_obs) is type(deque()):
                    buf_temp = buf_obs
                else:
                    # Assumes buf_obs is a tensor with size (B T C H W)
                    buf_temp = deque(maxlen=self.buflen)
                    for i in range(buf_obs.size()[1]):
                        buf_temp.append(buf_obs[:, i].unsqueeze(1))
                

            for i in range(n):
                cat_obs = torch.cat(list(buf_temp), dim=1)
                hidden_base = self.policy.conv_stack(buf_temp[-1].squeeze(1))

                hidden_rl = F.relu(self.policy.fc_rl(hidden_base))
                hidden_sl = F.relu(self.policy.fc_sl(hidden_base))
                latent_action = self.policy.policy_head_sl(hidden_sl)
                fdm_out = self.fdm(cat_obs[:, :-1], latent_action)
                # fdm_out = self.fdm(cat_obs[:, 1:], latent_action)
                buf_temp.append(fdm_out.unsqueeze(1))
                frames.append(fdm_out.unsqueeze(1).cpu())

                if gif_buffer:
                    figure = plt.figure()
                    plt.imshow(fdm_out[0].permute(1, 2, 0).detach().cpu() + .5)
                    gif_buffer.add_to_buffer(figure)
                    plt.close(figure)
        
        # buffer.save_as_gif('imagine_hard-none.gif', duration=1)
        self.buf_imagine.append(frames)
        return frames
    

    def get_latent(self, obs):
        '''
        Get latent state from IMPALA network for obs
        '''
        with torch.no_grad():
            latent = self.policy.fc(self.policy.conv_stack(obs))
        return latent

    def get_latent_action(self, next_obs):
        with torch.no_grad():
            # sample action
            hidden_base = self.policy.conv_stack(next_obs)

            hidden_rl = F.relu(self.policy.fc_rl(hidden_base))
            hidden_sl = F.relu(self.policy.fc_sl(hidden_base))

            logits_rl = self.policy.policy_head_rl(hidden_rl)

            latent_action = self.policy.policy_head_sl(hidden_sl)

        return latent_action        

    def predict_next_state(self, obs, return_as='latent'):
        _, _, _, _, (_, fdm_out) = self.action_selection_hook(obs)
        fdm_out = fdm_out.to('cuda')

        if return_as == 'latent':
            return self.policy.fc(self.policy.conv_stack(fdm_out)) # Get state embedding of predicted state
        else: return fdm_out



class EvalEnv():
    def __init__(self, cfg, device='cuda'):

        self.cfg = cfg
        self.rl_cfg = cfg.stage3
        self.device = device

        self.envs = env_utils.setup_procgen_env(
            # num_envs=cfg.stage3.num_envs,
            num_envs=64,
            env_id=cfg.env_name,
            gamma=cfg.stage3.gamma,
            distribution_mode='easy'
        )

        self.buf = create_buffer(
            self.rl_cfg.num_steps,
            self.rl_cfg.num_envs,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
        )

    # def fill_buffer(self):
    #     for step in range(0, 3):
    #         # next_obs = augment(next_obs)
    #         buf['obs'][step] = next_obs
    #         buf['dones'][step] = next_done
    #         action, logprobs, _, value, (logits, fdm_out) = wm.action_selection_hook(normalize_obs(next_obs))
    #         buf["values"][step] = value.flatten()
    #         last_obs = next_obs
    #         next_obs, reward, done, info = envs.step(action.cpu().numpy())
    #         next_obs = torch.from_numpy(next_obs).permute((0, 3, 1, 2)).to(device)
    