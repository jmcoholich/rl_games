from os import stat_result
import matplotlib.pyplot as plt
import numpy as np
import torch

from tasks.aliengo_utils.utils import batch_z_2D_rot_mat

PLOT = False
NEW_THING = False
PLOT_AFTER = 100

class ValueProcesser:
    def __init__(self, player, des_dir=0.0, des_dir_coef=0.0,
                 start_after_n_steps=60, file_prefix="value_search", box_len=0.15,
                 grid_points=13, random_footsteps=False):
        """
        Update: This code is now vectorized across environments. The first
        dimension is always the environment dimension.
        """
        # grid_points = 4  # this is for search 8D
        self.random_footsteps = random_footsteps
        self.player = player
        self.task = self.player.env.task
        self.num_envs = self.task.num_envs
        self.file_prefix = file_prefix
        self.start_idx = self.task.observe.get_footstep_obs_start_idx()
        self.device = self.task.device
        self.pi = 3.14159265358979
        self.fg = self.task.footstep_generator

        self.env_arange = torch.arange(self.num_envs, device=self.device)

        self.box_len = box_len  # this is actually half of box len
        self.grid_points = grid_points
        self.grid_points = 17
        self.save_video_frames = False
        self.make_plots = False

        self.normalize_heatmap_scale = False
        self.optim_targets = True
        self.start_after_n_steps = start_after_n_steps
        self.num_plot_rows = 5
        self.optmize_current_step = True
        # heading in radians
        self.des_dir = torch.tensor([des_dir * self.pi], device=self.device)
        self.des_dir_weight = des_dir_coef

        self.max_pixel = -float('inf')
        self.min_pixel = float('inf')
        self.max_pixel = None
        self.min_pixel = None

        self.foot_names = ["FL", "FR", "RL", "RR"]

        if self.make_plots and (self.grid_points - 1) % (self.num_plot_rows - 1) != 0:
            raise ValueError("Number of grid points incompatible with "
                             "number of plots")

        self.grid = torch.linspace(-self.box_len, self.box_len,
                                   self.grid_points, device=self.device)

        if self.task.is_stepping_stones:
            self.stone_pos = self.task.stepping_stones.stone_pos[0]

    def __call__(self, obs):
        # if self.env.task.progress_buf[0] % 500000 != 0:
        #     return
        if self.task.progress_buf[0] == 0:
            return obs
        # assert obs.shape[0] == 1

        self.obs_len = obs.shape[1]

        if self.task.progress_buf[0] >= self.start_after_n_steps:
            # if self.task.is_stepping_stones:
            #     return self.search_stepping_stones(obs)
            # return self.search_4D(obs)
            return self.gd_8D(obs)
            # return self.search_8D(obs)
        else:
            return obs
            # self.four_feet_optim(obs)
            # self.diag_feet_optim(obs)

            # if self.optim_targets:
            #     raise NotImplementedError
            #     footstep_obs = obs[0, self.start_idx:self.start_idx + 12]
            #     self.player.set_new_footstep(footstep_obs)

    # def generate_ss_test_obs(self, obs):
    #     # search for footsteps in a radius for the first foot
    #     foot_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep)[0]
    #     xy_dists = (self.stone_pos
    #                 - self.fg.footsteps[0, self.fg.current_footstep[0] - 2, 0])[:, :2].norm(dim=1)
    #     idcs0 = xy_dists < self.ss_search_r

    #     # search for footsteps in a radius for the second foot
    #     xy_dists = (self.stone_pos
    #                 - self.fg.footsteps[
    #                     0, self.fg.current_footstep[0] - 2, 1])[:, :2].norm(dim=1)
    #     idcs1 = xy_dists < self.ss_search_r

    #     test_obs = obs[0].tile(idcs0.count_nonzero(), idcs1.count_nonzero(), 1)

    #     # set observations to the relavent positions according to global
    #     # coordinate frame

    #     # foot 0
    #     start0 = self.start_idx + 3 * foot_idcs[0]
    #     end0 = self.start_idx + 3 * foot_idcs[0] + 2
    #     # ss position - foot current position
    #     rel_pos = (self.stone_pos[idcs0][:, :2]
    #                - self.task.foot_center_pos[0, foot_idcs[0], :2])
    #     test_obs[:, :, start0: end0] = rel_pos.view(idcs0.count_nonzero(), 1, 2)

    #     # foot 1
    #     start1 = self.start_idx + 3 * foot_idcs[1]
    #     end1 = self.start_idx + 3 * foot_idcs[1] + 2
    #     # ss position - foot current position
    #     rel_pos = (self.stone_pos[idcs1][:, :2]
    #                - self.task.foot_center_pos[0, foot_idcs[1], :2])
    #     test_obs[:, :, start1: end1] = rel_pos.view(1, idcs1.count_nonzero(), 2)

    #     # rotate to align with robot yaw
    #     yaw = self.task.base_euler[0, 2]
    #     rot_mat = batch_z_2D_rot_mat(-yaw)
    #     test_obs[:, :, start0: end0 + 1] = \
    #         (rot_mat @ test_obs[:, :, start0: end0 + 1].unsqueeze(-1)).squeeze(-1)
    #     test_obs[:, :, start1: end1 + 1] = \
    #         (rot_mat @ test_obs[:, :, start1: end1 + 1].unsqueeze(-1)).squeeze(-1)

    #     return test_obs, idcs0, idcs1


    # def search_stepping_stones(self, obs):
    #     output_obs = obs.clone()
    #     # ss_idcs0 and ss_idcs1 are the boolean idcs of the footsteps within a
    #     # radius of foot 0 and 1 respectively
    #     test_obs, ss_idcs0, ss_idcs1 = self.generate_ss_test_obs(obs)

    #     len0 = ss_idcs0.count_nonzero()
    #     len1 = ss_idcs1.count_nonzero()
    #     values = self.get_values(test_obs.view(len0 * len1, self.obs_len))
    #     values = values.view(len0, len1)

    #     if self.optim_targets:
    #         values = self.adjust_ss_values(values, ss_idcs0, ss_idcs1)

    #         optimal_targets, max_idx = self.get_optimal_targets_from_ss(
    #             values, ss_idcs0, ss_idcs1)

    #         self.fg.footsteps[0, self.fg.current_footstep[0]] = optimal_targets
    #         self.fg.plot_footstep_targets(current_only=True)
    #         if self.optmize_current_step:
    #             output_obs = self.task.observe(recalculating_obs=True)

    #     # if self.make_plots or self.save_video_frames:
    #     #     self.plot_ss_values(values, max_idx, ss_idcs0)
    #     return output_obs

    # def adjust_ss_values(self, values, ss_idcs0, ss_idcs1):
    #     x_coef = self.des_dir.cos() * self.des_dir_weight
    #     y_coef = self.des_dir.sin() * self.des_dir_weight

    #     values[:, :] += self.stone_pos[ss_idcs0][:, 0].view(ss_idcs0.count_nonzero(), 1) * x_coef
    #     values[:, :] += self.stone_pos[ss_idcs0][:, 1].view(ss_idcs0.count_nonzero(), 1) * y_coef

    #     values[:, :] += self.stone_pos[ss_idcs1][:, 0].view(1, ss_idcs1.count_nonzero()) * x_coef
    #     values[:, :] += self.stone_pos[ss_idcs1][:, 1].view(1, ss_idcs1.count_nonzero()) * y_coef
    #     return values

    # def plot_ss_values(self, values, max_idx, ss_idcs0):
    #     """Plots the values of the footsteps in the radius given the
    #     best target for the other foot is chosen.
    #     The values of the stepping stones are indicated by color and height.
    #     The chosen footstep is extra thicc.
    #     """
    #     pass

    # def get_optimal_targets_from_ss(self, values, ss_idcs0, ss_idcs1):
    #     max_idx = (values == torch.max(values)).nonzero()[0]

    #     ss0 = self.stone_pos[ss_idcs0][max_idx[0]]
    #     ss1 = self.stone_pos[ss_idcs1][max_idx[1]]

    #     optimal_targets = torch.stack((ss0, ss1))
    #     return optimal_targets, max_idx
    def gd_8D(self, obs, params=None):

        # init sgd optimizer
        obs = obs.clone()
        # params = torch.zeros(self.num_envs, 8, device=self.device, requires_grad=True)
        if params is None:
            self.params = torch.zeros(self.num_envs, 8, device=self.device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.params], lr=0.01)

        num_iters = 5
        for i in range(num_iters):
            self.optimizer.zero_grad()
            test_obs = self.generate_8D_test_obs(obs, self.params)
            loss = self.loss_8D(test_obs)
            loss.backward(retain_graph=True)
            # print("after backward", params.grad)
            self.optimizer.step()
            # print(f"iter {i}, loss {loss.item()}, params {test_obs[0, self.start_idx: self.start_idx + 12].detach().cpu().numpy()}")
            # print(params)
            # print()
            # print()
            # print()
        # print(loss)
        # breakpoint()
        # set footstep targets
        # for i in range(0, 8, 2):
        #     print("norm", self.params[:, i:i+2].norm(dim=1).cpu().item())
        # print()
        yaws = self.task.base_euler[:, 2]
        rot_mats = batch_z_2D_rot_mat(yaws)
        curr_footstep_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep)
        fut_footstep_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep + 1)

        # transform from obs space to world space
        curr_opt_tgt_foot0 = (rot_mats @ self.params[:, :2].view(self.num_envs, 2, 1)).squeeze(-1) + self.task.foot_center_pos[self.env_arange, curr_footstep_idcs[:, 0], :2]

        curr_opt_tgt_foot1 = (rot_mats @ self.params[:, 2:4].view(self.num_envs, 2, 1)).squeeze(-1) + self.task.foot_center_pos[self.env_arange, curr_footstep_idcs[:, 1], :2]

        future_opt_tgt_foot0 = (rot_mats @ self.params[:, 4:6].view(self.num_envs, 2, 1)).squeeze(-1) + self.task.foot_center_pos[self.env_arange, fut_footstep_idcs[:, 0], :2]

        future_opt_tgt_foot1 = (rot_mats @ self.params[:, 6:8].view(self.num_envs, 2, 1)).squeeze(-1) + self.task.foot_center_pos[self.env_arange, fut_footstep_idcs[:, 1], :2]


        # assign back to footstep targets
        self.fg.footsteps[self.env_arange, self.fg.current_footstep, 0] = curr_opt_tgt_foot0.detach()
        self.fg.footsteps[self.env_arange, self.fg.current_footstep, 1] = curr_opt_tgt_foot1.detach()
        self.fg.footsteps[self.env_arange, self.fg.current_footstep + 1, 0] = future_opt_tgt_foot0.detach()
        self.fg.footsteps[self.env_arange, self.fg.current_footstep + 1, 1] = future_opt_tgt_foot1.detach()

        self.fg.plot_footstep_targets(current_only=True)
        if self.optmize_current_step:
            output_obs = self.task.observe(recalculating_obs=True).clamp(-5.0, 5.0)
            # print(self.fg.get_footstep_idcs(self.fg.current_footstep))
            # print(self.params)
            # print(output_obs[:, self.start_idx: self.start_idx + 12])
            # print()
            # breakpoint()

        return output_obs

    def loss_8D(self, obs):
        values = self.get_values(obs)
        loss = -torch.mean(values)
        return loss

    def generate_8D_test_obs(self, obs, params):
        foot_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep)
        # set first footstep
        obs[self.env_arange, self.start_idx + foot_idcs[:, 0]: self.start_idx + foot_idcs[:, 0] + 2] = params[:, :2]
        # set second footstep
        obs[self.env_arange, self.start_idx + foot_idcs[:, 1]: self.start_idx + foot_idcs[:, 1] + 2] = params[:, 2:4]
        # set future footstep
        obs[:, self.start_idx + 8: self.start_idx + 12] = params[:, 4:]
        return obs

    def search_8D(self, obs):
        """
        I want to also modify the footsteps in place.
        dim 0: foot 0  x
        dim 1: foot 0  y
        dim 2: foot 1  x
        dim 3: foot 1  y

        NEXT FOOTSTEP TARGET PAIR
        dim 4: foot 0  x
        dim 5: foot 0  y
        dim 6: foot 1  x
        dim 7: foot 1  y
        """
        output_obs = obs.clone()
        test_obs = self.generate_8D_test_obs(obs)

        values = self.get_values(
            test_obs.view(self.num_envs * self.grid_points**8, self.obs_len))
        values = values.view([self.num_envs] + [self.grid_points] * 8)

        if self.optim_targets:
            values = self.adjust_8D_values(values)
            optimal_targets, optimal_next_next_targets, max_idx = self.get_optimal_targets_from_8D(values)

            # apply first order low-pass filter

            # self.fg.footsteps[self.env_arange, self.fg.current_footstep] = 0.1 * optimal_targets + 0.9 * self.fg.footsteps[self.env_arange, self.fg.current_footstep]
            self.fg.footsteps[self.env_arange, self.fg.current_footstep] = optimal_targets
            self.fg.footsteps[self.env_arange, self.fg.current_footstep + 1] = optimal_next_next_targets
            self.fg.plot_footstep_targets(current_only=True)
            if self.optmize_current_step:
                output_obs = self.task.observe(recalculating_obs=True)

        if self.make_plots or self.save_video_frames:
            self.plot_4D_values(values[0], max_idx[0])
        return output_obs

    def search_4D(self, obs):
        """
        I want to also modify the footsteps in place.
        dim 0: foot 0  x
        dim 1: foot 0  y
        dim 2: foot 1  x
        dim 3: foot 1  y
        """
        output_obs = obs.clone()
        test_obs = self.generate_4D_test_obs(obs)
        # new = test_obs.view(self.task.num_envs, -1, test_obs.shape[-1])
        # test_obs = self.generate_4D_test_obs_cart_prod(obs)
        # new2 = []
        # for batch in test_obs:
        #     new2.append(batch)
        # breakpoint()

        # test_obs_old = self.generate_4D_test_obs_old(obs, 0)

        # for i in range(self.num_envs):
        #     test_obs_old = self.generate_4D_test_obs_old(obs, i)
        #     same = (test_obs_old == test_obs[i]).all()
        #     print(same.item())
        #     assert same

        # import sys; sys.exit()


        values = self.get_values(
            test_obs.view(self.num_envs * self.grid_points**4, self.obs_len))
        values = values.view([self.num_envs] + [self.grid_points] * 4)
        print(-values.max())
        breakpoint()

        # values_old = self.get_values(
        #     test_obs_old.view(self.grid_points**4, self.obs_len))
        # values_old = values_old.view([self.grid_points] * 4)

        # same = (values == values_old).all()
        # print(same)
        # assert same
        # import sys; sys.exit()


        if self.optim_targets:
            values = self.adjust_4D_values(values)
            optimal_targets, max_idx = self.get_optimal_targets_from_4D(values)

            # for i in range(self.num_envs):
            #     old_opt_tar, old_max_idx = self.get_optimal_targets_from_4D_old(values, i)
            #     same = (optimal_targets[i] == old_opt_tar).all() and (max_idx[i] == old_max_idx).all()
            #     print(same.item())
            #     assert same

            # apply first order low-pass filter

            # self.fg.footsteps[self.env_arange, self.fg.current_footstep] = 0.1 * optimal_targets + 0.9 * self.fg.footsteps[self.env_arange, self.fg.current_footstep]
            if NEW_THING:
                yaw = batch_z_2D_rot_mat(-self.task.base_euler[:,2]).view(self.num_envs, 1, 2, 2)
                optimal_targets = (yaw @ optimal_targets.unsqueeze(-1)).squeeze(-1)
            if PLOT and self.task.progress_buf[0] % PLOT_AFTER == 0:
                plt.figure("world")
                env = 0
                plt.scatter(-optimal_targets[env, 0, 1].cpu().numpy(), optimal_targets[env, 0, 0].cpu().numpy(), facecolors='none', edgecolors='c')
                plt.scatter(-optimal_targets[env, 1, 1].cpu().numpy(), optimal_targets[env, 1, 0].cpu().numpy(), facecolors='none', edgecolors='c')
                plt.show()
            self.fg.footsteps[self.env_arange, self.fg.current_footstep] = optimal_targets
            self.fg.plot_footstep_targets(current_only=True)
            if self.optmize_current_step:
                output_obs = self.task.observe(recalculating_obs=True).clamp(-5.0, 5.0)

        if self.make_plots or self.save_video_frames:
            self.plot_4D_values(values[0], max_idx[0])
        return output_obs


    # def generate_8D_test_obs(self, obs):

    #     p = 2  # this is the obs len per footstep target. Was previously 3
    #     ea = self.env_arange
    #     n_env = self.num_envs

    #     test_obs = obs.view(n_env, 1, 1, 1, 1, 1, 1, 1, 1, self.obs_len).tile([1] + [self.grid_points] * 8 + [1])
    #     foot_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep)
    #     # NOTE next line will fail if executed before agent hits first targets
    #     prev_targets = self.fg.footsteps[ea, self.fg.current_footstep - 2]

    #     # generate grids centered around previous footstep targets aligned
    #     # with global coordinate system
    #     # TODO make sure these actually update upon assignment
    #     test_obs[ea, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[:, 0] * p] = \
    #         self.grid.view(1, self.grid_points, 1, 1, 1, 1, 1, 1, 1) \
    #         + prev_targets[:, 0, 0].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1) \
    #         - self.task.foot_center_pos[ea, foot_idcs[:, 0], 0].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1)

    #     test_obs[ea, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[:, 0] * p + 1] = \
    #         self.grid.view(1, 1, self.grid_points, 1, 1, 1, 1, 1, 1) \
    #         + prev_targets[:, 0, 1].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1) \
    #         - self.task.foot_center_pos[ea, foot_idcs[:, 0], 1].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1)

    #     test_obs[ea, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[:, 1] * p] = \
    #         self.grid.view(1, 1, 1, self.grid_points, 1, 1, 1, 1, 1) \
    #         + prev_targets[:, 1, 0].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1) \
    #         - self.task.foot_center_pos[ea, foot_idcs[:, 1], 0].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1)

    #     test_obs[ea, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[:, 1] * p + 1] = \
    #         self.grid.view(1, 1, 1, 1, self.grid_points, 1, 1, 1, 1) \
    #         + prev_targets[:, 1, 1].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1) \
    #         - self.task.foot_center_pos[ea, foot_idcs[:, 1], 1].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1)

    #     foot_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep - 1)
    #     prev_targets = self.fg.footsteps[ea, self.fg.current_footstep - 1]

    #     test_obs[ea, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[:, 0] * p] = \
    #         2 * self.grid.view(1, 1, 1, 1, 1, self.grid_points, 1, 1, 1) \
    #         + prev_targets[:, 0, 0].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1) \
    #         - self.task.foot_center_pos[ea, foot_idcs[:, 0], 0].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1)

    #     test_obs[ea, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[:, 0] * p + 1] = \
    #         2 * self.grid.view(1, 1, 1, 1, 1, 1, self.grid_points, 1, 1) \
    #         + prev_targets[:, 0, 1].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1) \
    #         - self.task.foot_center_pos[ea, foot_idcs[:, 0], 1].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1)

    #     test_obs[ea, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[:, 1] * p] = \
    #         2 * self.grid.view(1, 1, 1, 1, 1, 1, 1, self.grid_points, 1) \
    #         + prev_targets[:, 1, 0].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1) \
    #         - self.task.foot_center_pos[ea, foot_idcs[:, 1], 0].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1)

    #     test_obs[ea, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[:, 1] * p + 1] = \
    #         2 * self.grid.view(1, 1, 1, 1, 1, 1, 1, 1, self.grid_points) \
    #         + prev_targets[:, 1, 1].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1) \
    #         - self.task.foot_center_pos[ea, foot_idcs[:, 1], 1].view(n_env, 1, 1, 1, 1, 1, 1, 1, 1)

    #     # # subtract robot feet global positions from the grid
    #     # test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * p] -= \
    #     #     self.task.foot_center_pos[0, foot_idcs[0], 0]
    #     # test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * p + 1] -= \
    #     #     self.task.foot_center_pos[0, foot_idcs[0], 1]
    #     # test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * p] -= \
    #     #     self.task.foot_center_pos[0, foot_idcs[1], 0]
    #     # test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * p + 1] -= \
    #     #     self.task.foot_center_pos[0, foot_idcs[1], 1]

    #     # rotate to align with robot yaw
    #     yaw = self.task.base_euler[:, 2]
    #     rot_mat = batch_z_2D_rot_mat(-yaw).view(n_env, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2)
    #     ea = ea.unsqueeze(-1)

    #     for i in range(self.num_envs):
    #         rot_mat = batch_z_2D_rot_mat(-yaw[i])
    #         test_obs[i, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[i, 0] * p: self.start_idx + foot_idcs[i, 0] * p + p] = \
    #             (rot_mat @ test_obs[i, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[i, 0] * p: self.start_idx + foot_idcs[i, 0] * p + p].unsqueeze(-1)).squeeze(-1)
    #         test_obs[i, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[i, 1] * p: self.start_idx + foot_idcs[i, 1] * p + p] = \
    #             (rot_mat @ test_obs[i, :, :, :, :, :, :, :, :, self.start_idx + foot_idcs[i, 1] * p: self.start_idx + foot_idcs[i, 1] * p + p].unsqueeze(-1)).squeeze(-1)

    #     return test_obs

    def generate_4D_test_obs_cart_prod(self, obs):
        # I don't expect to be doing this for many envs at a time, so I will just loop over envs
        # unfortunately there is no batched version of cartesian_prod, so I'm using a for loop

        # TODO for now just return the same thing as generate_4D_test_obs. After testing this, switch to returning a memory-efficient iterator
        output = []
        for i in range(self.num_envs):
            orig_obs = obs[i].clone()
            orig_footstep_obs = orig_obs[self.start_idx:self.start_idx + 8]
            foot_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep)[i]
            prev_targets = self.fg.footsteps[i, self.fg.current_footstep[i] - 2]
            temp = torch.cartesian_prod(
                self.grid + prev_targets[0, 0] - self.task.foot_center_pos[i, foot_idcs[0], 0],  # foot 0  x
                self.grid + prev_targets[0, 1] - self.task.foot_center_pos[i, foot_idcs[0], 1],  # foot 0  y
                self.grid + prev_targets[1, 0] - self.task.foot_center_pos[i, foot_idcs[1], 0],  # foot 1  x
                self.grid + prev_targets[1, 1] - self.task.foot_center_pos[i, foot_idcs[1], 1],  # foot 1  y
                )
            yaw = self.task.base_euler[i, 2]
            rot_mat = batch_z_2D_rot_mat(-yaw)
            temp[:, 0:2] = (rot_mat @ temp[:, 0:2].unsqueeze(-1)).squeeze(-1)
            temp[:, 2:4] = (rot_mat @ temp[:, 2:4].unsqueeze(-1)).squeeze(-1)
            footstep_obs = torch.zeros(temp.shape[0], 8, device=self.device)
            if foot_idcs[0] == 1:  # idcs are [1, 2] These are the idcs being optimized over
                assert foot_idcs[1] == 2
                footstep_obs[:, 0] = orig_footstep_obs[0]
                footstep_obs[:, 1] = orig_footstep_obs[1]
                footstep_obs[:, 2] = temp[:, 0]
                footstep_obs[:, 3] = temp[:, 1]
                footstep_obs[:, 4] = temp[:, 2]
                footstep_obs[:, 5] = temp[:, 3]
                footstep_obs[:, 6] = orig_footstep_obs[6]
                footstep_obs[:, 7] = orig_footstep_obs[7]
            elif foot_idcs[0] == 0:
                assert foot_idcs[1] == 3
                footstep_obs[:, 0] = temp[:, 0]
                footstep_obs[:, 1] = temp[:, 1]
                footstep_obs[:, 2] = orig_footstep_obs[2]
                footstep_obs[:, 3] = orig_footstep_obs[3]
                footstep_obs[:, 4] = orig_footstep_obs[4]
                footstep_obs[:, 5] = orig_footstep_obs[5]
                footstep_obs[:, 6] = temp[:, 2]
                footstep_obs[:, 7] = temp[:, 3]
            else:
                raise ValueError
            obs = torch.zeros(temp.shape[0], orig_obs.shape[0], device=self.device)
            obs[:] = orig_obs
            obs[:, self.start_idx:self.start_idx + 8] = footstep_obs
            output.append(obs)
            breakpoint()
        return MyIterator(torch.stack(output))

    def generate_4D_test_obs(self, obs):
        """
        dim 0: foot 0  x
        dim 1: foot 0  y
        dim 2: foot 1  x
        dim 3: foot 1  y
        """
        p = 2  # this is the obs len per footstep target. Was previously 3
        ea = self.env_arange
        n_env = self.num_envs

        foot_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep)
        # NOTE next line will fail if executed before agent hits first targets
        prev_targets = self.fg.footsteps[ea, self.fg.current_footstep - 2]
        test_obs = obs.view(n_env, 1, 1, 1, 1, self.obs_len).tile([1] + [self.grid_points] * 4 + [1])

        # generate grids centered around previous footstep targets aligned
        # with global coordinate system
        # TODO make sure these actually update upon assignment
        test_obs[ea, :, :, :, :, self.start_idx + foot_idcs[:, 0] * p] = \
            self.grid.view(1, self.grid_points, 1, 1, 1) \
            + prev_targets[:, 0, 0].view(n_env, 1, 1, 1, 1) \
            - self.task.foot_center_pos[ea, foot_idcs[:, 0], 0].view(n_env, 1, 1, 1, 1)

        test_obs[ea, :, :, :, :, self.start_idx + foot_idcs[:, 0] * p + 1] = \
            self.grid.view(1, 1, self.grid_points, 1, 1) \
            + prev_targets[:, 0, 1].view(n_env, 1, 1, 1, 1) \
            - self.task.foot_center_pos[ea, foot_idcs[:, 0], 1].view(n_env, 1, 1, 1, 1)

        test_obs[ea, :, :, :, :, self.start_idx + foot_idcs[:, 1] * p] = \
            self.grid.view(1, 1, 1, self.grid_points, 1) \
            + prev_targets[:, 1, 0].view(n_env, 1, 1, 1, 1) \
            - self.task.foot_center_pos[ea, foot_idcs[:, 1], 0].view(n_env, 1, 1, 1, 1)

        test_obs[ea, :, :, :, :, self.start_idx + foot_idcs[:, 1] * p + 1] = \
            self.grid.view(1, 1, 1, 1, self.grid_points) \
            + prev_targets[:, 1, 1].view(n_env, 1, 1, 1, 1) \
            - self.task.foot_center_pos[ea, foot_idcs[:, 1], 1].view(n_env, 1, 1, 1, 1)

        if PLOT and self.task.progress_buf[0] % PLOT_AFTER == 0:


            import matplotlib.pyplot as plt
            # global WORLD_FIG
            # WORLD_FIG = plt.figure()
            plt.figure("world")
            plt.title("World frame (x is up)")
            env = 0
            test_obs_plt = test_obs[env].clone()
            if NEW_THING:
                # rotate the grid points to align with robot yaw
                yaw = self.task.base_euler[env, 2]
                rot_mat = batch_z_2D_rot_mat(yaw)
                test_obs_plt[ :, :, :, :, self.start_idx + foot_idcs[env, 0] * p: self.start_idx + foot_idcs[env, 0] * p + 2] = \
                    (rot_mat @ test_obs_plt[ :, :, :, :, self.start_idx + foot_idcs[env, 0] * p: self.start_idx + foot_idcs[env, 0] * p + 2].unsqueeze(-1)).squeeze(-1)
                test_obs_plt[ :, :, :, :, self.start_idx + foot_idcs[env, 1] * p: self.start_idx + foot_idcs[env, 1] * p + 2] = \
                    (rot_mat @ test_obs_plt[ :, :, :, :, self.start_idx + foot_idcs[env, 1] * p: self.start_idx + foot_idcs[env, 1] * p + 2].unsqueeze(-1)).squeeze(-1)
            test_obs_plt = test_obs_plt.cpu().numpy()

            prev_tgt_x_foot_0 = self.fg.footsteps[env, self.fg.current_footstep[env] - 2, 0, 0].cpu().numpy()
            prev_tgt_x_foot_1 = self.fg.footsteps[env, self.fg.current_footstep[env] - 2, 1, 0].cpu().numpy()
            prev_tgt_y_foot_0 = self.fg.footsteps[env, self.fg.current_footstep[env] - 2, 0, 1].cpu().numpy()
            prev_tgt_y_foot_1 = self.fg.footsteps[env, self.fg.current_footstep[env] - 2, 1, 1].cpu().numpy()
            curr_pos_x_foot_0 = self.task.foot_center_pos[env, foot_idcs[env, 0], 0].cpu().numpy()
            curr_pos_x_foot_1 = self.task.foot_center_pos[env, foot_idcs[env, 1], 0].cpu().numpy()
            curr_pos_y_foot_0 = self.task.foot_center_pos[env, foot_idcs[env, 0], 1].cpu().numpy()
            curr_pos_y_foot_1 = self.task.foot_center_pos[env, foot_idcs[env, 1], 1].cpu().numpy()


            # plot last footstep targets (I want the axes switched so that x is "up")
            plt.scatter(-prev_tgt_y_foot_0, prev_tgt_x_foot_0, c='r')
            plt.annotate("prev_tgt", (-prev_tgt_y_foot_0, prev_tgt_x_foot_0))
            plt.scatter(-prev_tgt_y_foot_1, prev_tgt_x_foot_1, c='r')
            plt.annotate("prev_tgt", (-prev_tgt_y_foot_1, prev_tgt_x_foot_1))

            # plot curret foot positions
            plt.scatter(-curr_pos_y_foot_0, curr_pos_x_foot_0, c='g')
            plt.annotate("curr_pos", (-curr_pos_y_foot_0, curr_pos_x_foot_0))
            plt.scatter(-curr_pos_y_foot_1, curr_pos_x_foot_1, c='g')
            plt.annotate("curr_pos", (-curr_pos_y_foot_1, curr_pos_x_foot_1))

            # plot search points
            marker_size = 10
            plt.scatter(-(test_obs_plt[ :, :, :, :, self.start_idx + foot_idcs[env, 0] * p + 1].flatten() + curr_pos_y_foot_0),
                        test_obs_plt[ :, :, :, :, self.start_idx + foot_idcs[env, 0] * p].flatten() + curr_pos_x_foot_0, marker='x', s=marker_size)
            plt.scatter(-(test_obs_plt[ :, :, :, :, self.start_idx + foot_idcs[env, 1] * p + 1].flatten() + curr_pos_y_foot_1),
                        test_obs_plt[ :, :, :, :, self.start_idx + foot_idcs[env, 1] * p].flatten() + curr_pos_x_foot_1, marker='x', s=marker_size)
            # draw rectangle around search box. Just do previous footstep targets +- box_len
            plt.plot(
                [
                -(prev_tgt_y_foot_0 - self.box_len),
                -(prev_tgt_y_foot_0 - self.box_len),
                -(prev_tgt_y_foot_0 + self.box_len),
                -(prev_tgt_y_foot_0 + self.box_len),
                -(prev_tgt_y_foot_0 - self.box_len),
                ],
                [prev_tgt_x_foot_0 - self.box_len,
                prev_tgt_x_foot_0 + self.box_len,
                prev_tgt_x_foot_0 + self.box_len,
                prev_tgt_x_foot_0 - self.box_len,
                prev_tgt_x_foot_0 - self.box_len],
                c='k')
            plt.plot([
                -(prev_tgt_y_foot_1 - self.box_len),
                -(prev_tgt_y_foot_1 - self.box_len),
                -(prev_tgt_y_foot_1 + self.box_len),
                -(prev_tgt_y_foot_1 + self.box_len),
                -(prev_tgt_y_foot_1 - self.box_len),
                ],
                        [prev_tgt_x_foot_1 - self.box_len, prev_tgt_x_foot_1 + self.box_len, prev_tgt_x_foot_1 + self.box_len, prev_tgt_x_foot_1 - self.box_len, prev_tgt_x_foot_1 - self.box_len], c='k')
            # plot an arrow at the origin pointing in robot yaw direction
            arrow_len = 0.1
            plt.arrow(0, 0, -arrow_len * np.sin(self.task.base_euler[env, 2].cpu().numpy()), arrow_len * np.cos(self.task.base_euler[env, 2].cpu().numpy()), head_width=0.05, head_length=0.1, fc='k', ec='k')
            plt.axis('equal')
            plt.grid()



        # # subtract robot feet global positions from the grid
        # test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * p] -= \
        #     self.task.foot_center_pos[0, foot_idcs[0], 0]
        # test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * p + 1] -= \
        #     self.task.foot_center_pos[0, foot_idcs[0], 1]
        # test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * p] -= \
        #     self.task.foot_center_pos[0, foot_idcs[1], 0]
        # test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * p + 1] -= \
        #     self.task.foot_center_pos[0, foot_idcs[1], 1]

        # rotate to align with robot yaw
        if not NEW_THING:
            yaw = self.task.base_euler[:, 2]
            rot_mat = batch_z_2D_rot_mat(-yaw).view(n_env, 1, 1, 1, 1, 2, 2)
            ea = ea.unsqueeze(-1)

            for i in range(self.num_envs):
                rot_mat = batch_z_2D_rot_mat(-yaw[i])
                test_obs[i, :, :, :, :, self.start_idx + foot_idcs[i, 0] * p: self.start_idx + foot_idcs[i, 0] * p + p] = \
                    (rot_mat @ test_obs[i, :, :, :, :, self.start_idx + foot_idcs[i, 0] * p: self.start_idx + foot_idcs[i, 0] * p + p].unsqueeze(-1)).squeeze(-1)
                test_obs[i, :, :, :, :, self.start_idx + foot_idcs[i, 1] * p: self.start_idx + foot_idcs[i, 1] * p + p] = \
                    (rot_mat @ test_obs[i, :, :, :, :, self.start_idx + foot_idcs[i, 1] * p: self.start_idx + foot_idcs[i, 1] * p + p].unsqueeze(-1)).squeeze(-1)
        if PLOT and self.task.progress_buf[0] % PLOT_AFTER == 0:
            # plot the search points in the robot frame
            # global ROBO_FIG
            # ROBO_FIG = plt.figure()
            plt.figure("robot")
            plt.title("Robot frame (x is up), first foot search points")
            marker_size = 10
            plt.scatter(-test_obs[env, :, :, :, :, self.start_idx + foot_idcs[env, 0] * p + 1].view(-1).cpu().numpy(),
                        test_obs[env, :, :, :, :, self.start_idx + foot_idcs[env, 0] * p].view(-1).cpu().numpy(), marker='x', s=marker_size)
            # plt.scatter(test_obs[env, :, :, :, :, self.start_idx + foot_idcs[env, 1] * p + 1].view(-1).cpu().numpy(),
            #             test_obs[env, :, :, :, :, self.start_idx + foot_idcs[env, 1] * p].view(-1).cpu().numpy(), marker='x', s=marker_size)
            plt.axis('equal')
            plt.grid()
            # plt.show()
        return test_obs

    '''
    def get_optimal_targets_from_4D_old(self, values, i):
        # initialize with current targets (to keep correct z-height)
        optimal_targets = torch.zeros(2, 2, device=self.device)
        values = values[i]

        max_idx = (values == torch.max(values)).nonzero()[0]

        optimal_targets[0, 0] = self.grid2m(max_idx[0])
        optimal_targets[0, 1] = self.grid2m(max_idx[1])
        optimal_targets[1, 0] = self.grid2m(max_idx[2])
        optimal_targets[1, 1] = self.grid2m(max_idx[3])

        # now add prev targets to grid positions
        prev_targets = self.fg.footsteps[i, self.fg.current_footstep[i] - 2]
        optimal_targets[:, :2] += prev_targets[:, :2]

        return optimal_targets, max_idx
    '''

    def get_optimal_targets_from_4D(self, values):
        # initialize with current targets (to keep correct z-height)
        optimal_targets = torch.zeros(self.num_envs, 2, 2, device=self.device)

        max_idx = (values == values.amax(dim=(1, 2, 3, 4), keepdim=True)).nonzero()
        # prev line could return multiple indices equal to the max per env, so we just select the first one
        # the first colum of max_idx is the env idx

        # first row is always valid
        valid_rows = torch.ones(max_idx.shape[0], device=self.device,
                                dtype=torch.bool)
        valid_rows[1:] = max_idx[:-1, 0] != max_idx[1:, 0]

        # only take valid rows and chop off the env idx
        max_idx = max_idx[valid_rows][:, 1:]

        if self.random_footsteps:
            max_idx = torch.randint_like(max_idx, low=0, high=self.grid_points)

        optimal_targets = self.grid2m(max_idx).view(self.num_envs, 2, 2) \
            + self.fg.footsteps[self.env_arange, self.fg.current_footstep - 2]

        return optimal_targets, max_idx

    def get_optimal_targets_from_8D(self, values):
        # initialize with current targets (to keep correct z-height)
        optimal_targets = torch.zeros(self.num_envs, 2, 2, device=self.device)
        optimal_next_next_targets = torch.zeros(self.num_envs, 2, 2, device=self.device)

        max_idx = (values == values.amax(dim=(1, 2, 3, 4, 5, 6, 7, 8), keepdim=True)).nonzero()
        # prev line could return multiple indices equal to the max per env, so we just select the first one
        # the first colum of max_idx is the env idx

        # first row is always valid
        valid_rows = torch.ones(max_idx.shape[0], device=self.device,
                                dtype=torch.bool)
        valid_rows[1:] = max_idx[:-1, 0] != max_idx[1:, 0]

        # only take valid rows and chop off the env idx
        max_idx = max_idx[valid_rows][:, 1:]

        if self.random_footsteps:
            max_idx = torch.randint_like(max_idx, low=0, high=self.grid_points)
        optimal_targets = self.grid2m(max_idx[:, :4]).view(self.num_envs, 2, 2) \
            + self.fg.footsteps[self.env_arange, self.fg.current_footstep - 2]
        optimal_next_next_targets = self.grid2m(max_idx[:, 4:], grid_dialation=2.0).view(self.num_envs, 2, 2) \
            + self.fg.footsteps[self.env_arange, self.fg.current_footstep - 1]

        return optimal_targets, optimal_next_next_targets, max_idx

    def grid2m(self, idx, grid_dialation=1):
        """Treats every input as a position in the grid.
        Works with tensors and numbers.
        """
        return (idx / (self.grid_points - 1.0) - 0.5) * 2 * self.box_len * grid_dialation

    def plot_4D_values(self, values, max_idx):
        if self.normalize_heatmap_scale:
            # if this is the first call to search 4D set,
            # set min and max
            # min_pixel = values.min()
            # max_pixel = values.max()
            # if max_pixel > self.max_pixel:
            #     self.max_pixel = max_pixel
            # if min_pixel < self.min_pixel:
            #     self.min_pixel = min_pixel
            if (self.task.progress_buf[0] == self.start_after_n_steps
                    or self.task.progress_buf[0] == 1):
                min_pixel = values.min()
                max_pixel = values.max()
                margin = (max_pixel - min_pixel) * 1.0
                if self.max_pixel is None:
                    self.max_pixel = max_pixel + margin
                if self.min_pixel is None:
                    self.min_pixel = min_pixel
            vmin = self.min_pixel
            vmax = self.max_pixel
        else:
            vmin = None
            vmax = None

        foot_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep)[0]
        skip = self.grid_points // (self.num_plot_rows - 1)
        values[max(0, max_idx[0] - skip // 2): max_idx[0] + skip // 2,
               max(0, max_idx[1] - skip // 2): max_idx[1] + skip // 2,
               max(0, max_idx[2] - 1): max_idx[2] + 2,
               max(0, max_idx[3] - 1): max_idx[3] + 2] = float('nan')
        _values = values.cpu().numpy()
        plt.figure(num=2, figsize=[12.6, 10.8])
        plt.suptitle(f"Value plots for {self.foot_names[foot_idcs[1]]} "
                     f"footstep targets in [{-self.box_len}, {self.box_len}] box",
                     fontsize=16)
        for i in range(0, self.grid_points, skip):
            for j in range(0, self.grid_points, skip):
                plt.subplot(self.num_plot_rows,
                            self.num_plot_rows,
                            i//skip * self.num_plot_rows + j//skip + 1)
                # I need to switch x and y to align with the robot and
                # reverse both axes
                x_idx = self.grid_points - i - 1
                y_idx = self.grid_points - j - 1
                cmap = plt.cm.get_cmap("viridis").copy()
                cmap.set_bad((1, 0, 0, 1))
                plt.imshow(np.flip(_values[x_idx, y_idx], (0, 1)),
                           cmap=cmap, interpolation='nearest', vmin=vmin,
                           vmax=vmax)
                tick_locations = torch.arange(
                    0, self.grid_points + 1, self.grid_points // 4).numpy()
                plt.axis('off')
                num_ticks = len(tick_locations)
                # plt.xticks(tick_locations, np.around(torch.linspace(-self.box_len, self.box_len, num_ticks).numpy(), decimals=2))
                # plt.yticks(tick_locations, np.around(torch.linspace(self.box_len, -self.box_len, num_ticks).numpy(), decimals=2))

                plt.title(
                    f"{self.foot_names[foot_idcs[0]]}: "
                    f"({self.grid2m(x_idx): 0.2f},"
                    f"{self.grid2m(y_idx): 0.2f})",
                    fontsize=10)
                plt.colorbar()
                plt.grid()

        if self.make_plots:
            plt.show()
        if self.save_video_frames:
            plt.savefig(f'test_imgs/{self.file_prefix}-{self.task.progress_buf[0]}.png')


    def adjust_4D_values(self, values):
        """Adds other optimzation terms (direction) to 4D values.

        dim 0: foot 0  x
        dim 1: foot 0  y
        dim 2: foot 1  x
        dim 3: foot 1  y
        """
        # self.des_dir = torch.tensor(
        #     [self.pi * (self.task.progress_buf[0] % 4) / 2.0
        #      + self.pi / 4.0],
        #     device=self.device)
        # values = torch.zeros([self.grid_points] * 4, device=self.device)
        # values = torch.rand([self.grid_points] * 4, device=self.device)

        # add directional coefficients
        # mean_var = values.var() + 0.1
        x_coef = self.des_dir.cos() * self.des_dir_weight  # * mean_var
        y_coef = self.des_dir.sin() * self.des_dir_weight  # * mean_var

        values[...] += self.grid.view(1, self.grid_points, 1, 1, 1) * x_coef
        values[...] += self.grid.view(1, 1, self.grid_points, 1, 1) * y_coef
        values[...] += self.grid.view(1, 1, 1, self.grid_points, 1) * x_coef
        values[...] += self.grid.view(1, 1, 1, 1, self.grid_points) * y_coef

        return values

    def adjust_8D_values(self, values):
        """Adds other optimzation terms (direction) to 4D values.

        dim 0: foot 0  x
        dim 1: foot 0  y
        dim 2: foot 1  x
        dim 3: foot 1  y
        """
        # self.des_dir = torch.tensor(
        #     [self.pi * (self.task.progress_buf[0] % 4) / 2.0
        #      + self.pi / 4.0],
        #     device=self.device)
        # values = torch.zeros([self.grid_points] * 4, device=self.device)
        # values = torch.rand([self.grid_points] * 4, device=self.device)

        # add directional coefficients
        # mean_var = values.var() + 0.1
        x_coef = self.des_dir.cos() * self.des_dir_weight  # * mean_var
        y_coef = self.des_dir.sin() * self.des_dir_weight  # * mean_var

        values[...] += self.grid.view(1, 1, 1, 1, 1, self.grid_points, 1, 1, 1) * x_coef
        values[...] += self.grid.view(1, 1, 1, 1, 1, 1, self.grid_points, 1, 1) * y_coef
        values[...] += self.grid.view(1, 1, 1, 1, 1, 1, 1, self.grid_points, 1) * x_coef
        values[...] += self.grid.view(1, 1, 1, 1, 1, 1, 1, 1, self.grid_points) * y_coef

        return values

    def get_values(self, batch_obs):
        test_obs = self.player._preproc_obs(batch_obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : test_obs,
            'rnn_states' : self.player.states
        }
        with torch.cuda.amp.autocast(enabled=False):
            res_dict = self.player.model(input_dict)
        raw_values = res_dict["values"]
        values = self.player.value_mean_std(raw_values, unnorm=True)
        return values


class MyIterator:
    def __init__(self, data, batch_size=10):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            output = self.data[self.index: self.index + self.batch_size]
            self.index += self.batch_size
            return output
        else:
            raise StopIteration