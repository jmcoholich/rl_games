from os import stat_result
import matplotlib.pyplot as plt
import numpy as np
import torch

from tasks.aliengo_utils.utils import batch_z_2D_rot_mat


class ValueProcesser:
    def __init__(self, player, des_dir=0.0, des_dir_coef=0.0,
            start_after_n_steps=60, file_prefix="value_search", box_len=0.15,
            grid_points=13, random_footsteps=False):
        """
        Update: This code is now vectorized across environments. The first
        dimension is always the environment dimension.
        """
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
        self.ss_search_r = 0.2
        self.grid_points = grid_points
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

        if (self.grid_points - 1) % (self.num_plot_rows - 1) != 0:
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
            return self.search_4D(obs)
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
            self.fg.footsteps[self.env_arange, self.fg.current_footstep] = optimal_targets
            self.fg.plot_footstep_targets(current_only=True)
            if self.optmize_current_step:
                output_obs = self.task.observe(recalculating_obs=True)

        if self.make_plots or self.save_video_frames:
            self.plot_4D_values(values[0], max_idx[0])
        return output_obs

    '''
    def generate_4D_test_obs_old(self, obs, i):
        """
        dim 0: foot 0  x
        dim 1: foot 0  y
        dim 2: foot 1  x
        dim 3: foot 1  y
        """
        foot_idcs = self.fg.get_footstep_idcs(self.fg.current_footstep)[i]
        # next line will fail if executed before agent hits first targets
        prev_targets = self.fg.footsteps[i, self.fg.current_footstep[i] - 2]
        test_obs = obs[i].tile([self.grid_points] * 4 + [1])

        # generate grids centered around previous footstep targets aligned
        # with global coordinate system
        test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * 2] = \
            self.grid.view(self.grid_points, 1, 1, 1) + prev_targets[0, 0]

        test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * 2 + 1] = \
            self.grid.view(1, self.grid_points, 1, 1) + prev_targets[0, 1]

        test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * 2] = \
            self.grid.view(1, 1, self.grid_points, 1) + prev_targets[1, 0]

        test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * 2 + 1] = \
            self.grid.view(1, 1, 1, self.grid_points) + prev_targets[1, 1]

        # subtract robot feet global positions from the grid
        test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * 2] -= \
            self.task.foot_center_pos[i, foot_idcs[0], 0]
        test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * 2 + 1] -= \
            self.task.foot_center_pos[i, foot_idcs[0], 1]
        test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * 2] -= \
            self.task.foot_center_pos[i, foot_idcs[1], 0]
        test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * 2 + 1] -= \
            self.task.foot_center_pos[i, foot_idcs[1], 1]

        # rotate to align with robot yaw
        yaw = self.task.base_euler[i, 2]
        rot_mat = batch_z_2D_rot_mat(-yaw)
        test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * 2: self.start_idx + foot_idcs[0] * 2 + 2] = \
            (rot_mat @ test_obs[:, :, :, :, self.start_idx + foot_idcs[0] * 2: self.start_idx + foot_idcs[0] * 2 + 2].unsqueeze(-1)).squeeze(-1)
        test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * 2: self.start_idx + foot_idcs[1] * 2 + 2] = \
            (rot_mat @ test_obs[:, :, :, :, self.start_idx + foot_idcs[1] * 2: self.start_idx + foot_idcs[1] * 2 + 2].unsqueeze(-1)).squeeze(-1)

        return test_obs
    '''

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
        yaw = self.task.base_euler[:, 2]
        rot_mat = batch_z_2D_rot_mat(-yaw).view(n_env, 1, 1, 1, 1, 2, 2)
        ea = ea.unsqueeze(-1)

        for i in range(self.num_envs):
            rot_mat = batch_z_2D_rot_mat(-yaw[i])
            test_obs[i, :, :, :, :, self.start_idx + foot_idcs[i, 0] * p: self.start_idx + foot_idcs[i, 0] * p + p] = \
                (rot_mat @ test_obs[i, :, :, :, :, self.start_idx + foot_idcs[i, 0] * p: self.start_idx + foot_idcs[i, 0] * p + p].unsqueeze(-1)).squeeze(-1)
            test_obs[i, :, :, :, :, self.start_idx + foot_idcs[i, 1] * p: self.start_idx + foot_idcs[i, 1] * p + p] = \
                (rot_mat @ test_obs[i, :, :, :, :, self.start_idx + foot_idcs[i, 1] * p: self.start_idx + foot_idcs[i, 1] * p + p].unsqueeze(-1)).squeeze(-1)

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

        # optimal_targets[0, 0] = self.grid2m(max_idx[0])
        # optimal_targets[0, 1] = self.grid2m(max_idx[1])
        # optimal_targets[1, 0] = self.grid2m(max_idx[2])
        # optimal_targets[1, 1] = self.grid2m(max_idx[3])

        # now add prev targets to grid positions
        # prev_targets = self.fg.footsteps[0, self.fg.current_footstep[0] - 2]
        # optimal_targets[:, :2] += prev_targets[:, :2]

        return optimal_targets, max_idx

    def grid2m(self, idx):
        """Treats every input as a position in the grid.
        Works with tensors and numbers.
        """
        return (idx / (self.grid_points - 1.0) - 0.5) * 2 * self.box_len

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

    def get_values(self, batch_obs):
        test_obs = self.player._preproc_obs(batch_obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : test_obs,
            'rnn_states' : self.player.states
        }
        with torch.no_grad():
            res_dict = self.player.model(input_dict)
        raw_values = res_dict["values"]
        values = self.player.value_mean_std(raw_values, unnorm=True)
        return values

