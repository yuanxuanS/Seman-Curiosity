import torch
from torch import nn
import torch.nn.functional as F

from .train_utils import get_loss_fn, get_activation_fn
from .model import get_semantic_encoder_decoder


class SemanticMapperModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Define loss functions
        object_loss_type = self.cfg.MODEL.object_loss_type
        assert object_loss_type in ["bce", "l1", "l2", "xent"]
        self.object_loss_fn = get_loss_fn(object_loss_type)
        area_loss_type = self.cfg.MODEL.area_loss_type
        assert area_loss_type in ["bce", "l1", "l2"]
        self.area_loss_fn = get_loss_fn(area_loss_type)
        # Define models
        ndirs = None
        self.dirs_map = None
        self.inv_dists_map = None
        enable_directions = self.cfg.DATASET.enable_directions
        ndirs = len(self.cfg.DATASET.prediction_directions)
        enable_locations = self.cfg.DATASET.enable_locations
        enable_actions = self.cfg.DATASET.enable_actions
        assert not (enable_locations and enable_directions)
        assert not (enable_actions and enable_directions)
        assert not (enable_actions and enable_locations)
        enable_area_head = self.cfg.DATASET.enable_unexp_area
        if enable_directions:
            assert object_loss_type == "xent"
            assert self.cfg.MODEL.object_activation == "none"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "dirs"
            self.cfg.MODEL.ndirs = ndirs
            self.cfg.freeze()
        if enable_locations:
            assert object_loss_type in ["l1", "l2"]
            assert self.cfg.MODEL.object_activation == "sigmoid"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "locs"
            self.cfg.freeze()
        if enable_actions:
            assert object_loss_type == "xent"
            assert self.cfg.MODEL.object_activation == "none"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "acts"
            self.cfg.freeze()
        if enable_area_head:
            self.cfg.defrost()
            self.cfg.MODEL.enable_area_head = enable_area_head
            self.cfg.freeze()
        (
            self.encoder,
            self.object_decoder,
            self.area_decoder,
        ) = get_semantic_encoder_decoder(self.cfg)
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.OPTIM.lr)
        # Define scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.OPTIM.lr_sched_milestones,
            gamma=cfg.OPTIM.lr_sched_gamma,
        )
        # Define activation functions
        self.object_activation = get_activation_fn(self.cfg.MODEL.object_activation)
        self.area_activation = get_activation_fn(self.cfg.MODEL.area_activation)

    def forward(self, x):
        embedding = self.encoder(x)
        object_preds = self.object_activation(self.object_decoder(embedding))
        area_preds = None
        if self.area_decoder is not None:
            area_preds = self.area_activation(self.area_decoder(embedding))
        return object_preds, area_preds

    def get_inv_dists_map(self, x):
        # x - (bs, N, M, M)
        M = x.shape[2]
        assert x.shape[3] == M
        Mby2 = M // 2
        # Compute a directions map that stores the direction value in
        # degrees for each location on the map.
        x_values = torch.arange(0, M, 1).float().unsqueeze(0) - Mby2  # (1, M)
        y_values = torch.arange(0, M, 1).float().unsqueeze(1) - Mby2  # (M, 1)
        dirs_map = torch.atan2(y_values, x_values)  # (M, M)
        dirs_map = torch.rad2deg(dirs_map).view(1, 1, M, M)
        dirs_map = (dirs_map + 360) % 360
        # Compute a distances map that stores the distance value in unit
        # cells for each location on the map.
        dists_map = torch.sqrt(x_values**2 + y_values**2)  # (M, M)
        inv_dists_map = torch.exp(-dists_map).view(1, 1, M, M)
        return dirs_map.to(x.device), inv_dists_map.to(x.device)

    def infer(self, x, do_forward_pass=True, input_maps=None, avg_preds=True):
        if do_forward_pass:
            object_preds, area_preds = self(x)
        else:
            assert input_maps is not None
            object_preds, area_preds = x
            x = input_maps

        if self.cfg.MODEL.output_type == "dirs":
            ####################################################################
            # Convert predicted directions to points on a map.
            ####################################################################
            # Convert predictions to angles
            # object_preds - (B, N, D)
            angles = torch.Tensor(self.cfg.DATASET.prediction_directions)
            angles = angles.to(object_preds.device)  # (D, )
            pred_dir_ixs = torch.argmax(object_preds, dim=2)  # (B, N)
            B, N = pred_dir_ixs.shape
            pred_dir_ixs = pred_dir_ixs.view(-1)  # (B * N)
            pred_dirs = torch.gather(angles, 0, pred_dir_ixs)
            pred_dirs = pred_dirs.view(B, N, 1, 1)  # (B, N, 1, 1)
            # Identify frontiers
            frontiers = self.calculate_frontiers(x)  # (B, 1, H, W)
            # Select the nearest frontier point along the predicted direction
            if self.dirs_map is None:
                self.dirs_map, self.inv_dists_map = self.get_inv_dists_map(x)
            dirs_map = self.dirs_map
            inv_dists_map = self.inv_dists_map
            delta = torch.abs(angles[1] - angles[0]).item() / 2
            dirs_map = dirs_map.to(x.device)
            abs_diff = torch.abs(dirs_map - pred_dirs) % 360  # (B, N, H, W)
            diff = (abs_diff - 180) % 360 - 180  # Convert from (0, 360) to (-180, 180)
            is_within_angle = (torch.abs(diff) <= delta).float()
            inv_dists_map = inv_dists_map.to(x.device)  # (1, 1, H, W)
            complex_mask = frontiers * is_within_angle * inv_dists_map  # (B, N, H, W)
            _, _, H, W = complex_mask.shape
            fpoint = torch.argmax(complex_mask.view(B, N, -1), dim=2)  # (B, N)
            # If no frontier point exists, sample a random point
            # along the predicted direction.
            no_frontier_mask = torch.all(
                complex_mask.view(B, N, -1) == 0, dim=2
            )  # (B, N)
            angle_mask = is_within_angle.view(B * N, H * W)
            spoint = torch.multinomial(angle_mask, 1)  # (B * N, 1)
            spoint = spoint.view(B, N)
            fpoint[no_frontier_mask] = spoint[no_frontier_mask]
            # Create a map with these predictions
            preds_map = torch.zeros_like(complex_mask)  # (B, N, H, W)
            preds_map = preds_map.view(B, N, -1)  # (B, N, H * W)
            preds_map.scatter_(2, fpoint.unsqueeze(2), 1.0)
            preds_map = preds_map.view(B, N, H, W)
            object_preds = F.max_pool2d(preds_map, 7, stride=1, padding=3)
        elif self.cfg.MODEL.output_type == "locs":
            ####################################################################
            # Convert predicted locations to points on a map.
            ####################################################################
            # Convert predictions to map locations
            # preds - (B, N, 2)
            B, N, H, W = x.shape
            preds_x = torch.clamp(object_preds[:, :, 0] * W, 0, W - 1)  # (B, N)
            preds_y = torch.clamp(object_preds[:, :, 1] * H, 0, H - 1)  # (B, N)
            # Convert to row-major form
            preds_xy = (preds_y * W + preds_x).long()  # (B, N)
            # Create a map with these predictions
            preds_map = torch.zeros_like(x)  # (B, N, H, W)
            preds_map = preds_map.view(B, N, -1)
            preds_map.scatter_(2, preds_xy.unsqueeze(2), 1.0)
            preds_map = preds_map.view(B, N, H, W)
            object_preds = F.max_pool2d(preds_map, 7, stride=1, padding=3)
        elif self.cfg.MODEL.output_type == "acts":
            ####################################################################
            # Retain predicted actions as actions
            ####################################################################
            # object_preds - (B, N, 4)
            assert not avg_preds
        # By default, average the two predictions and return it.
        if avg_preds:
            outputs = object_preds
            if area_preds is not None:
                outputs = (object_preds + area_preds) / 2.0
            return outputs
        else:
            return object_preds, area_preds

    def calculate_frontiers(self, x):
        # x - semantic map of shape (B, N, H, W)
        free_map = (x[:, 0] >= 0.5).float()  # (B, H, W)
        exp_map = torch.max(x, dim=1).values >= 0.5  # (B, H, W)
        unk_map = (~exp_map).float()  # (B, H, W)
        # Compute frontiers (reference below)
        # https://github.com/facebookresearch/exploring_exploration/blob/09d3f9b8703162fcc0974989e60f8cd5b47d4d39/exploring_exploration/models/frontier_agent.py#L132
        unk_map_shiftup = F.pad(unk_map, (0, 0, 0, 1))[:, 1:]
        unk_map_shiftdown = F.pad(unk_map, (0, 0, 1, 0))[:, :-1]
        unk_map_shiftleft = F.pad(unk_map, (0, 1, 0, 0))[:, :, 1:]
        unk_map_shiftright = F.pad(unk_map, (1, 0, 0, 0))[:, :, :-1]
        frontiers = (
            (free_map == unk_map_shiftup)
            | (free_map == unk_map_shiftdown)
            | (free_map == unk_map_shiftleft)
            | (free_map == unk_map_shiftright)
        ) & (
            free_map == 1
        )  # (B, H, W)
        # Dilate the frontiers
        frontiers = frontiers.unsqueeze(1).float()  # (B, 1, H, W)
        frontiers = torch.nn.functional.max_pool2d(frontiers, 7, stride=1, padding=3)
        return frontiers

    def undo_memory_opts(self, batch):
        inputs, labels = batch
        inputs["semmap"] = inputs["semmap"].float()
        labels["semmap"] = labels["semmap"].float()
        labels["object_pfs"] = labels["object_pfs"].float() / 1000.0
        if "area_pfs" in labels:
            labels["area_pfs"] = labels["area_pfs"].float() / 1000.0
        return (inputs, labels)

    def batch_step(self, batch):
        inputs, labels = batch
        input_maps = inputs["semmap"]
        object_preds, area_preds = self(input_maps)
        losses = {}
        if self.cfg.MODEL.output_type == "map":
            # object_preds - (B, N + 2, H, W)
            # Ignore free-space, wall predictions
            y_hat = object_preds[:, 2:]
            y = labels["object_pfs"][:, 2:]
            mask = labels["loss_masks"][:, 2:]
            loss = self.object_loss_fn(y_hat, y)
            # Evaluate predictions only on mask = 1 regions
            mask_sum = mask.sum(dim=3).sum(dim=2) + 1e-16  # (b, 1)
            loss = (loss * mask).sum(dim=3).sum(dim=2) / mask_sum  # (b, c)
            loss = loss.mean()
            losses["object_pf_loss"] = loss.item()
        elif self.cfg.MODEL.output_type == "dirs":
            D = len(self.cfg.DATASET.prediction_directions)
            # object_preds - (B, N + 2, D)
            # Ignore free-space, wall predictions
            y_hat = object_preds[:, 2:]  # (B, N, D)
            y = labels["dirs"][:, 2:]  # (B, N)
            # Replace non-object labels with 0 and mask these in the loss
            mask = y != D  # (B, N)
            y[~mask] = 0
            mask = mask.float()
            mask_sum = mask.sum(dim=1) + 1e-10  # (B, )
            y_hat = y_hat.permute(0, 2, 1)  # (B, D, N)
            loss = self.object_loss_fn(y_hat, y)  # (B, N)
            loss = (loss * mask).sum(dim=1) / mask_sum
            loss = loss.mean()
            losses["object_pf_loss"] = loss.item()
        elif self.cfg.MODEL.output_type == "locs":
            # object_preds - (B, N + 2, 2)
            # Ignore free-space, wall predictions
            y_hat = object_preds[:, 2:]  # (B, N, 2)
            y = labels["locs"][:, 2:]  # (B, N, 2)
            # Replace non-object labels with 0 and mask these in the loss
            mask = torch.all(y >= 0, dim=2, keepdim=True)  # (B, N, 1)
            mask = mask.expand(-1, -1, 2)  # (B, N, 2)
            y[~mask] = 0
            mask = mask.float()
            mask_sum = mask.sum(dim=2).sum(dim=1) + 1e-10  # (B, )
            loss = self.object_loss_fn(y_hat, y)  # (B, N, 2)
            loss = (loss * mask).sum(dim=2).sum(dim=1) / mask_sum  # (B, )
            loss = loss.mean()
            losses["object_pf_loss"] = loss.item()
        elif self.cfg.MODEL.output_type == "acts":
            # object_preds - (B, N + 2, 4)
            # Ignore free-space, wall predictions
            y_hat = object_preds[:, 2:]  # (B, N, 4)
            y = labels["acts"][:, 2:]  # (B, N)
            # Replace non-object labels with 0 and mask these in the loss
            mask = y >= 0  # (B, N)
            y[~mask] = 0
            mask = mask.float()
            mask_sum = mask.sum(dim=1) + 1e-10  # (B, )
            y_hat = y_hat.permute(0, 2, 1)  # (B, 4, N)
            loss = self.object_loss_fn(y_hat, y)  # (B, N)
            loss = (loss * mask).sum(dim=1) / mask_sum  # (B, )
            loss = loss.mean()
            losses["object_pf_loss"] = loss.item()

        if area_preds is not None:
            area_gts = labels["area_pfs"]  # (N, 1, H, W)
            area_pf_loss = self.area_loss_fn(area_preds, area_gts).mean()
            loss = loss + area_pf_loss
            losses["area_pf_loss"] = area_pf_loss.item()

        return {"loss": loss, "losses": losses}

    # def train_dataloader(self, is_distributed=False):
    #     self.train_dataset = SMPrecompDataset(self.cfg.DATASET, split="train")
    #     self.train_sampler = None
    #     if is_distributed:
    #         self.train_sampler = torch.utils.data.distributed.DistributedSampler(
    #             self.train_dataset
    #         )
    #     self.train_loader = DataLoader(
    #         self.train_dataset,
    #         batch_size=self.cfg.OPTIM.batch_size,
    #         shuffle=(self.train_sampler is None),
    #         num_workers=self.cfg.OPTIM.num_workers,
    #         pin_memory=True,
    #         sampler=self.train_sampler,
    #         collate_fn=collate_fn,
    #     )
    #     return self.train_loader

    # def val_dataloader(self):
    #     self.val_dataset = SMPrecompDataset(self.cfg.DATASET, split="val")
    #     self.val_loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.cfg.OPTIM.batch_size,
    #         num_workers=self.cfg.OPTIM.num_workers,
    #         pin_memory=True,
    #         collate_fn=collate_fn,
    #     )
    #     return self.val_loader

    # def test_dataloader(self):
    #     return self.val_dataloader()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def convert_to_data_parallel(self):
        self.encoder = nn.DataParallel(self.encoder)
        self.object_decoder = nn.DataParallel(self.object_decoder)
        if self.area_decoder is not None:
            self.area_decoder = nn.DataParallel(self.area_decoder)

    # def convert_to_distributed_data_parallel(self, rank):
    #     if self.cfg.LOGGING.verbose:
    #         print(f"=======> (0.3) breakpoint reached in local proc: {rank}")
    #     encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
    #     object_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.object_decoder)
    #     if self.area_decoder is not None:
    #         area_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.area_decoder)
    #     if self.cfg.LOGGING.verbose:
    #         print(f"=======> (0.4) breakpoint reached in local proc: {rank}")
    #     self.encoder = DDP(encoder, device_ids=[rank])
    #     self.object_decoder = DDP(object_decoder, device_ids=[rank])
    #     if self.area_decoder is not None:
    #         self.area_decoder = DDP(area_decoder, device_ids=[rank])
    #     if self.cfg.LOGGING.verbose:
    #         print(f"=======> (0.5) breakpoint reached in local proc: {rank}")
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.OPTIM.lr)
    #     # Define scheduler
    #     self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         self.optimizer,
    #         milestones=self.cfg.OPTIM.lr_sched_milestones,
    #         gamma=self.cfg.OPTIM.lr_sched_gamma,
    #     )

    def convert_object_pf_to_distance(self, opfs, min_value=1e-20, max_value=1.0):
        """
        opfs - (bs, N, H, W)
        """
        opfs = torch.clamp(opfs, min_value, max_value)
        data_cfg = self.cfg.DATASET
        max_d = data_cfg.object_pf_cutoff_dist
        dists = max_d - opfs * max_d
        return dists

    def convert_distance_to_object_pf(self, dists):
        """
        dists - (bs, N, H, W)
        """
        data_cfg = self.cfg.DATASET
        max_d = data_cfg.object_pf_cutoff_dist
        opfs = torch.clamp((max_d - dists) / max_d, 0.0, 1.0)
        return opfs

    def get_pf_cfg(self):
        return {"dthresh": self.cfg.DATASET.object_pf_cutoff_dist}

