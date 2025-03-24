"""
Main model implementation
Reference: https://github.com/sxyu/pixel-nerf
"""
import torch
from .encoder import ImageEncoder
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from src.util import repeat_interleave
import os
import os.path as osp
import warnings
import pdb
import numpy as np
import src.util as util

class PixelNeRFNet2(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.encoder = make_encoder(conf["encoder"])
        self.use_encoder = conf.get_bool("use_encoder", True)

        self.use_xyz = conf.get_bool("use_xyz", False)

        assert self.use_encoder or self.use_xyz  # Must use some feature..

        self.normalize_z = conf.get_bool("normalize_z", True)

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.get_bool("use_code", False)  # Positional encoding
        self.use_code_viewdirs = conf.get_bool(
            "use_code_viewdirs", True
        )  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)
        self.use_global_encoder = conf.get_bool("use_global_encoder", False)

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        if self.use_global_encoder:
            # Global image feature
            self.global_encoder = ImageEncoder.from_conf(conf["global_encoder"])
            self.global_latent_size = self.global_encoder.latent_size
            d_latent += self.global_latent_size

        d_out = 129

        self.latent_size = self.encoder.latent_size

        self.mlp_coarse = torch.nn.ModuleList([
            torch.nn.Linear(554, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 129),
        ])
        self.mlp_fine = None
        self.mlp_coarse_rgb = torch.nn.ModuleList([
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 131),
        ])
        self.mlp_fine_rgb = None

        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.num_objs = 0
        self.num_views_per_obj = 1
        self.use_KB = False
        self.exo_cams = None
        self.device = "cpu"

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:
            self.global_encoder(images)

    def set_KB(self,use_KB=False, exo_cams = None, device="cpu"):
        self.use_KB = use_KB
        self.exo_cams = exo_cams
        self.device = device

    def forward(self, xyz, coarse=True, viewdirs=None, far=False,use_KB=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[
                ..., 0
            ]
            xyz = xyz_rot + self.poses[:, None, :3, 3]
            
            if self.d_in > 0:
                # * Encode the xyz coordinates
                if self.use_xyz:
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else:
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs:
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)

                if self.use_viewdirs:
                    # * Encode the view directions
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                    viewdirs = torch.matmul(
                        self.poses[:, None, :3, :3], viewdirs
                    )  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat(
                        (z_feature, viewdirs), dim=1
                    )  # (SB*B, 4 or 6)

                if self.use_code and self.use_code_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                mlp_input = z_feature

            if self.use_encoder:
                # Grab encoder's latent code.
                if self.use_KB == False:
                    uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
                    uv *= repeat_interleave(
                        self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
                    )
                    uv += repeat_interleave(
                        self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
                    )  # (SB*NS, B, 2)
                else:
                    xyz[:, :, 1:3] *= -1
                    xyz = xyz / xyz[:, :, 2:]
                    xyz_np = xyz.cpu().detach().numpy()
                    uv = np.zeros((SB*NS,B,2))
                    for sb in range(SB*NS):
                        for i in range(B):
                            proj =  self.exo_cams[sb%NS].project(xyz_np[sb][i])
                            if proj is not None:
                                uv[sb,i] = proj/( self.exo_cams[0].get_image_size()[0] / 480)
                    uv = torch.from_numpy(uv).to(self.device).to(torch.float32)
                latent = self.encoder.index(
                    uv, None, self.image_shape
                )  # (SB * NS, latent, B)

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2).reshape(
                    -1, self.latent_size
                )  # (SB * NS * B, latent)


                if self.d_in == 0:
                    # z_feature not needed
                    mlp_input = latent
                else:
                    mlp_input = torch.cat((latent, z_feature), dim=-1)

            if self.use_global_encoder:
                # Concat global latent code if enabled
                global_latent = self.global_encoder.latent
                assert mlp_input.shape[0] % global_latent.shape[0] == 0
                num_repeats = mlp_input.shape[0] // global_latent.shape[0]
                global_latent = repeat_interleave(global_latent, num_repeats)
                mlp_input = torch.cat((global_latent, mlp_input), dim=-1)

            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None

            mlp_input = util.combine_interleaved(
                        mlp_input, (self.num_views_per_obj, B), "average"
                    )
            if coarse or self.mlp_fine is None:
                mlp_output = mlp_input
                for layer in self.mlp_coarse:
                    mlp_output = layer(mlp_output)
            else:
                mlp_output = mlp_input
                for layer in self.mlp_fine:
                    mlp_output = layer(mlp_output)

            # Interpret the output
            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :-1]
            sigma = mlp_output[..., -1:]

            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)
        return output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
    
    def get_color(self,rgb_final,coarse):
        if coarse:
            for layer in self.mlp_coarse_rgb:
                rgb_final = layer(rgb_final)
        else:
            for layer in self.mlp_fine_rgb:
                rgb_final = layer(rgb_final)
        return rgb_final
