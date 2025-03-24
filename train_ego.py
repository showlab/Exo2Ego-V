import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import warnings
from src.model import make_model, loss
from src.render import NeRFRenderer
import src.util as util
import numpy as np
import torch.nn.functional as F
import torch
from dotmap import DotMap
import pdb
import math
import shutil
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId

class PixelNeRFTrainer():
    def __init__(self,net,conf,renderer=None,render_par=None,args=None,device='cpu'):
        self.args = args
        self.net = net
        self.device = device
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.nviews = list(map(int, args.nviews.split()))
        if renderer is not None:
            self.renderer = renderer
        else:
            self.renderer = NeRFRenderer.from_conf(conf["renderer"]).to(
                device=device
            )

        if render_par is not None:
            self.render_par = render_par
        else:
            self.render_par = self.renderer.bind_parallel(self.net, args.gpu_id)

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                self.renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=self.device)
                )

        self.z_near = args.near
        self.z_far = args.far
        self.ego_z_near = args.near
        self.ego_z_far = args.far

        self.use_bbox = False
        self.unproj = torch.load(os.path.join(args.pose_dir,"unproj_ego.pt"))
        self.exo_cams,self.unproj_exo = None, None
        if args.use_KB:
            self.net.set_KB(args.use_KB, self.exo_cams,device,args.exo_width)

        
    def calc_losses(self, data, is_train=True, global_step=0, weight_dtype=torch.float32):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=self.device).to(weight_dtype) # (SB, NV, 3, H, W)
        all_ego_images = data["ego_img"].to(device=self.device).to(weight_dtype)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=self.device).to(weight_dtype)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"].to(weight_dtype)  # (SB)
        all_c = data.get("c").to(weight_dtype)  # (SB)
        all_ego_focals =  data["ego_focal"].to(weight_dtype)
        all_scenes = data['scene']

        if self.use_bbox and global_step >= self.args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            ego_focal = all_ego_focals[obj_idx]
            ego_images = all_ego_images[obj_idx][None]
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
                ego_c = data["ego_c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5
            image_ego = ego_images* 0.5 + 0.5
            cam_rays = util.gen_rays(poses[:-1], W, H, focal, self.z_near, self.z_far, c=c,use_KB=self.args.use_KB,unproj=self.unproj_exo)  # (NV, H, W, 8)
            cam_rays_ego = util.gen_rays(poses[-1:], data['ego_img'].shape[-2], data['ego_img'].shape[-1], ego_focal, self.ego_z_near, self.ego_z_far, c=ego_c, use_fishereye=self.args.use_fishereye,unproj=self.unproj[all_scenes[obj_idx]].to(self.device).to(weight_dtype))  # (NV, H, W, 8)
            rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)
            rgb_gt_ego_all = image_ego
            rgb_gt_ego_all = (
                rgb_gt_ego_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)

            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, self.args.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                if not self.args.use_fishereye:
                    pix_inds = torch.randint(0, NV * H * W+data['ego_img'].shape[-2]*data['ego_img'].shape[-1], (self.args.ray_batch_size,))
                    pix_inds_ego = torch.masked_select(pix_inds, pix_inds >= NV * H * W) - NV * H * W
                    pix_inds = torch.masked_select(pix_inds, pix_inds < NV * H * W)
                else:
                    if not self.args.render_whole:
                        exo_num = (self.args.ray_batch_size//NV)*(NV-1)
                        ego_num = self.args.ray_batch_size - exo_num
                        pix_inds = torch.randint(0, NV * H * W, (exo_num,))
                        indexes = cam_rays_ego[0][:,:,-3]
                        indexes = indexes.view(-1,)
                        non_nan_indices = torch.nonzero(~torch.isnan(indexes), as_tuple=False)
                        pix_inds_ego = torch.randperm(non_nan_indices.size(0))[:ego_num]
                        pix_inds_ego = non_nan_indices[pix_inds_ego].squeeze(-1)
                    else:
                        indexes = cam_rays_ego[0][:,:,-3]
                        indexes = indexes.view(-1,)
                        non_nan_indices = torch.nonzero(~torch.isnan(indexes), as_tuple=False)
                        pix_inds_ego = torch.randperm(non_nan_indices.size(0))[:self.args.ray_batch_size]
                        pix_inds_ego = non_nan_indices[pix_inds_ego].squeeze(-1)
                        pix_inds = None



            if not self.args.render_whole:
                rgb_gt = rgb_gt_all[pix_inds] # (ray_batch_size, 3)
                rgb_gt_ego = rgb_gt_ego_all[pix_inds_ego]
                rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                    device=self.device
                )  # (ray_batch_size, 8)
                rays_ego = cam_rays_ego.view(-1, cam_rays_ego.shape[-1])[pix_inds_ego].to(
                    device=self.device
                )  # (ray_batch_size, 8)
                rays = torch.cat((rays,rays_ego))
                rgb_gt = torch.cat((rgb_gt,rgb_gt_ego))

                all_rgb_gt.append(rgb_gt)
                all_rays.append(rays)
            else:
                if pix_inds is not None:
                    rgb_gt = rgb_gt_all[pix_inds] # (ray_batch_size, 3)
                    rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                        device=self.device
                    )  
                    all_rgb_gt.append(rgb_gt)
                    all_rays.append(rays)
                if pix_inds_ego is not None:
                    rgb_gt_ego = rgb_gt_ego_all[pix_inds_ego]
                    rays_ego = cam_rays_ego.view(-1, cam_rays_ego.shape[-1])[pix_inds_ego].to(
                        device=self.device
                    )  
                    rays = rays_ego
                    rgb_gt = rgb_gt_ego
                    all_rgb_gt.append(rgb_gt)
                    all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(self.device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None
        self.net.encode(
            src_images,
            src_poses,
            all_focals.to(device=self.device),
            c=all_c.to(device=self.device) if all_c is not None else None,
        )

        render_dict = DotMap(self.render_par(all_rays, want_weights=True,))
        features_final = None
        rgb_coarse_np = None
        if self.args.render_whole and pix_inds_ego is not None:
            features_final = torch.zeros((data['ego_img'].shape[-2]*data['ego_img'].shape[-1],render_dict.coarse.feature[0].shape[-1])).to(self.device).to(weight_dtype)
            features_final[pix_inds_ego] = render_dict.coarse.feature[0]
            features_final = features_final.reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1],render_dict.coarse.feature[0].shape[-1])
            features_final = features_final.permute(2,0,1)[None]

            rgb_coarse_np = torch.zeros((data['ego_img'].shape[-2]*data['ego_img'].shape[-1],3)).to(self.device)
            rgb_coarse_np[pix_inds_ego] = render_dict.coarse.rgb[0]
            rgb_coarse_np = rgb_coarse_np.reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1],3).permute(2,0,1)[None]

        rgb_loss = self.rgb_coarse_crit(render_dict.coarse.rgb, all_rgb_gt)
        return render_dict,features_final,rgb_loss,rgb_coarse_np


    def vis_step(self, data, global_step, idx=None,weight_dtype=torch.float32):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx

        images = data["images"][batch_idx].to(device=self.device).to(weight_dtype)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=self.device).to(weight_dtype) # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1].to(weight_dtype)  # (1)
        ego_focal  = data["ego_focal"][batch_idx : batch_idx + 1].to(weight_dtype)  # (1)
        ego_img = data["ego_img"][batch_idx : batch_idx + 1].to(weight_dtype)
        c = data.get("c").to(weight_dtype)
        ego_c = data.get("ego_c").to(weight_dtype)
        scene = data.get("scene")[batch_idx : batch_idx + 1]

        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
            ego_c = ego_c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses[:-1], W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)

        if self.args.use_fishereye:
            cam_rays_ego = util.gen_rays(poses[-1:], ego_img.shape[-2], ego_img.shape[-1], ego_focal, self.ego_z_near, self.ego_z_far, c=ego_c,use_fishereye= self.args.use_fishereye,unproj=self.unproj[scene[0]].to(self.device).to(weight_dtype))  # (NV, H, W, 8)
        elif self.args.use_KB:
            cam_rays_ego = util.gen_rays(poses[-1:], ego_img.shape[-2], ego_img.shape[-1], ego_focal, self.ego_z_near, self.ego_z_far, c=ego_c, use_KB= True,unproj=self.unproj_exo)  # (NV, H, W, 8)
        else:
            cam_rays_ego = util.gen_rays(poses[-1:], ego_img.shape[-2], ego_img.shape[-1], ego_focal, self.ego_z_near, self.ego_z_far, c=ego_c)  # (NV, H, W, 8)

        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)
        images_0to1_ego = ego_img* 0.5 + 0.5

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        if NV - curr_nviews>0:
            view_dest = np.random.randint(0, NV - curr_nviews)
            for vs in range(curr_nviews):
                view_dest += view_dest >= views_src[vs]

        else:
            view_dest = NV-1
        if self.args.only_ego:
            views_src = np.array([0,1,2,3])
            view_dest = 4

        views_src = torch.from_numpy(views_src)
        self.renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .float()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        gt = images_0to1_ego[0].permute(1, 2, 0).cpu().float().numpy().reshape(data['ego_img'].shape[-2], data['ego_img'].shape[-1], 3)

        import time
        start_time = time.time()
        with torch.no_grad():
            test_rays = cam_rays_ego[0]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            self.net.encode(
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=self.device),
                c=c.to(device=self.device) if c is not None else None,
            )
            if not  self.args.use_fishereye:
                test_rays = test_rays.reshape(1, data['ego_img'].shape[-2]*data['ego_img'].shape[-1], -1)
                render_dict = DotMap(self.render_par(test_rays, want_weights=True))
                coarse = render_dict.coarse
                fine = render_dict.fine
                alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1])
                rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1], 3)
                depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1])
            else:
                test_rays = test_rays.reshape(1, data['ego_img'].shape[-2]*data['ego_img'].shape[-1], -1)
                non_nan_indices = torch.nonzero(~torch.isnan(test_rays[0,:,-3]), as_tuple=False).squeeze(-1)
                nan_indices = torch.nonzero(torch.isnan(test_rays[0,:,-3]), as_tuple=False).squeeze(-1)
                test_rays = test_rays[:,non_nan_indices,:]
                render_dict = DotMap(self.render_par(test_rays, want_weights=True))
                coarse = render_dict.coarse
                fine = render_dict.fine
                alpha_coarse_np = torch.zeros(data['ego_img'].shape[-2]*data['ego_img'].shape[-1]).to(self.device).to(weight_dtype)
                rgb_coarse_np = torch.zeros((data['ego_img'].shape[-2]*data['ego_img'].shape[-1],3)).to(self.device).to(weight_dtype)
                depth_coarse_np = torch.zeros(data['ego_img'].shape[-2]*data['ego_img'].shape[-1]).to(self.device).to(weight_dtype)
                alpha_coarse_np[non_nan_indices] = coarse.weights[0].sum(dim=-1)
                alpha_coarse_np = alpha_coarse_np.cpu().float().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1])
                rgb_coarse_np[non_nan_indices] = coarse.rgb[0]
                rgb_coarse_np = rgb_coarse_np.cpu().float().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1],3)
                depth_coarse_np[non_nan_indices] = coarse.depth[0]
                depth_coarse_np = depth_coarse_np.cpu().float().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1])

            using_fine = len(fine) > 0

            if using_fine:
                if not self.args.use_fishereye:
                    alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1])
                    depth_fine_np = fine.depth[0].cpu().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1])
                    rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1], 3)
                else:
                    alpha_fine_np = torch.zeros(data['ego_img'].shape[-2]*data['ego_img'].shape[-1]).to(self.device)
                    rgb_fine_np = torch.zeros((data['ego_img'].shape[-2]*data['ego_img'].shape[-1],3)).to(self.device)
                    depth_fine_np = torch.zeros(data['ego_img'].shape[-2]*data['ego_img'].shape[-1]).to(self.device)
                    alpha_fine_np[non_nan_indices] = fine.weights[0].sum(dim=-1)
                    alpha_fine_np = alpha_fine_np.cpu().float().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1])
                    rgb_fine_np[non_nan_indices] = fine.rgb[0]
                    rgb_fine_np = rgb_fine_np.cpu().float().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1],3)
                    depth_fine_np[non_nan_indices] = fine.depth[0]
                    depth_fine_np = depth_fine_np.cpu().float().numpy().reshape(data['ego_img'].shape[-2],data['ego_img'].shape[-1])

        end_time = time.time()
        execution_time = end_time - start_time
        print("Code execution time: {:.2f} seconds".format(execution_time))

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        if H == data['ego_img'].shape[-2]:
            vis_list = [
                *source_views,
                gt,
                depth_coarse_cmap,
                rgb_coarse_np,
                alpha_coarse_cmap,
            ]
        else:
            vis_list = [
                gt,
                depth_coarse_cmap,
                rgb_coarse_np,
                alpha_coarse_cmap,
            ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            if H == data['ego_img'].shape[-2]:
                vis_list = [
                    *source_views,
                    gt,
                    depth_fine_cmap,
                    rgb_fine_np,
                    alpha_fine_cmap,
                ]
            else:
                vis_list = [
                    gt,
                    depth_fine_cmap,
                    rgb_fine_np,
                    alpha_fine_cmap,
                ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        self.renderer.train()
        return vis, vals
