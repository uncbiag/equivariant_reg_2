####python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=152.2.132.94 --master_port=1234 unigradicon_train_parallel.py                                         



from fixed_point_carl import show
from datetime import datetime
from icon_registration.config import device
import fixed_point_carl as fpc
import footsteps
import icon_registration as icon
import icon_registration.carl as carl
import icon_registration.data
import icon_registration.networks as networks
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchvision.utils
import os
os.environ["OMP_NUM_THREADS"]="8"

input_shape = [1, 1, 160, 160, 160]
def make_net(dimension, input_shape):
    unet = fpc.Equivariantize(fpc.SomeDownsampleNoDilationNet(dimension=dimension))
    ar = fpc.AttentionFeaturizer(unet, dimension=dimension)
    ts = ar

    ts = icon.network_wrappers.DownsampleNet(ts, dimension)
    for _ in range(3):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 21),
             #ts,
             icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension)), dimension)
         )
    for _ in range(2):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 11),
             #ts,
             icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension)), dimension)
         )
    ts = icon.network_wrappers.DownsampleNet(ts, dimension)
    #ts = icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension))

    for _ in range(1):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 8),
             #ts,
             carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension))
         )
    net = icon.losses.GradientICONSparse(ts, icon.LNCC(sigma=4), lmbda=1.5)
    #net = icon.losses.DiffusionRegularizedNet(ts, icon.losses.SquaredLNCC(sigma=4), lmbda=10)
    net.assign_identity_map(input_shape)
    net = carl.augmentify(net)
    net.assign_identity_map(input_shape)
    net.train()
    return net


def make_make_pair():

    def make_pair():
        image_A = []
        image_B = []
        for i in range(BATCH_SIZE):
            ds = random.choice(datasets.datasets_)
            pair = ds.get_pair()
            image_A.append(pair[0])
            image_B.append(pair[1])
        image_A = torch.cat(image_A)
        image_B = torch.cat(image_B)
        return image_A, image_B
    return make_pair
from icon_registration.train import to_floats, write_stats

def train_batchfunction(
    net,
    make_batch,
    steps
):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """
    import footsteps
    from torch.utils.tensorboard import SummaryWriter


    loss_curve = []

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local-rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    argv = parser.parse_args()

    import os

    world_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_rank == 0:
        writer = SummaryWriter(
            footsteps.output_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
            flush_secs=30,
        )

    torch.distributed.init_process_group(backend="nccl")

    device = torch.device("cuda:{}".format(local_rank))

    icon_registration.config.device = device

    net.to(device)
    
    ddp_model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    ddp_model.train()


    visualization_moving, visualization_fixed = [m[:8].to(device) for m in make_batch()]

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.0001)



    for iteration in range(0, steps):
        optimizer.zero_grad()
        moving_image, fixed_image = make_batch()
        loss_object = ddp_model(moving_image.to(device), fixed_image.to(device))
        loss = torch.mean(loss_object.all_loss)
        loss.backward()


        optimizer.step()

        if world_rank == 0:

            print(to_floats(loss_object))
            write_stats(writer, loss_object, iteration)

            if iteration % 5000 == 0:
                torch.save(
                    ddp_model.module.regis_net.state_dict(),
                    footsteps.output_dir + "network_weights_" + str(iteration),
                )
        if iteration % 300 == 0:
            ddp_model.eval()
            print("val (from train set)")
            warped = []
            with torch.no_grad():
                for i in range(8):
                    print( ddp_model(visualization_moving[i:i + 1], visualization_fixed[i:i + 1]))
                    warped.append(ddp_model.module.warped_image_A.cpu())
                warped = torch.cat(warped)
            ddp_model.train()
            if world_rank == 0:

                def render(im):
                    if len(im.shape) == 5:
                        im = im[:, :, :, :, im.shape[4] // 2]
                    if torch.min(im) < 0:
                        im = im - torch.min(im)
                    if torch.max(im) > 1:
                        im = im / torch.max(im)
                    return im[:, [0, 0, 0]].detach().cpu()

                writer.add_images(
                    "moving_image", render(visualization_moving[:8]), iteration, dataformats="NCHW"
                )
                writer.add_images(
                    "fixed_image", render(visualization_fixed[:8]), iteration, dataformats="NCHW"
                )
                writer.add_images(
                    "warped_moving_image",
                    render(warped),
                    iteration,
                    dataformats="NCHW",
                )
                writer.add_images(
                    "difference",
                    render(torch.clip((warped[:8, :1] - visualization_fixed[:8, :1].cpu()) + 0.5, 0, 1)),
                    iteration,
                    dataformats="NCHW",
                )



if __name__ == "__main__":
    import datasets


    net = make_net(3, input_shape)
    net.regis_net.load_state_dict(torch.load("results/unic-1/network_weights_20000"))



    BATCH_SIZE = 4

    train_batchfunction(
        net,
        make_make_pair(),
        steps=450000
    )

