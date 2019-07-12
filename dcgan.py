import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_discriminator_model(isize, nc, ndf, n_extra_layers=0):
    assert isize % 16 == 0, "isize has to be a multiple of 16"

    model = nn.Sequential(
        nn.Conv3d(nc,ndf,4,2, padding = 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    )

    i, csize, cndf = 3, isize / 2, ndf

    for _ in range(n_extra_layers):
        model.add_module(str(i),
                        nn.Conv3d(cndf, cndf, 3, 1, 1, bias=False))
        model.add_module(str(i+1),
                        nn.BatchNorm3d(cndf))
        model.add_module(str(i+2),
                        nn.LeakyReLU(0.2, inplace=True))
        i += 3

    while csize > 4:
        in_feat = cndf
        out_feat = cndf * 2
        model.add_module(str(i),
                        nn.Conv3d(in_feat, out_feat, 4, 2, 1, bias=False))
        model.add_module(str(i+1),
                        nn.BatchNorm3d(out_feat))
        model.add_module(str(i+2),
                        nn.LeakyReLU(0.2, inplace=True))
        i+=3
        cndf = cndf * 2
        csize = csize / 2

    # state size. K x 4 x 4 x 4
    model.add_module(str(i),
                    nn.Conv3d(cndf, 1, 4, 1, 0, bias=False))
    model.add_module(str(i+1), nn.Sigmoid())

    return model


def make_generator_model(isize, nz, nc, ngf, n_extra_layers=0):
    assert isize % 16 == 0, "isize has to be a multiple of 16"

    cngf, tisize = ngf//2, 4
    while tisize != isize:
        cngf = cngf * 2
        tisize = tisize * 2

    model = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose3d(nz, cngf, 4, 1, 0, bias=False),
        nn.BatchNorm3d(cngf),
        nn.ReLU(True),
    )

    i, csize = 3, 4
    while csize < isize//2:
        model.add_module(str(i),
            nn.ConvTranspose3d(cngf, cngf//2, 4, 2, 1, bias=False))
        model.add_module(str(i+1),
                        nn.BatchNorm3d(cngf//2))
        model.add_module(str(i+2),
                        nn.ReLU(True))
        i += 3
        cngf = cngf // 2
        csize = csize * 2

    # Extra layers
    for _ in range(n_extra_layers):
        model.add_module(str(i),
                        nn.Conv3d(cngf, cngf, 3, 1, 1, bias=False))
        model.add_module(str(i+1),
                        nn.BatchNorm3d(cngf))
        model.add_module(str(i+2),
                        nn.ReLU(True))
        i += 3

    model.add_module(str(i),
                    nn.ConvTranspose3d(cngf, nc, 4, 2, 1, bias=False))
    model.add_module(str(i+1), nn.Tanh())

    return model
