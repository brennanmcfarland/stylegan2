import random
from functools import partial
import os
import numpy as np
from math import sqrt

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional
from torch import multiprocessing

import arc23.data.retrieval as rt
import arc23.data.handling.handling as hd
import arc23.data.handling.dali as dt
import arc23.model.initialization as initialization
import arc23.shape_inference as sh
import arc23.callbacks as cb
import arc23.output as out
from arc23.__types import LabelledValue
from arc23.__utils import on_interval
from arc23 import bindable
from arc23.profiling import profile_cuda_memory_by_layer
from arc23.model.execution import dry_run, train, validate, test, Trainer, TrainState
from arc23.performance import \
    optimize_cuda_for_fixed_input_size, checkpoint_sequential, adapt_checkpointing
from arc23.layers.reshape import Reshape
from arc23.model import serialization
from arc23.layers.gan import GAN
from arc23.layers.normalization import PixelNorm, EqualizedLR, MinibatchStdDev, ModDemod
from arc23.layers.noise import ScaledNoise
from arc23.functional.core import ffork
from arc23.model import hooks as mh
from arc23.__utils import cache_on_interval


dbg_tensorboard_writer = out.tensorboard_writer()

# TODO: original uses batch size 32, may not matter
data_dir = '/media/guest/Main Storage/HDD Data/celebA/images/preprocessed_images/'
device = 'cuda:0'
metadata_path = './meta.csv'
# TODO: all three may not be necessary
generator_train_state_path = './generator_train_state.tar'
discriminator_train_state_path = './discriminator_train_state.tar'
gan_train_state_path = './gan_train_state.tar'


# TODO: this may not be necessary, goal is to fix CUDA initialization error
multiprocessing.set_start_method('spawn', force=True)
torch.manual_seed(1234)


def generate_random_latent_vector():
    # TODO: check if already seeded, else seed?
    #torch.seed()

    # this may actually be the correct distribution
    def _apply():
        # TODO: fix this mess, shouldn't be so tightly coupled
        if torch.rand(1).item() < .9:  # mixing probability
            return torch.normal(0, 1, size=(512,)), torch.normal(0, 1, size=(512,))
        else:
            return torch.normal(0, 1, size=(512,))
    return _apply


def make_loader(metadata):
    vector_function = generate_random_latent_vector()
    return dt.DALIIterableDataset(
        dt.dali_random_vector_to_image_pipeline(data_dir, metadata_path, vector_function),
        metadata,
        batch_size=16,
        # necessary for use of DALI's PythonFunction
        exec_async=False,
        exec_pipelined=False
    )


def wasserstein_loss():
    # given positive vs negative real vs fake labels, multiplying by discriminator output (bounded over all real #s)
    # results in a "realness" score over all real #s
    def _apply(outputs, gtruth):
        return torch.mean(torch.mul(outputs, gtruth))
    return _apply


def nonsaturating_logistic_loss():
    # TODO: remove this, for debugging
    loss_writer = out.tensorboard_writer()
    # outputs = discriminator output, gtruth = label (so both 1 dimensional)
    def _apply(outputs, gtruth, *args, **kwargs):
        value = torch.mean(torch.nn.functional.softplus(outputs))
        loss_writer.add_scalar('unregularized_loss', value)
        return value
    return _apply


# lmbda: effect of gradient clipping term (defaults to canonical value)
def wasserstein_gp(lmbda=10):
    def _apply(outputs, gtruth):
        return lmbda * (torch.norm(outputs) - 1) ** 2
    return _apply


# r1 gradient regularization penalty
def r1_gp(gamma=10.0):
    # discriminator outputs, discriminator gtruth labels, real images or None if this batch is fake
    def _apply(outputs, gtruth, images):
        if images is None:
            return 0
        else:
            o = outputs.sum()
            g = torch.autograd.grad(o, inputs=images, retain_graph=True)[0]
            g = g.view(g.size(0), -1)
            g = g.norm(2, dim=1) ** 2
            g = g.mean()
            gp = gamma / 2 * g # torch.norm(g) ** 2
            print('r1 gp: ', gp)
            return gp
    return _apply


# TODO: minibatch_divisor should be 2, but setting it higher breaks the net because it expects constant batch size; fix
def pathreg_gp(latent_func, generator, minibatch_divisor=1, a_decay=.01, pl_weight=2.0): # pulled these values from the example
    # TODO: fix
    batch_size = 16
    # use a smaller batch size for efficiency
    pl_batch_size = batch_size // minibatch_divisor
    latent_size = (pl_batch_size, 512) # TODO: fix
    a = [0] # TODO: this is a hack for mutability, fix it

    # TODO: may need to zero gradients each time, but I don't think that's required for torch.autograd.grad
    def _apply(outputs, gtruth):
        z = torch.normal(0, 1, size=latent_size, device=device)
        w = generator(z)
        y = torch.normal(0, 1, size=w.size(), device=device) / (torch.prod(torch.tensor(w.size(), dtype=torch.uint8)[2:]) ** .5)
        wy = torch.dot(w.view(-1), y.view(-1))
        (jacobian,) = torch.autograd.grad(wy, w)
        path_length = torch.norm(jacobian)
        a[0] += a_decay * (path_length.mean() - a[0])
        pl_gp = (path_length - a[0]) ** 2
        print('pl gp: ', pl_gp)
        return pl_gp * pl_weight
    return _apply


def combine_loss_terms(*loss_terms):

    # TODO: remove this and make it nice
    def _dbg(x, outputs, gtruth, *args, **kwargs):
        o = x(outputs, gtruth, *args, **kwargs)
        if hasattr(x, 'func'):
            dbg_tensorboard_writer.add_scalar(x.func.__name__, o)
        else:
            dbg_tensorboard_writer.add_scalar(x.__name__, o)
        return o

    def _apply(outputs, gtruth, *args, **kwargs):
        #return sum(loss_term(outputs, gtruth) for loss_32term in loss_terms)
        return sum(_dbg(loss_term, outputs, gtruth, *args, **kwargs) for loss_term in loss_terms)
    return _apply


def generator_loss(generator):
    # return nonsaturating_logistic_loss()
    #return combine_loss_terms(wasserstein_loss(), cache_on_interval(pathreg_gp(generate_random_latent_vector(), generator), 16))
    return combine_loss_terms(nonsaturating_logistic_loss(), cache_on_interval(pathreg_gp(generate_random_latent_vector(), generator), 16))


def discriminator_loss():
    #return combine_loss_terms(wasserstein_loss(), wasserstein_gp(), cache_on_interval(r1_gp(), 16))
    return nonsaturating_logistic_loss()
    # TODO: uncomment the below to restore loss regularization
    # return combine_loss_terms(nonsaturating_logistic_loss(), cache_on_interval(r1_gp(), 16))


# TODO: datatypes?
def make_real_label_batch(inputs, device):
    return torch.ones(inputs.size()[:1], dtype=torch.float32, device=device)


def make_fake_label_batch(inputs, device):
    return torch.ones(inputs.size()[:1], dtype=torch.float32, device=device) * -1


class LearnedConstant(nn.Module):
    def __init__(self, size, batch_size, device, dtype=torch.float32):
        super(LearnedConstant, self).__init__()
        self.batch_size = batch_size
        self.constant = nn.Parameter(torch.normal(0, 1, size=size, dtype=dtype, device=device))

    def forward(self):
        return self.constant.expand(self.batch_size, *(-1 for _ in self.constant.size()))


class Bias(nn.Module):
    def __init__(self, size):
        super(Bias, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x + self.bias


# TODO: generalize
class PaddedConv2D(nn.Module):

    def __init__(self, *args, **kwargs):
        super(PaddedConv2D, self).__init__()
        self.sublayer = nn.Conv2d(*args, **kwargs)
        self.weight = self.sublayer.weight # TODO: fix this, terribly hacky

    # TODO: nn.Conv2D documentation has formula for how to get this ahead of time which should be much more efficient, do that
    # TODO: may be able to just bake this into the padding param instead of a whole nother layer
    def forward(self, x):
        y = self.sublayer(x)
        diff = [i - j for i, j in zip(x.size(), y.size())]
        y = nn.functional.pad(y, [diff[-2]//2, diff[-2]//2, diff[-1]//2, diff[-1]//2], 'reflect')
        return y


# algorithm based on the paper for getting the # features to set for a given layer
def nf(stage):
    fmap_base = 16 * 2 ** 10
    fmap_decay = 1.0
    fmap_min = 1
    fmap_max = 512
    return int(np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max))


# TODO: obviously clean this up/make it nice/finish it and make this implementation use it
class StyleGAN2(nn.Module):
    def __init__(self, in_features, out_features, synthesizer_blocks, mapper, constant_size, device):
        super(StyleGAN2, self).__init__()
        print(len(synthesizer_blocks))
        # TODO: abstract out these vars
        resolution = 32
        log_resolution = int(np.log2(resolution))
        self.input_size = 32 // (2 ** len(synthesizer_blocks)) # TODO: this should be 4 and the # blocks determines the final resolution (which should be enforced as same in generator and discriminator)
        self.constant = EqualizedLR(LearnedConstant((512, self.input_size, self.input_size), 16, device=device))
        self.mapper = mapper
        # TODO: figure out why *32
        self.maps_to_styles = nn.ModuleList((EqualizedLR(nn.Linear(512, 512*32)).to(device),) + tuple(
            EqualizedLR(nn.Linear(512, nf(r)*32)).to(device) for r in range(2, log_resolution)))
        self.synthesizer_blocks = nn.ModuleList(
            ModDemod(synthesizer_block, map_to_style).to(device) # TODO: I don't think we need to equalize again, but double check
            for synthesizer_block, map_to_style in zip(synthesizer_blocks, self.maps_to_styles)
        )
        # TODO: fix hardcoding size
        self.biases = nn.ModuleList([EqualizedLR(Bias((16, 512, self.input_size * (2 ** s), self.input_size * (2 ** s)))).to(device) for s in range(len(synthesizer_blocks))])
        self.noise_blocks = nn.ModuleList([EqualizedLR(ScaledNoise(512)).to(device) for _ in synthesizer_blocks])
        self.final_layer = EqualizedLR(PaddedConv2D(512, 3, 3)).to(device)
        self.trgbs = nn.ModuleList([EqualizedLR(PaddedConv2D(512, 3, 1)).to(device) for _ in synthesizer_blocks])
        self.mix_styles = False
        # TODO: make sure this works right
        for layer in (*self.mapper.children(), *self.synthesizer_blocks, *self.maps_to_styles, self.final_layer, *self.trgbs):
            if hasattr(layer, 'weight'):
                torch.nn.init.normal_(layer.weight)
            if hasattr(layer, 'bias'):
                torch.nn.init.zeros_(layer.bias)
        for map_to_style in self.maps_to_styles:
            torch.nn.init.ones_(map_to_style.bias)

    def forward(self, x):
        if self.mix_styles:
            x, x2 = x
            mixing_point = torch.rand(1).item() * len(self.synthesizer_blocks) // len(self.synthesizer_blocks)
        block_output = self.constant()
        tensor_output = torch.zeros((16, 3, self.input_size, self.input_size), device="cuda:0")
        for b, (synthesizer_block, map_to_style, noise_block, bias, trgb) in enumerate(zip(self.synthesizer_blocks, self.maps_to_styles, self.noise_blocks, self.biases, self.trgbs)):
        # for b, (synthesizer_block, map_to_style, bias, trgb) in enumerate(
        #         zip(self.synthesizer_blocks, self.maps_to_styles, self.biases, self.trgbs)):
            if self.mix_styles and b > mixing_point:
                x = x2
            y = synthesizer_block(block_output, map_to_style(self.mapper(x)))
            y = bias(y)
            noise_size = list(y.size())
            noise_size[1] = 1
            y = noise_block(y, torch.normal(mean=torch.zeros(size=noise_size, device=device), std=torch.ones(size=noise_size, device=device)))
            tensor_output += trgb(y)
            tensor_output = nn.functional.interpolate(tensor_output, scale_factor=2, mode='bilinear')
            block_output = nn.functional.interpolate(y, scale_factor=2, mode='bilinear') # upsampling
            #print(block_output.data.cpu().numpy())
        block_output = self.final_layer(block_output)
        tensor_output += block_output
        # tensor_output = block_output

        self.mix_styles = False
        return tensor_output


def define_synthesizer_blocks():
    # TODO: abstract out these vars
    resolution = 32
    log_resolution = int(np.log2(resolution))
    # TODO: get rid of input_size
    return [sh.Input(input_size=512),
            *(
                lambda i: EqualizedLR(PaddedConv2D(i, nf(r-1), 3, bias=False))
              for r in range(log_resolution, 2, -1))
    ]


def define_mapper_layers():
    return [
        sh.Input(input_size=512),
        lambda i: PixelNorm(),
        *(lambda i: EqualizedLR(nn.Linear(i, 512)) for _ in range(8)),
    ]


def define_generator_layers(loader):

    return StyleGAN2(
        512,
        32,
        [EqualizedLR(layer) for layer in sh.infer_shapes(define_synthesizer_blocks(), loader)],
        initialization.from_iterable([EqualizedLR(layer) for layer in sh.infer_shapes(define_mapper_layers(), None)]),
        (nf(1), 0, 0), # TODO: this should be fed in, or internalized, or something instead of the 0s being overwritten
        device="cuda:0"
    )


# TODO: make sure equalized lr still applied even to wrapped layers
# TODO: generalize to arbitrary func to apply to residual?
class DownsampledResidual(nn.Module):
    def __init__(self, sublayer, residual_transform):
        super(DownsampledResidual, self).__init__()
        self.sublayer = sublayer
        self.pool = nn.AvgPool2d(2)
        self.residual_transform = residual_transform

    def forward(self, x):
        y = self.sublayer(x)
        y = nn.functional.interpolate(y, scale_factor=.5, mode='bilinear')
        x = self.residual_transform(x)
        x = nn.functional.interpolate(x, scale_factor=.5, mode='bilinear')
        return (x + y) / np.sqrt(2)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def define_discriminator_layers():
    # TODO: abstract these constants/variables out
    resolution = 32 # TODO: abstract out
    log_resolution = int(np.log2(resolution))
    layers = [
        sh.Input(),
        lambda i: EqualizedLR(PaddedConv2D(3, nf(log_resolution-1), 1)),
        *(lambda i: DownsampledResidual(nn.Sequential(
            EqualizedLR(PaddedConv2D(nf(r-1), nf(r-1), 3)),
            nn.LeakyReLU(.05),
            EqualizedLR(PaddedConv2D(nf(r-1), nf(r-2), 3)),
            nn.LeakyReLU(.05),
        ), EqualizedLR(PaddedConv2D(nf(r-1), nf(r-2), 1))) for r in range(log_resolution, 2, -1)),
        lambda i: MinibatchStdDev(),
        lambda i: PaddedConv2D(nf(2)+1, nf(1), 3),  # +1 from extra minibatch stddev channel
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.Flatten(),
        lambda i: nn.Linear(i, nf(0)),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.Linear(i, i),
    ]
    for layer in layers:
        if hasattr(layer, 'weight'):
            torch.nn.init.normal_(layer.weight)
        if hasattr(layer, 'bias'):
            torch.nn.init.zeros_(layer.bias)
    return layers


def build_net(layer_func, loader):
    layers = layer_func()
    layers = sh.infer_shapes(layers, loader)
    layers = [EqualizedLR(layer) for layer in layers]
    net = initialization.from_iterable(layers)
    net = net.to(device)
    return net


g_loss_last = torch.ones((1,))
d_loss_last = torch.ones((1,))
d_loss_real_last = torch.ones((1,))
d_loss_fake_last = torch.ones((1,))

# TODO: can we just abstract this to the loss func and use normal train step func? or maybe use a wrapper train step
# TODO: func? may require some modification/generalization of the original step func
# TODO: this is a critical function, any way to make it more efficient?
def gan_train_step(
        generator,
        discriminator,
        g_trainer,
        d_trainer,
        real_label_func,
        fake_label_func,
        device=None,
        g_per_d_train_ratio=1,
        # TODO: change back to 1?
        d_per_g_train_ratio=5,
):
    if g_per_d_train_ratio != 1 and d_per_g_train_ratio != 1:
        raise Exception(
            "Cannot specify values for both generator per discriminator ratio and discriminator per generator ratio"
        )

    step = [0]  # array allows mutability TODO: better solution for this

    # the whole GAN is used for training on both real and fake examples, so the whole thing needs to fit in memory
    def _apply(fake_inputs, real_outputs):
        # TODO: if mixing styles, fix this, shouldn't be so tightly coupled
        mix_styles = not (fake_inputs.size()[0] == 16 and fake_inputs.size()[1] == 512)
        if mix_styles:
            z1, z2 = fake_inputs
        else:
            z1 = fake_inputs

        generator.mix_styles = mix_styles

        global d_loss_last
        global g_loss_last
        global d_loss_real_last
        global d_loss_fake_last

        g_loss, d_loss = g_loss_last, d_loss_last
        d_loss_real, d_loss_fake = d_loss_real_last, d_loss_fake_last

        if mix_styles:
            z1, z2, real_outputs = z1.to(device, non_blocking=True), z2.to(device, non_blocking=True),\
                                   real_outputs.to(device, non_blocking=True)
            fake_outputs = generator(z1, z2)
        else:
            z1, real_outputs = z1.to(device, non_blocking=True), real_outputs.to(device, non_blocking=True)
            fake_outputs = generator(z1)
        # make sure real_outputs is marked as required_grad so it can be used in loss function gradient calculation
        if real_outputs.is_leaf:
            real_outputs.requires_grad = True

        real_labels = real_label_func(real_outputs, device)
        fake_labels = fake_label_func(fake_outputs, device)

        d_fake_outputs = torch.squeeze(discriminator(fake_outputs), -1)
        # print(fake_outputs.data.cpu().numpy())

        if step[0] % d_per_g_train_ratio == 0:
            d_trainer.optimizer.zero_grad()  # reset the gradients to zero
            d_real_outputs = torch.squeeze(discriminator(real_outputs), -1)
            d_loss_real = d_trainer.loss(-d_real_outputs, real_labels, real_outputs)
            d_fake_outputs_detached = d_fake_outputs.clone() # TODO: this used to be detach(), is this right?
            d_loss_fake = d_trainer.loss(d_fake_outputs_detached, fake_labels, None)
            d_loss = d_loss_real + d_loss_fake

            train_state = []
            for param in generator.parameters():
                train_state.append(param.requires_grad)
                param.requires_grad = False
            # generator.train(False)

            d_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0) # gradient clipping not part of the original architecture but might help
            d_trainer.optimizer.step()

            for param, ts in zip(generator.parameters(), train_state):
                param.requires_grad = ts

            # generator.train(True)

            d_loss_last = d_loss
            d_loss_real_last, d_loss_fake_last = d_loss_real, d_loss_fake

        if step[0] % g_per_d_train_ratio == 0:
            train_state = []
            for param in discriminator.parameters():
                train_state.append(param.requires_grad)
                param.requires_grad = False
            # discriminator.train(False)

            g_trainer.optimizer.zero_grad()  # reset the gradients to zero
            g_loss = g_trainer.loss(-d_fake_outputs, real_labels)
            g_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            g_trainer.optimizer.step() # gradient clipping not part of the original architecture but might help

            for param, ts in zip(discriminator.parameters(), train_state):
                param.requires_grad = ts
            # discriminator.train(True)

            g_loss_last = g_loss

        step[0] += 1

        return {'generator': g_loss, 'discriminator': d_loss, 'd_real': d_loss_real, 'd_fake': d_loss_fake}
    return _apply


def get_metadata(quiet=False):
    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
        metadata_path,
        cols=(COL_ID, COL_PATH),
        delimiter=' ',
    )

    # shuffle at beginning to get random sampling for train, test and validation datasets
    random.shuffle(metadata)

    if not quiet:
        print(class_to_index)
        print(index_to_class)

    return metadata


def make_trainers(generator, discriminator):
    g_loss, d_loss = generator_loss(generator), discriminator_loss()
    lr = .001
    map_params, nonmap_params = [], []
    map_param_names = [n for n, p in generator.mapper.named_parameters()]
    # TODO: clean up this mess
    for n, p in generator.named_parameters():
        if n in map_param_names:
            map_params.append(p)
        else:
            nonmap_params.append(p)
    # [p for n, p in generator.named_parameters() if n not in [n for n, p in generator.mapper.named_parameters()]]
    g_optimizer, d_optimizer = torch.optim.Adam(
        # generator.parameters()
        [
        {'params': nonmap_params, 'lr': lr},
        {'params': map_params, 'lr': lr * .01},
            # {'params': generator.parameters()},
    ]
        , betas=(0, .99), eps=1E-8),\
                               torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0, .99), eps=1E-8)
    g_trainer = Trainer(g_optimizer, g_loss)
    d_trainer = Trainer(d_optimizer, d_loss)
    return g_trainer, d_trainer


# TODO: abstract GAN out?
def run():
    print('preparing metadata...')
    metadata = get_metadata()

    # TODO: remove
    torch.autograd.set_detect_anomaly(True)

    if not torch.cuda.is_available():
        raise Exception("Training GANs on CPU?  Aint nobody got time for that.")

    print('building main loader')

    loader = make_loader(metadata[:len(metadata) // 16])

    loader.build()

    print('creating net builder loader')

    net_builder_loader = \
        dt.DALIIterableDataset(
                dt.dali_image_to_vector_pipeline(data_dir, metadata_path, lambda: torch.tensor([0] * 16, device='cpu')),
                [0] * loader.len_metadata,
                loader.batch_size,
                # necessary for use of DALI's PythonFunction
                exec_async=False,
                exec_pipelined=False
            )

    print('building net builder loader')

    net_builder_loader.build()

    print('building networks')

    # TODO: loader for discriminator? it's only for shape inference, so we may want to abstract out/make not needed
    generator, discriminator =\
        define_generator_layers(loader).to(device),\
        build_net(
            define_discriminator_layers,
            net_builder_loader
        )


    g_trainer, d_trainer = make_trainers(generator, discriminator)

    # TODO: metrics
    metrics = []

    # TODO: saving automatic checkpionting for later because it needs rework to work with generator and discriminator being passed in
    # TODO: but both at least having their weights in memory (for training)
    # print('checkpointing')

    # TODO: smarter checkpointing system that can take both at once, and checkpoint separately for when training generator vs discriminator?

    # TODO: make an object instead of just a tuple, generalized, for generator vs discriminator (vs gan optionally?)
    # generator = adapt_checkpointing(
    #     checkpoint_sequential,
    #     lambda n: dry_run(n, loader, g_trainer, partial(gan_train_step, **dict(zip(('g_trainer', 'd_trainer'), make_trainers(generator, discriminator)))), device=device)(),
    #     generator
    # )
    # discriminator = adapt_checkpointing(
    #     checkpoint_sequential,
    #     lambda n: dry_run(n, loader, d_trainer, **dict(zip(('g_trainer', 'd_trainer'), make_trainers(generator, discriminator))), device=device)(),
    #     discriminator
    # )

    print('optimizing CUDA')

    # TODO: uncomment, it's useful but not initially necessary
    # profile_cuda_memory_by_layer(
    #     net,
    #     dry_run(net, loader, trainer, gan_train_step, device=device),
    #     device=device
    # )
    optimize_cuda_for_fixed_input_size()

    gan = {'generator': generator, 'discriminator': discriminator}
    gan_trainer = {'g_trainer': g_trainer, 'd_trainer': d_trainer}

    train_state = TrainState()

    # if we have all the save files, continue from there
    # TODO: do we need all three train states?
    # TODO: uncomment this once we're ready to save + load train state
    # if os.path.isfile(generator_train_state_path)\
    #         or os.path.isfile(discriminator_train_state_path)\
    #         or os.path.isfile(gan_train_state_path):
    #     generator, g_trainer, train_state = serialization.load_train_state(generator_train_state_path)(generator,
    #                                                                                                    g_trainer,
    #                                                                                                    train_state)()
    #     discriminator, d_trainer, train_state = serialization.load_train_state(discriminator_train_state_path)(discriminator,
    #                                                                                                    d_trainer,
    #                                                                                                    train_state)()
    #     gan, gan_trainer, train_state = serialization.load_train_state(gan_train_state_path)(gan,
    #                                                                                                    gan_trainer,
    #                                                                                                    train_state)()

    # TODO: generalize, will need this elsewhere
    def loss_wrapper(loss):
        return LabelledValue(loss.label, {v: loss.value[v].item() for v in loss.value.keys()})

    def image_func(epoch):
        with torch.no_grad():
            return LabelledValue(str(epoch), generator(torch.normal(0, 1, size=(16, 512,)).to(device))[0]) # TODO: use the vectro func and get rid of index if possible

    tensorboard_writer = out.tensorboard_writer()
    # tensorboard_writer.add_graph(generator, (generate_random_latent_vector()().expand(16, -1).to(device)))
    # print("printed graph to writer")
    # tensorboard_writer.add_graph(discriminator, ())

    def histograms(epoch):
        guid = 0
        for g_module in generator.modules():
            tag = 'generator' + str(type(g_module))
            if hasattr(g_module, 'weight'):
                tensorboard_writer.add_histogram(tag+' weight'+str(guid), g_module.weight, epoch)
            if hasattr(g_module, 'weight_orig'):
                tensorboard_writer.add_histogram(tag + ' weight_orig'+str(guid), g_module.weight_orig, epoch)
            guid += 1
        for d_module in discriminator.modules():
            tag = 'discriminator' + str(type(g_module))
            if hasattr(d_module, 'weight'):
                tensorboard_writer.add_histogram(tag+' weight'+str(guid), d_module.weight, epoch)
            if hasattr(d_module, 'weight_orig'):
                tensorboard_writer.add_histogram(tag + ' weight_orig'+str(guid), d_module.weight_orig, epoch)
            guid += 1
        for g_param in generator.parameters():
            tag = 'generator_param' + str(type(g_param))
            tensorboard_writer.add_histogram(tag+str(guid), g_param, epoch)
            guid += 1

    callbacks = {
        "on_step": [
            lambda steps_per_epoch: on_interval(
                out.print_with_step(bindable.wrap_return(cb.loss(), loss_wrapper))(steps_per_epoch),
                8
            ),
            out.scalar_to_tensorboard(
                bindable.wrap_return(
                    cb.loss(),
                    lambda loss: LabelledValue(loss.label + '_generator', loss.value['generator'])
                ), tensorboard_writer),
            out.scalar_to_tensorboard(
                bindable.wrap_return(
                    cb.loss(),
                    lambda loss: LabelledValue(loss.label + '_discriminator', loss.value['discriminator'])
                ), tensorboard_writer),
            out.scalar_to_tensorboard(
                bindable.wrap_return(
                    cb.loss(),
                    lambda loss: LabelledValue(loss.label + 'd_real', loss.value['d_real'])
                ), tensorboard_writer),
            out.scalar_to_tensorboard(
                bindable.wrap_return(
                    cb.loss(),
                    lambda loss: LabelledValue(loss.label + 'd_fake', loss.value['d_fake'])
                ), tensorboard_writer),
        ],
        "on_epoch": [
            out.image_to_tensorboard(image_func, tensorboard_writer),
            out.image_to_tensorboard(lambda epoch: LabelledValue('real' + str(epoch), next(iter(loader))['labels'][0]), tensorboard_writer),
            # out.print_tables(
            #     cb.layer_stats(
            #         generator,
            #         dry_run(
            #             generator,
            #             loader,
            #             g_trainer,
            #             lambda net, trn, device: gan_train_step(generator, discriminator, g_trainer, d_trainer, make_real_label_batch, make_fake_label_batch, device=device),
            #             device=device
            #         ),
            #         [
            #             mh.weight_stats_hook((torch.var_mean,)),
            #             mh.output_stats_hook((torch.var_mean,)),
            #             mh.grad_stats_hook((torch.var_mean,)),
            #         ]
            #     ), titles=["WEIGHT STATS", "OUTPUT_STATS", "GRADIENT STATS"], headers=["Layer", "Value"]
            # ),
            # out.print_tables(
            #     cb.layer_stats(
            #         discriminator,
            #         dry_run(
            #             discriminator,
            #             loader,
            #             d_trainer,
            #             lambda net, trn, device: gan_train_step(generator, discriminator, g_trainer, d_trainer,
            #                                                     make_real_label_batch, make_fake_label_batch,
            #                                                     device=device),
            #             device=device
            #         ),
            #         [
            #             mh.weight_stats_hook((torch.var_mean,)),
            #             mh.output_stats_hook((torch.var_mean,)),
            #             mh.grad_stats_hook((torch.var_mean,)),
            #         ]
            #     ), titles=["WEIGHT STATS", "OUTPUT_STATS", "GRADIENT STATS"], headers=["Layer", "Value"]
            # ),
            lambda steps_per_epoch: lambda epoch: histograms(epoch)
        ],
        "on_epoch_end": [
            lambda steps_per_epoch: lambda epoch: serialization.save_train_state(generator_train_state_path)(generator,
                                                                                                       g_trainer,
                                                                                                       train_state),
            lambda steps_per_epoch: lambda epoch: serialization.save_train_state(discriminator_train_state_path)(discriminator,
                                                                                                   d_trainer,
                                                                                                   train_state),
            lambda steps_per_epoch: lambda epoch: serialization.save_train_state(gan_train_state_path)(gan,
                                                                                             gan_trainer,
                                                                                             train_state)
        ]
    }

    # dry_run(generator, loader, g_trainer, lambda net, trn, device: lambda inputs, labels: tensorboard_writer.add_graph(generator, inputs.to(device)), device=device)()
    # dry_run(discriminator, loader, d_trainer,
    #         lambda net, trn, device: lambda inputs, labels: tensorboard_writer.add_graph(discriminator, inputs.to(device)),
    #         device=device)()
    def print_modules(module, depth):
        print(module, ', depth=', depth)
        if len(list(module.children())) > 0:
            for m in module.children():
                print_modules(m, depth+1)

    print("GENERATOR MODULES:")
    print_modules(generator, 0)
    print("DISCRIMINATOR MODULES:")
    print_modules(discriminator, 0)

    print('about to train')
    train(
        gan,
        loader,
        gan_trainer,
        callbacks,
        device,
        gan_train_step(
            generator,
            discriminator,
            g_trainer,
            d_trainer,
            make_real_label_batch,
            make_fake_label_batch,
            device
        ),
        train_state,
        300
    )


if __name__ == '__main__':
    COL_ID = 1
    COL_PATH = 0
    run()
