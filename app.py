import random
from functools import partial
import os

import torch
import torch.cuda
import torch.nn as nn
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
from arc23.functional.core import ffork


data_dir = '/media/guest/Main Storage/HDD Data/celebA/images/preprocessed_images/'
device = 'cuda:0'
metadata_path = './meta.csv'
# TODO: all three may not be necessary
generator_train_state_path = './generator_train_state.tar'
discriminator_train_state_path = './discriminator_train_state.tar'
gan_train_state_path = './gan_train_state.tar'


# TODO: this may not be necessary, goal is to fix CUDA initialization error, which is still a problem
#  see https://github.com/pytorch/pytorch/issues/2517  https://github.com/pytorch/pytorch/issues/30900
#  it's initializing CUDA multiple times, which is apparently not allowed, but how to fix?
#  when fixed, remove CUDA_LAUNCH_BLOCKIING env var from run configuration
multiprocessing.set_start_method('spawn', force=True)
# TODO: did this make it work? or it might still be failing intermittently
torch.manual_seed(1234)


def generate_random_latent_vector():
    # TODO: check if already seeded, else seed?
    #torch.seed()

    # TODO: make it like the paper
    def _apply():
        # TODO: fix
        return torch.normal(0, 1, size=(256,))
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


def generator_loss(discriminator):
    return nn.BCEWithLogitsLoss()  # with logits so it can accept all real numbers, not just [0, 1]
    # def _apply(outputs, _):
    #     raise NotImplementedError("TODO: implement")
    # return _apply


def discriminator_loss():
    return nn.BCEWithLogitsLoss() # with logits so it can accept all real numbers, not just [0, 1]
    # def _apply(outputs, gtruth):
    #     raise NotImplementedError("TODO: implement")
    # return _apply


# TODO: start small to make sure it works (decently well), then work with larger images
def define_generator_layers():
    # TODO: define a legit model
    return [
        # TODO: fix this stupid reflection thing, make a layer for it?
        sh.Input(),
        lambda i: Reshape(1, 16, 16),
        # TODO: shape inference is screwed up because it's inferring from the flat input before passin gto the convolution and reshape does nothing,
        # TODO: so this extra input is neccessary; should make it not be so though if possible
        lambda i: sh.Input(input_size=1),
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.ReflectionPad2d((2, 1, 2, 1)),
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.ReflectionPad2d((1, 2, 1, 2)),
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.ReflectionPad2d((2, 1, 2, 1)),
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.ReflectionPad2d(1),
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.Upsample(scale_factor=2),
        lambda i: nn.Conv2d(i, 3, 3),
        lambda i: nn.LeakyReLU(.05)
    ]


def define_discriminator_layers():
    # TODO: define a legit model
    return [
        sh.Input(),
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.Flatten(),
        lambda i: nn.Linear(i, 2048),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.Linear(i, 1024),
        lambda i: nn.LeakyReLU(.05),
        lambda i: nn.Linear(i, 1),
        lambda i: nn.LeakyReLU(.05)
    ]


def build_net(layer_func, loader):
    layers = layer_func()
    net = initialization.from_iterable(sh.infer_shapes(layers, loader))
    net = net.to(device)
    return net


# TODO: **************************************************this area probably needs the most attention next, then the layer definition functions***********************************************
# TODO: can we just abstract this to the loss func and use normal train step func? or maybe use a wrapper train step
# TODO: func? may require some modification/generalization of the original step func
def gan_train_step(generator, discriminator, g_trainer, d_trainer, device=None):
    def _apply(fake_inputs, real_outputs):

        # TODO: would some of this be more efficient in place?
        # TODO: there's probably also a lot of repetition to be gotten rid of
        # TODO: split generator vs discriminator steps, have share example when they both run?
        # TODO: also add support for doing m generator steps every n discriminator steps
        # TODO: detach() may be useful/needed here sometimes
        fake_inputs, real_outputs = fake_inputs.to(device, non_blocking=True), real_outputs.to(device, non_blocking=True)

        fake_outputs = generator(fake_inputs)

        # TODO: soft/noisy labels?
        # TODO: should these be on CPU?
        # TODO: datatypes?
        real_labels = torch.ones(real_outputs.size()[:1], dtype=torch.float32, device=device)
        fake_labels = torch.zeros(fake_outputs.size()[:1], dtype=torch.float32, device=device)

        g_trainer.optimizer.zero_grad()  # reset the gradients to zero
        d_trainer.optimizer.zero_grad()  # reset the gradients to zero

        d_real_outputs = torch.squeeze(discriminator(real_outputs), -1)
        d_fake_outputs = torch.squeeze(discriminator(fake_outputs.detach()), -1)

        d_loss = d_trainer.loss(d_real_outputs, real_labels) + d_trainer.loss(d_fake_outputs, fake_labels)

        d_loss.backward(retain_graph=True) # TODO: may be able to get rid of retain_graph and thus free some memory if generator and discriminator histories can be separated

        d_trainer.optimizer.step()

        # TODO: not sure if this is correct, or if we should be detaching somewhere
        g_loss = g_trainer.loss(d_fake_outputs, real_labels)

        g_loss.backward()

        g_trainer.optimizer.step()

        return {'generator': g_loss, 'discriminator': d_loss}
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
    g_loss, d_loss = generator_loss(discriminator), discriminator_loss()
    g_optimizer, d_optimizer = torch.optim.Adam(generator.parameters()), torch.optim.Adam(discriminator.parameters())
    g_trainer = Trainer(g_optimizer, g_loss)
    d_trainer = Trainer(d_optimizer, d_loss)
    return g_trainer, d_trainer


# TODO: abstract GAN out?
def run():
    print('preparing metadata...')
    metadata = get_metadata()

    if not torch.cuda.is_available():
        raise Exception("Training GANs on CPU?  Aint nobody got time for that.")

    print('building main loader')

    loader = make_loader(metadata)

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
        build_net(define_generator_layers, loader),\
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
            return LabelledValue(str(epoch), generator(torch.normal(0, 1, size=(1, 256,)).to(device))[0]) # TODO: use the vectro func and get rid of index if possible

    tensorboard_writer = out.tensorboard_writer()

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
        ],
        "on_epoch": [
            out.image_to_tensorboard(image_func, tensorboard_writer),
        ],
        "on_epoch_end": [
            lambda steps_per_epoch: serialization.save_train_state(generator_train_state_path)(generator,
                                                                                                       g_trainer,
                                                                                                       train_state),
            lambda steps_per_epoch: serialization.save_train_state(discriminator_train_state_path)(discriminator,
                                                                                                   d_trainer,
                                                                                                   train_state),
            lambda steps_per_epoch: serialization.save_train_state(gan_train_state_path)(gan,
                                                                                             gan_trainer,
                                                                                             train_state)
        ]
    }

    # TODO: obviously refactor once working
    print('about to train')
    train(gan, loader, gan_trainer, callbacks, device, gan_train_step(generator, discriminator, g_trainer, d_trainer, device), train_state, 10)


if __name__ == '__main__':
    COL_ID = 1
    COL_PATH = 0
    run()
