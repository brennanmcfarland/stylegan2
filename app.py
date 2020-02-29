import random
import torch.cuda

import arc23.data.retrieval as rt
import arc23.data.handling.handling as hd
import arc23.model.initialization as initialization
import arc23.shape_inference as sh
from arc23.profiling import profile_cuda_memory_by_layer
from arc23.model.execution import dry_run, train, validate, test, train_step, Trainer
from arc23.performance import \
    optimize_cuda_for_fixed_input_size, checkpoint_sequential, adapt_checkpointing


metadata_path = '/media/guest/Main Storage/HDD Data/CMAopenaccess/data.csv'
device = 'cuda:0'
meta_out_path = './meta.csv'


def make_loader(metadata):
    hd.write_transformed_metadata_to_file(
        metadata, meta_out_path, lambda m: m[0] + '.png\n'
    )
    raise Exception("TODO: implement DALI dataset")


def define_layers():
    raise Exception("TODO: implement layers/model")


def get_metadata(quiet=False):
    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
        metadata_path,
        cols=(COL_ID, COL_IMG_WEB)
    )
    len_metadata = 31149  # TODO: either the dataset is corrupted/in a different format after this point or the endpoint was down last I tried
    metadata = metadata[:len_metadata]

    # shuffle at beginning to get random sampling for train, test and validation datasets
    random.shuffle(metadata)

    if not quiet:
        print(class_to_index)
        print(index_to_class)

    return metadata


def run():
    print('preparing metadata...')
    metadata = get_metadata()

    loader = make_loader()

    loader.build()

    if not torch.cuda.is_available():
        raise Exception("Training GANs on CPU?  Aint nobody got time for that.")

    layers = define_layers()
    net = initialization.from_iterable(sh.infer_shapes(layers, loader))
    net = net.to(device)

    # TODO: GAN loss and metrics
    loss_func = None
    optimizer = torch.optim.Adam(net.parameters())
    trainer = Trainer(optimizer, loss_func)
    metrics = []

    net = adapt_checkpointing(
        checkpoint_sequential,
        lambda n: dry_run(n, loader, trainer, train_step, device=device)(),
        net
    )

    profile_cuda_memory_by_layer(
        net,
        dry_run(net, loader, trainer, train_step, device=device),
        device=device
    )
    optimize_cuda_for_fixed_input_size()

    # TODO: callbacks
    callbacks = []

    train(net, loader, trainer, callbacks, device, 10)


if __name__ == '__main__':
    COL_ID = 0
    COL_IMG_WEB = -4  # lower resolution makes for smaller dataset/faster training
    run()
