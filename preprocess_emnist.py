import arc23.data.retrieval as rt
import arc23.transforms.image_transforms as it
from arc23.functional.core import pipe


metadata_in_path = '/media/guest/Main Storage/HDD Repositories/handrite.io/ml_pipeline/ml_models/data/emnist/emnist_byclass_train/metadata.csv'
metadata_out_path = './meta.csv'
data_in_dir = '/media/guest/Main Storage/HDD Repositories/handrite.io/ml_pipeline/ml_models/data/emnist/emnist_byclass_train/images'
data_out_dir = './emnist_preprocessed/'


def get_from_metadata():
    get = lambda path: rt.from_file(path).convert('RGB')

    def _apply(metadatum):
        filepath = data_in_dir + metadatum[0]
        return get(filepath)
    return _apply


def run():
    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
        metadata_in_path,
        cols=(COL_ID,),
    )

    with open(metadata_out_path, 'w+', newline='', encoding="utf8") as metadata_file:
        for m, metadatum in enumerate(metadata):
            filepath = metadatum[0] + '.png'
            img = pipe(get_from_metadata(), it.random_fit_to((32, 32)))(metadatum)
            filepath = metadatum[0][:-3] + 'png'  # removing jpg extension
            img.save(data_out_dir + filepath, img_format='PNG')
            print('preprocessed ', metadatum[0])
            metadata_file.write(filepath + ' ' + str(m) + '\n')


if __name__ == '__main__':
    COL_ID = 1
    run()
