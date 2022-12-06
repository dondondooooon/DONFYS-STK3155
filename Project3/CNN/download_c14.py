import tensorflow as tf

# Download and extract the ChestX-ray14 dataset to the /data/chestxray14 directory
tf.keras.utils.get_file('chestxray14.tgz',
                        'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/chest-xray-pneumonia.tgz',
                        cache_dir='/data/chestxray14',
                        extract=True,
                        archive_format='tar',
                        cache_subdir='')

