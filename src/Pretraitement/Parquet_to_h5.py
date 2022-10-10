import h5py
import os
import pandas as pd
import sys


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()


data_dir = "D:\\VR3D\\VR3D_DATASET\\Out_train\\Out_train_fin_filtre\\"
# indoor3d_data_dir = os.path.join(data_dir, 'stanford_indoor3d')

# NUM_POINT = 1024
NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 4]
# data_dim = [NUM_POINT, 7]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
# filelist = os.path.join(data_dir, 'my_data_label.txt')
# = [os.path.join(data_dir, line.rstrip()) for line in open(filelist)]


output_dir = os.path.join(data_dir, 'h5_data1')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_filename_prefix = os.path.join(output_dir, 'train_19_10_2021_H5')
# output_filename_prefix = os.path.join(output_dir, 'test_eclated_from_H5_100')

output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')


def room2blocks_randomed_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                            random_sample=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    elif data_label_filename[-3:] == 'uet':
        data_label = pd.read_parquet(data_label_filename).to_numpy()
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_randomed_plus_normalized(data_label, num_point, block_size, stride,
                                                random_sample, sample_num, sample_aug)


def room2blocks_randomed_plus_normalized(data_label, num_point, block_size, stride,
                                         random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    print('\nmise en block...')
    data = data_label[:, 0:4]
    # data[:,3:6] /= 255.0
    label = data_label[:, -1].astype(np.uint8)
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    # data_batch, label_batch = room2blocks_random(data_label, num_point)
    data_batch, label_batch = room2blocks_no_random(data_label, num_point)

    new_data_batch = np.zeros((data_batch.shape[0], num_point, 7))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 4] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 5] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 6] = data_batch[b, :, 2] / max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx + block_size / 2)
        data_batch[b, :, 1] -= (miny + block_size / 2)
    new_data_batch[:, :, 0:4] = data_batch

    return new_data_batch, label_batch


def room2blocks_random(data, num_point):
    # RandomisÃ© data numpy
    np.random.shuffle(data)
    # calculer nombre de blocs de 4096 dans data
    if (data.shape[0] % num_point == 0):
        num_block = data.shape[0] / num_point
    else:
        num_block = data.shape[0] / num_point + 1

    print('num_block = ', num_block)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0

    for i in range(int(num_block)):
        # Recuper donnees x,y,z,i
        block_data_sampled = data[int(num_point) * i:int(num_point) * (i + 1), 0:4]

        # Recuper donnees x,y,z,i
        block_label_sampled = data[int(num_point) * i:int(num_point) * (i + 1), -1]

        # print('---------------------------------------------------')
        # print(block_data_sampled)
        # print('---------------------------------------------------')

        # block_label_list.append(np.expand_dims(block_label_sampled, 0))
        if block_data_sampled.shape[0] < num_point:
            sample = np.random.choice(block_data_sampled.shape[0], num_point - block_data_sampled.shape[0])
            dup_data = data[sample, 0:4]
            dup_label = data[sample, -1]
            block_data_sampled = np.concatenate([block_data_sampled, dup_data], 0)

            block_label_sampled = np.concatenate([block_label_sampled, dup_label], 0)
        # print(block_data_sampled.shape[0])
        # print(block_label_sampled.shape[0])

        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)


def room2blocks_no_random(data, num_point):
    # calculer nombre de blocs de 4096 dans data
    if (data.shape[0] % num_point == 0):
        num_block = data.shape[0] / num_point
    else:
        num_block = data.shape[0] / num_point + 1

    print('num_block = ', num_block)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0

    for i in range(int(num_block)):
        # Recuper donnees x,y,z,i
        block_data_sampled = data[int(num_point) * i:int(num_point) * (i + 1), 0:4]

        # Recuper donnees x,y,z,i
        block_label_sampled = data[int(num_point) * i:int(num_point) * (i + 1), -1]

        # print('---------------------------------------------------')
        # print(block_data_sampled)
        # print('---------------------------------------------------')

        # block_label_list.append(np.expand_dims(block_label_sampled, 0))
        if block_data_sampled.shape[0] < num_point:
            sample = np.random.choice(block_data_sampled.shape[0], num_point - block_data_sampled.shape[0])
            dup_data = data[sample, 0:4]
            dup_label = data[sample, -1]
            block_data_sampled = np.concatenate([block_data_sampled, dup_data], 0)
            block_label_sampled = np.concatenate([block_label_sampled, dup_label], 0)
        # print(block_data_sampled.shape[0])
        # print(block_label_sampled.shape[0])

        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)


# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype=np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype=np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0  # state: the next h5 file to save


def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    print("\ninsert_batch...")

    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        print('if :', buffer_size + data_size, ' <= ', h5_batch_data.shape[0])
        print("there is space... insert new batch")
        h5_batch_data[buffer_size:buffer_size + data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size + data_size] = label
        buffer_size += data_size
    else:  # not enough space
        print('if :', buffer_size + data_size, ' > ', h5_batch_data.shape[0])
        print("Space is not enough... ")
        capacity = h5_batch_data.shape[0] - buffer_size
        assert (capacity >= 0)
        if capacity > 0:
            h5_batch_data[buffer_size:buffer_size + capacity, ...] = data[0:capacity, ...]
            h5_batch_label[buffer_size:buffer_size + capacity, ...] = label[0:capacity, ...]
            # Save batch data and label to h5 file, reset buffer_size
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        print('if : last_batch = True et buffer_size = ', buffer_size, 'est >0')

        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...],
                data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0

    print('\nFINISHED.\nNext h5_index for filename output = ', h5_index, '\n')
    return


output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')

file_list = pd.read_csv("D:\VR3D\VR3D_DATASET\Out_train\File_list.txt", header=None)
file_name = [file[0] for file in file_list.to_numpy()]

sample_cnt = 0
for i, data_label_filename in enumerate(file_name):

    print('\n', data_dir + data_label_filename, '...')

    # block_size = 200000.0
    # stride = 100000.0
    data, label = room2blocks_randomed_wrapper_normalized(data_dir + data_label_filename, NUM_POINT, block_size=1.0,
                                                          stride=0.5,
                                                          random_sample=False, sample_num=None)
    print('\n Sortie Data shape = {0}, Label shape = {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename) + '\n')

    sample_cnt += data.shape[0]

    print('IS i=', i, ' === ', 'len(file_name)-1=', len(file_name) - 1)
    insert_batch(data, label, i == len(file_name) - 1)

fout_room.close()
print("Total samples: {0}".format(sample_cnt))