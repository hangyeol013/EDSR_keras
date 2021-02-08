

import h5py as h5

hf1 = h5.File('training_sample/train_DIV2Kaug_scale4.h5')
hf2 = h5.File('training_sample/train_DIV2Kaug_scale4_3.h5')
hf3 = h5.File('training_sample/train_DIV2Kaug_scale4_4.h5')

hf = h5.File('training_sample/train_DIV2Kaug_scale4_merge.h5')


hf1_data = hf1['data']
hf2_data = hf2['data']
hf3_data = hf3['data']

print(hf1_data)
print(hf2_data)
print(hf3_data)

data_shape = hf1['data'].shape[2]
data_shape


hf1_label = hf1['label']
hf2_label = hf2['label']
hf3_label = hf3['label']

print(hf1_label)
print(hf2_label)
print(hf3_label)

label_shape = hf1['label'].shape[2]
print(label_shape)

#nums = hf1['label'].shape[0]
#total_nums = nums * 2

nums_1 = hf2['label'].shape[0]
nums_2 = hf3['label'].shape[0]

total_nums = nums_1 + nums_2


print(nums_1)
print(nums_2)
print(total_nums)


#del hf['data']
#del hf['label']

data = hf.create_dataset('data', (total_nums, 3, data_shape, data_shape), 'float64')
label = hf.create_dataset('label', (total_nums, 3, label_shape, label_shape), 'float64')


print(hf.keys())


data[:nums_1, :, :, :] = hf2_data
data[nums_1:nums_2, :, :, :] = hf3_data
#data[nums*2:, :, :, :] = hf3_data

label[:nums_1, :, :, :] = hf2_label
label[nums_1:nums_2, :, :, :] = hf3_label
#label[nums*2:, :, :, :] = hf3_label

print(hf.keys())


