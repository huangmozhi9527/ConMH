# model
model_name = 'conmh'
use_checkpoint = None
feature_size = 4096
hidden_size = 256
max_frames = 25
nbits = 64
transformer_type = 'small'

# dataset
dataset = 'fcv'
workers = 1
batch_size = 512
mask_prob = 0.75

# train
seed = 1
num_epochs = 805
a = 1.0
temperature = 0.5
tau_plus = 0.1
train_num_sample = 45585

# test
test_batch_size = 128
test_num_sample = 45600

# optimizer
optimizer_name = 'Adam'
schedule = 'StepLR'
lr = 1e-4
min_lr = 1e-5
lr_decay_rate = 20
lr_decay_gamma = 0.9
weight_decay = 0.0

# path
data_root = '/data/dataset/fcv/'
home_root = '/data/conmh/'

train_feat_path = [data_root + 'fcv_train_feats.h5']
test_feat_path = [data_root + 'fcv_test_feats.h5']
label_path = [data_root + 'fcv_test_labels.mat']

save_dir = home_root + 'saved_model/' + model_name + '_' + dataset
file_path = save_dir + '_' + str(nbits) + 'bit'