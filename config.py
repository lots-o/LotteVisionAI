#config
batch_size=64
lr=4e-04
weight_decay=1e-2
n_epochs=20
k_fold=3
model_name='efficientnet-b3'

log_dir='Log/'
root_dir='train/'
save_dir='weights/'

#Early stopping
patience=5

#lr_scheduler
steplr_step_size=10
steplr_gamma=0.1
