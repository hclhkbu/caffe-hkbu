epoch_size=60000
gpu_count="${gpu_count:-1}"
learning_rate="${learning_rate:-0.05}"
batch_size="${batch_size:-1024}"
num_epochs="${num_epochs:-40}"
network_name=fcn5sy${num_hidden}l
batches_per_epoch=`awk "BEGIN {print int( (${epoch_size}+${batch_size}-1)/${batch_size} )}"` #50000/32 
max_iter=`awk "BEGIN {print int( ${batches_per_epoch}*${num_epochs} )}"` #50000/32 * 40
display_interval=1 #`awk "BEGIN {print int( ${batches_per_epoch}/4 )}"` # 50000 / 32 
test_interval=`awk "BEGIN {print int( ${batches_per_epoch} )}"` # 50000/32
device=GPU
model_file=${network_name}-b${batch_size}.prototxt
cp ${network_name}.prototxt ${model_file}
sed -i -e "s/BATCHSIZE/${batch_size}/g" ${model_file}
sed -i -e "s|HOME|${HOME}|g" ${model_file}
