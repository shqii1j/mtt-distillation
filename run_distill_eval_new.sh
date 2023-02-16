'''init buffer'''
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca \
--buffer_path=buffer --data_path=dataset

'''first distill'''
python distill_eval_new.py --dataset=CIFAR10 --ipc=16 --syn_steps=30 --expert_epochs=2 --data_path=dataset --zca \
--buffer_path=buffer/CIFAR10 --intervals=0-20 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001

'''buffer on S1'''
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca \
--data_path=dataset --syn_image_path=logged_files \
--run_name=enchanting-rocket-469 --files_name=0-20 --lrs_net=0.035 --reparam_syn

'''sencond distill'''
python distill_eval_new.py --dataset=CIFAR10 --ipc=16 --syn_steps=30 --expert_epochs=2 --data_path=dataset --zca \
--buffer_path=./logged_files/CIFAR10/enchanting-rocket-469/0-20/buffer \
--intervals=0-20 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.01 \
--run_name=enchanting-rocket-469 --files_name=0-20 --lrs_net=0.035 --reparam_syn

'''buffer on S2'''
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca \
--data_path=dataset --syn_image_path=logged_files \
--run_name=enchanting-rocket-469 --files_name=0-20,0-20+alight-fireworks-496 --lrs_net=0.035,0.01 --reparam_syn

'''third distill'''
python distill_eval_new.py --dataset=CIFAR10 --ipc=16 --syn_steps=30 --expert_epochs=2 --data_path=dataset --zca \
--buffer_path=./logged_files/CIFAR10/enchanting-rocket-469/0-20+alight-fireworks-496/buffer \
--intervals=0-10 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.01 \
--run_name=enchanting-rocket-469 --files_name=0-20,0-20+alight-fireworks-496 --lrs_net=0.035,0.01 --reparam_syn





