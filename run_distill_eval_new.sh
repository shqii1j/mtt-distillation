python distill_new.py --dataset=CIFAR10 --ipc=25 --syn_steps=30 --expert_epochs=2 \
--zca --buffer_path=./logged_files/CIFAR10/clear-cloud-72/0-12/buffer --data_path=dataset --intervals=0-5 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 \
--run_name=clear-cloud-72 --file_name=0-12