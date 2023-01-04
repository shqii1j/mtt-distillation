python distill_new.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_end_epoch=5 \
--zca --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 \
--buffer_path=buffer --data_path=dataset
python buffer.py