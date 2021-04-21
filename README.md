# FedAvg
This is a implementtion of FedAvg in paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

# How to run the codes?

At first, you should creat two dir called 'logs' and 'checkpoints', then you can cd into 'src', and run FedAvg on iid cifar10 and mnist
```shell
python3 federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=300 --lr=0.01 --local_ep=5 --local_bs=32

python3 federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32

```

noniid cifar10 and mnist:
```shell
python3 federated_main.py --model=cnn --dataset=cifar --iid=0 --epochs=300 --lr=0.01 --local_ep=5 --local_bs=32

python3 federated_main.py --model=cnn --dataset=mnist --iid=0 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32

```
