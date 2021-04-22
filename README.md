# FedAvg
This is a implemention of FedAvg in paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

# How to run the codes?

At first, you should creat two dirs called 'logs' and 'checkpoints', then you can cd into 'src', and run FedAvg on iid cifar10 and mnist
```shell
python3 federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=300 --lr=0.01 --local_ep=5 --local_bs=32

python3 federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32

```

noniid cifar10 and mnist:
```shell
python3 federated_main.py --model=cnn --dataset=cifar --iid=0 --epochs=300 --lr=0.01 --local_ep=5 --local_bs=32

python3 federated_main.py --model=cnn --dataset=mnist --iid=0 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32

```
For best test acc：

|- |CIFAR10  |MNIST |
| ------------- | ------------- |------------ |
| IID  | 68.70%  | 98.87% |
| Non-IID  | 68.11%  | 98.05%|


test acc curve(for cifar10 both in iid and noniid setting):

![企业微信截图_20210422101626](https://user-images.githubusercontent.com/33173674/115645971-e97f9e00-a353-11eb-9fad-8b5e18fc24ea.png)
