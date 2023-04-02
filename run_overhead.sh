
for dataset in MNIST FashionMNIST CIFAR-10 Shakespeare HAR HPWREN
do
  python run.py --config=configs/"$dataset"/async_noniid.json --selection=coreset --association=gurobi --trial=0
done