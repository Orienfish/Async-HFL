# Usage:
#   Run FL on various dataset using various servers and client selection algorithms, e.g.,
#     bash ./run_exp.sh MNIST noniid
#   Dataset choices: MNIST, FashionMNIST, CIFAR-10, FEMNIST, Shakespeare, Synthetic, HAR, HPWREN
#   Data distribution choices: iid, noniid
#     (Note: the FEMNIST, Shakespeare, Synthetic and HPWREN datasets are naturally noniid,
#     so only noniid selection is available)

cd ..;
for trial in 0 1 2
do
  for sel in coreset_v1 coreset_v2 coreset_v3 coreset_v4
  do
    for asso in gurobi_v1 gurobi_v2
    do
      python3.7 run.py --config=configs/"$1"/async_"$2".json --selection="$sel" --association="$asso" --trial="$trial"
    done
  done
done
