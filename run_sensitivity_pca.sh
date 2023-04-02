# Usage:
#   Run FL on various dataset using various servers and client selection algorithms, e.g.,
#     bash ./run_exp.sh MNIST noniid trial#
#   Dataset choices: MNIST, FashionMNIST, CIFAR-10, FEMNIST, Shakespeare, Synthetic, HAR, HPWREN
#   Data distribution choices: iid, noniid
#     (Note: the FEMNIST, Shakespeare, Synthetic and HPWREN datasets are naturally noniid,
#     so only noniid selection is available)

for trial in 0 1 2
do
  for cs in coreset_v1
  do
    for ca in gurobi_v1
    do
      for pca in 50 100 150
      do
        python3.7 run.py --config=configs/"$1"/async_"$2".json --delay_mode=nycmesh \
          --selection=$cs --association=$ca --pca_dim=$pca \
          --trial="$trial" | tee log_"$1"_nycmesh_async_"$cs"_"$ca"_"$pca"_"$trial"
      done
      python3.7 run.py --config=configs/"$1"/async_"$2".json --delay_mode=nycmesh \
        --selection=$cs --association=$ca \
        --trial="$trial" | tee log_"$1"_nycmesh_async_"$cs"_"$ca"_"$pca"_"$trial"
    done
  done
done
