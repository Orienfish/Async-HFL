# Usage:
#   Run FL on various dataset using various servers and client selection algorithms, e.g.,
#     bash ./run_exp.sh MNIST noniid trial#
#   Dataset choices: MNIST, FashionMNIST, CIFAR-10, FEMNIST, Shakespeare, Synthetic, HAR, HPWREN
#   Data distribution choices: iid, noniid
#     (Note: the FEMNIST, Shakespeare, Synthetic and HPWREN datasets are naturally noniid,
#     so only noniid selection is available)

alpha=$3
for trial in 0 1 2
do
  for cs in coreset_v1 coreset_v2 coreset_v3 coreset_v4
  do
    for ca in gurobi_v1 gurobi_v2
    do
      python3.7 run.py --config=configs/"$1"/async_"$2".json --delay_mode=nycmesh \
        --selection=$cs --association=$ca --cs_alpha=$alpha \
        --trial="$trial" | tee log_"$1"_nycmesh_async_"$cs"_"$alpha"_"$ca"_"$trial"
    done
  done

  python3.7 run.py --config=configs/"$1"/async_"$2".json --delay_mode=nycmesh \
    --selection=high_loss_first --association=random \
    --trial="$trial" | tee log_"$1"_nycmesh_async_high_loss_first_0.2_random_"$trial"
done
