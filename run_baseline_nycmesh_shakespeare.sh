# Usage:
#   Run FL baselines on various dataset using various servers and client selection algorithms, e.g.,
#     bash ./run_exp.sh MNIST noniid
#   Dataset choices: MNIST, FashionMNIST, CIFAR-10, FEMNIST, Shakespeare, Synthetic, HAR, HPWREN
#   Data distribution choices: iid, noniid
#     (Note: the FEMNIST, Shakespeare, Synthetic and HPWREN datasets are naturally noniid,
#     so only noniid selection is available)

for trial in 0 1 2
do
  # Sync baselines
  for sel in oort
  do
    python3.7 run.py --config=configs/"$1"/sync_"$2".json --delay_mode=nycmesh --selection="$sel" --trial="$trial" \
      | tee log_"$1"_nycmesh_sync_"$sel"_"$trial"
  done

  # RFL-HA baseline
  python3.7 run.py --config=configs/"$1"/rflha_"$2".json --delay_mode=nycmesh --selection=random --trial="$trial" \
    | tee log_"$1"_nycmesh_rflha_"$sel"_"$trial"

  # semi-async baselines
  for semi_period in 200 250 300
  do
    python3.7 run.py --config=configs/"$1"/semiasync_"$2".json --delay_mode=nycmesh --selection=random --trial="$trial" \
      --semi_period=$semi_period | tee log_"$1"_nycmesh_semiasync_"$sel"_"$trial"
  done
done
