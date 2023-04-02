# Usage:
#   Run FL baselines on various dataset using various servers and client selection algorithms, e.g.,
#     bash ./run_exp.sh MNIST noniid
#   Dataset choices: MNIST, FashionMNIST, CIFAR-10, FEMNIST, Shakespeare, Synthetic, HAR, HPWREN
#   Data distribution choices: iid, noniid
#     (Note: the FEMNIST, Shakespeare, Synthetic and HPWREN datasets are naturally noniid,
#     so only noniid selection is available)

for trial in 0
do
  # Pure Async baselines
  for sel in random
  do
    python3.7 run.py --config=configs/HAR/pureasync_noniid.json --delay_mode=nycmesh --selection="$sel" --trial="$trial"
  done

  # Async baselines
  for sel in short_latency_first
  do
    python3.7 run.py --config=configs/HAR/async_noniid.json --delay_mode=nycmesh --selection="$sel" --trial="$trial"
  done

  # semi-async baselines
  for p in 50 100 150 200
  do
    python3.7 run.py --config=configs/HAR/semiasync_noniid.json --delay_mode=nycmesh --selection=random --trial="$trial" \
      --semi_period $p
  done
done
