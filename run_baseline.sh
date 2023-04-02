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
  for sel in random high_loss_first divfl tier
  do
    python3.7 run.py --config=configs/"$1"/sync_"$2".json --selection="$sel" --trial="$trial"
  done

  # RFL-HA baseline
  python3.7 run.py --config=configs/"$1"/rflha_"$2".json --selection=random --trial="$trial"

  # Async baselines
  for sel in random high_loss_first
  do
    python3.7 run.py --config=configs/"$1"/async_"$2".json --selection="$sel" --trial="$trial"
  done

  # semi-async baselines
  for sel in random
  do
    python3.7 run.py --config=configs/"$1"/semiasync_"$2".json --selection="$sel" --trial="$trial"
  done
done
