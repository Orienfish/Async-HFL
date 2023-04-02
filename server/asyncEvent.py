class asyncEvent(object):
    """Basic async event."""
    def __init__(self, client, download_round, download_time, delay):
        self.client = client
        self.download_round = download_round
        self.download_time = download_time
        self.aggregate_time = download_time + delay

    def update(self, download_round, download_time, delay):
        self.download_epoch = download_round
        self.download_time = download_time
        self.aggregate_time = download_time + delay

    def __eq__(self, other):
        return self.aggregate_time == other.aggregate_time

    def __ne__(self, other):
        return self.aggregate_time != other.aggregate_time

    def __lt__(self, other):
        return self.aggregate_time < other.aggregate_time

    def __le__(self, other):
        return self.aggregate_time <= other.aggregate_time

    def __gt__(self, other):
        return self.aggregate_time > other.aggregate_time

    def __ge__(self, other):
        return self.aggregate_time >= other.aggregate_time