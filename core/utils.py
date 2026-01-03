from statistics import mean, stdev


def clamp_num_rounds(rounds: int) -> int:
    # minimum of 2 rounds needed to calculate standard deviation
    return max(2, min(rounds, 10))


def reject_outliers(data: list[float], m: int = 2) -> list[float]:
    mean_val = mean(data)
    stdev_val = stdev(data)
    return [x for x in data if abs(x - mean_val) < m * stdev_val]
