import numpy as np
import re

def parse_nlx_events(filename, pattern):
    timestamps = []
    with open(filename, 'r') as file:
        for line in file:
            if pattern in line:
                match = re.search(r'\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', line)
                if match:
                    timestamps.append(np.float64(match.group()))
    return np.array(timestamps, dtype=np.float64)


def mocap_events(filename, start, stop):
    events_start = parse_nlx_events(filename=filename, pattern=start)
    events_stop  = parse_nlx_events(filename=filename, pattern=stop)
    return (
        np.concatenate(
            (events_start[:,np.newaxis],
             events_stop[:,np.newaxis]),
            axis=1)
        )/1000
