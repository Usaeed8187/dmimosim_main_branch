from mobility import random_waypoint

if __name__ == "__main__":
    rw = random_waypoint(4, dimensions=(100, 100), velocity=(0.1, 1.0), wt_max=1.0)

    index = 0
    for xy in rw:
        index += 1
        print(xy)

        if index == 1:
            break

    index = 0
    for xy in rw:
        index += 1
        print(xy)

        if index == 10:
            break