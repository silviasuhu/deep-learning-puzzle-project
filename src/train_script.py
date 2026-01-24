import argparse


def main():
    # TODO


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size_x", type=int, default=6)
    ap.add_argument("-batch_size_y", type=int, default=6)
    ap.add_argument("-steps", type=int, default=300)

    args = ap.parse_args()
    print(args)
    main()
