import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to the input file.")
    parser.add_argument("--output", help="Path to the output file.")

    args = parser.parse_args()    
    return args


def main():
    args = parse_arguments()
    
    print(args)
    
    return 0


if __name__ == "__main__":
    exit(main())

