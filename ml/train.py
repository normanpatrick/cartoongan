import argparse
import networks

def main(args):
    generator = networks.Generator()
    print(generator)
    discriminator = networks.Discriminator()
    print(discriminator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Do something interesting")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(vars(args))
    main(args)
