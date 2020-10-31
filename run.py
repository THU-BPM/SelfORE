from selfore import SelfORE
import yaml


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    selfore = SelfORE(config)
    selfore.start()


if __name__ == "__main__":
    main()
