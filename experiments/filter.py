from argparse import ArgumentParser
import json


def main():
    parser = ArgumentParser()
    parser.add_argument("--files", type=str, nargs='+', required=True)
    parser.add_argument("--estimator", type=str, required=True)
    parser.add_argument("--change", type=str, required=True)
    parser.add_argument("--aggregator", type=str, required=True)
    parser.add_argument("--postfix", type=str, default='/log.json')
    args = parser.parse_args()

    for file in args.files:
        try:
            with open(file + '/args.json', 'rt') as f:
                data = json.load(f)
                if data['estimator'] != args.estimator:
                    continue
                if data['aggregator'] != args.aggregator:
                    continue
                if data['change'] != args.change:
                    continue
                print(file + args.postfix, end=' ')
        except:
            pass


if __name__ == "__main__":
    main()
