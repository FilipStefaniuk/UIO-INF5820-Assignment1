from data import Data
from model import Model
from estimator import Estimator
from utils import parse_args, parse_config, save_results
import os

if __name__ == '__main__':
    args = parse_args()
    config = parse_config(args.config) if args.config else {}

    # Create output directory and add path to config.
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    config['output_dir'] = args.output

    # Build model.
    model = Model(**config)
    model.model.summary()

    # Load dataset.
    if not os.path.exists(args.data):
        raise ValueError
    print('loadining data...')
    data = Data(args.data, **config)

    # Create estimator and train model,
    # save metrics from training to json file in output dir.
    estimator = Estimator(model, data)
    metrics = estimator.train(**config)
    save_results(os.path.join(args.output, 'training_metrics.json'), metrics)
    # print(metrics)

    # Evaluate trained model on validation data,
    # save metrics to json file in output dir.
    metrics = estimator.evaluate(data.x_val, data.y_val, **config)
    save_results(os.path.join(args.output, 'validation_metrics.json'), metrics)
    print(metrics)

    # If test flag was present in command line arguments,
    # evaluate model on test data and save metrics to json file.
    if args.test:
        metrics = estimator.evaluate(data.x_test, data.y_test, **config)
        save_results(os.path.join(args.output, 'test_metrics.json'), metrics)
        # print(metrics)
