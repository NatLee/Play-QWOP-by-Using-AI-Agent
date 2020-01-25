import ast
import logging
import argparse

from QWOP import QWOP

from modelUtils import CNNModel, A2CAgent

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=ast.literal_eval, default=True, help='Training the model. The input should be either "True" or "False".')
    parser.add_argument('--model_path', type=str, default=None, help='Model weights path. If you assign a path, the model will load the weights.')
    parser.add_argument('--model_save_path', type=str, default=None, help='Model save path. If you don\'t specific, the model won\'t be saved.')
    parser.add_argument('--test_env', type=ast.literal_eval, default=False, help='Use random action to test QWOP environment. The input should be either "True" or "False".')
    parser.add_argument('--chrome_driver_path', type=str, default='./webdriver/chromedriver', help='Specific your chrome driver path to run the selenium.')
    args = parser.parse_args()

    # start a environment
    env = QWOP(args.chrome_driver_path)
    env.start()

    if not args.test_env:
        # build a model
        model = CNNModel(num_actions=5)

        # test environment with model
        model.action_value(env.reset()[None, :])

        if args.model_path is not None:
            logging.info('Now loading model weights...')
            model.load_weights(args.model_path)

        # start a agent
        agent = A2CAgent(model)
        
        if args.train:
            # training
            logging.info('Start training ...')
            rewards_history = agent.train(env)
            logging.info('Training Completed')

        # testing
        logging.info('Start testing ...')
        total_rewards = agent.test(env)
        logging.info(f'Total Episode Reward: {total_rewards}')

        if args.model_save_path is not None:
            logging.info('Now saving model weights...')
            model.save_weights(args.model_save_path)

    else:
        while True:
            observation, reward, done = env.execute_action("random_go")
            if done:
                logging.info(f'reward {reward}')
                env.reset()