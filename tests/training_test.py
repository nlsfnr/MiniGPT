from minigpt import Config, train


def test_train_and_iter(config: Config) -> None:
    events = iter(train(config=config, seed=0))
    for _ in range(3):
        next(events)
