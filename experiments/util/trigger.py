class LogscaleTrigger:
    def __init__(self):
        self._increment = 1
        self._next = 0

    def __call__(self, trainer):
        """
        :param trainer: The trainer
        :type trainer: chainer.training.Trainer

        :return: True if the trigger fires, false otherwise
        :rtype: bool
        """
        updater = trainer.updater
        t = updater.iteration

        if len(f'{t}'.replace('0', '')) <= 1:
            return True
        else:
            return False
