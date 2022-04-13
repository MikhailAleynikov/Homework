
from logging import exception

class LowFuelError():
    exception ("Низкий уровень топлива", 10)


class NotEnoughFuel():
    exception ("Недостаточно топлива", 0)


class CargoOverload():
    exception ("Перегруз", 100)
    