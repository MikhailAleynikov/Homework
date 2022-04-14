
from logging import exception

class LowFuelError():
    Exception ("Низкий уровень топлива")


class NotEnoughFuel():
    Exception ("Недостаточно топлива")


class CargoOverload():
    Exception ("Перегруз")
    