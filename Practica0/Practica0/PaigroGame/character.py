class Person:
    def __init__(self, nm, hp, at, sh):
        self.name = nm
        self.health = hp
        self.maxHealth= hp
        self.atack = at
        self.shield = sh
        self.maxShield = sh

    def receiveDamage(enemyDamage):
        auxShield = shield
        excessDamage = abs(auxShield - enemyDamage)
        if(auxShield < 0):
            health -= excessDamage
        else:
            shield -= enemyDamage

    def restoreShield(quantity):
        if(shield + quantity >= maxShield):
            shield = maxShield
        else: shield += quantity